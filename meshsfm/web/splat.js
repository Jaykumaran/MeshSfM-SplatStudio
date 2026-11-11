'use strict';
// Gaussian-splat layer for the 3D view: streams the splat tiles near the camera (budgeted), keeps them in GPU
// textures, depth-sorts them in a worker and draws them after the mesh (depth-tested against it, so mesh and splat
// can be shown together, alone, or not at all).
const SPLAT = (() => {
  let gl = null, prog = null, tex = {}, idxBuf = null, vao = null, n = 0, layers = {}, loaded = new Map(), version = 0;
  let sorted = null, sortBusy = false, lastSortKey = '', positions = null, opac = null, lastErr = 0, drawn = 0;
  let enabled = true, budget = 3_000_000, radius = 150;
  const worker = new Worker(URL.createObjectURL(new Blob([`
    let pos = null;
    onmessage = e => {
      if (e.data.pos) { pos = e.data.pos; return; }
      const {f, c, id} = e.data; const n = pos.length / 3;
      const depth = new Int32Array(n); let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < n; i++) { const d = (pos[3*i]-c[0])*f[0] + (pos[3*i+1]-c[1])*f[1] + (pos[3*i+2]-c[2])*f[2]; const q = (d * 1024) | 0; depth[i] = q; if (q < mn) mn = q; if (q > mx) mx = q; }
      const B = 65536, scale = (B - 1) / Math.max(1, mx - mn); const cnt = new Uint32Array(B);
      const key = new Uint16Array(n);
      for (let i = 0; i < n; i++) { const k = ((mx - depth[i]) * scale) | 0; key[i] = k; cnt[k]++; }   // far first
      for (let i = 1; i < B; i++) cnt[i] += cnt[i - 1];
      const out = new Uint32Array(n);
      for (let i = n - 1; i >= 0; i--) out[--cnt[key[i]]] = i;
      postMessage({out, id}, [out.buffer]);
    };`], {type: 'text/javascript'})));
  let pend = null;            // next tile set: drawn only once its first sort is ready (the current one keeps drawing)
  worker.onmessage = e => {
    if (e.data.id === version) {
      if (pend) {
        for (const k of Object.keys(tex)) gl.deleteTexture(tex[k]);
        tex = pend.tex; n = pend.n; positions = pend.positions; opac = pend.opac; pend = null;
      }
      sorted = e.data.out; upload();
    }
    sortBusy = false; V3.redraw();
  };

  const VS = `#version 300 es
  precision highp float; precision highp int;
  uniform highp sampler2D uPos; uniform highp sampler2D uCovA; uniform highp sampler2D uCovB; uniform lowp sampler2D uCol;
  uniform mat4 uView, uProj; uniform vec2 uFocal, uViewport;
  in vec2 aCorner; in uint aIndex;
  out vec4 vCol; out vec2 vPos;
  ivec2 at(uint i) { return ivec2(int(i & 2047u), int(i >> 11)); }
  void main() {
    vec3 p = texelFetch(uPos, at(aIndex), 0).xyz;
    vec4 cam = uView * vec4(p, 1.0);
    vec4 clip = uProj * cam;
    float clipw = 1.2 * clip.w;
    if (cam.z > -0.1 || clip.x < -clipw || clip.x > clipw || clip.y < -clipw || clip.y > clipw) { gl_Position = vec4(0, 0, 2, 1); return; }
    vec3 a = texelFetch(uCovA, at(aIndex), 0).xyz, b = texelFetch(uCovB, at(aIndex), 0).xyz;
    mat3 S = mat3(a.x, a.y, a.z, a.y, b.x, b.y, a.z, b.y, b.z);
    float z = cam.z;
    mat3 J = mat3(uFocal.x / -z, 0.0, 0.0,  0.0, uFocal.y / -z, 0.0,  uFocal.x * cam.x / (z * z), uFocal.y * cam.y / (z * z), 0.0);
    mat3 W = mat3(uView);
    mat3 T = J * W;
    mat3 C = T * S * transpose(T);
    float c00 = C[0][0] + 0.3, c01 = C[0][1], c11 = C[1][1] + 0.3;
    float mid = 0.5 * (c00 + c11), rad = length(vec2(0.5 * (c00 - c11), c01));
    float l1 = mid + rad, l2 = max(mid - rad, 0.1);
    if (l2 < 0.0) { gl_Position = vec4(0, 0, 2, 1); return; }
    vec2 d = normalize(vec2(c01, l1 - c00)); if (abs(c01) < 1e-9) d = c00 >= c11 ? vec2(1, 0) : vec2(0, 1);
    vec2 v1 = min(sqrt(2.0 * l1), 1024.0) * d, v2 = min(sqrt(2.0 * l2), 1024.0) * vec2(d.y, -d.x);
    vec4 col = texelFetch(uCol, at(aIndex), 0);
    vCol = col; vPos = aCorner;
    vec2 ndc = clip.xy / clip.w;
    gl_Position = vec4(ndc + aCorner.x * v1 / uViewport * 2.0 + aCorner.y * v2 / uViewport * 2.0, clip.z / clip.w, 1.0);
  }`;
  const FS = `#version 300 es
  precision highp float;
  in vec4 vCol; in vec2 vPos; out vec4 o;
  void main() {
    float A = -dot(vPos, vPos);
    if (A < -4.0) discard;
    float B = exp(A) * vCol.a;
    if (B < 1.0 / 255.0) discard;
    o = vec4(vCol.rgb * B, B);
  }`;

  function init(g) {
    gl = g;
    const mk = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(p, 0, 'aCorner'); gl.bindAttribLocation(p, 1, 'aIndex'); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
    prog = {p, u};
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
    gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
  }

  function makeTex(fmt, data, comps, type, w, h) {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, comps, type, data); return t;
  }

  function rebuild() {
    // concatenate the loaded tiles -> textures (2048 wide)
    let total = 0; for (const t of loaded.values()) total += t.n;
    version++;
    if (pend) { for (const k of Object.keys(pend.tex)) gl.deleteTexture(pend.tex[k]); pend = null; }
    if (!total) { for (const k of Object.keys(tex)) gl.deleteTexture(tex[k]); tex = {}; n = 0; sorted = null; V3.redraw(); return; }
    const N = total, W = 2048, H = Math.ceil(N / W);
    const P = new Float32Array(W * H * 4), A = new Float32Array(W * H * 4), B = new Float32Array(W * H * 4), C = new Uint8Array(W * H * 4);
    const positions = new Float32Array(N * 3), opac = new Uint8Array(N);
    let o = 0; const O = V3.O;
    for (const t of loaded.values()) {
      for (let i = 0; i < t.n; i++, o++) {
        const x = t.pos[3*i] - O[0], y = t.pos[3*i+1] - O[1], z = t.pos[3*i+2] - O[2];
        P[4*o] = x; P[4*o+1] = y; P[4*o+2] = z; positions[3*o] = x; positions[3*o+1] = y; positions[3*o+2] = z;
        A[4*o] = t.cov[6*i]; A[4*o+1] = t.cov[6*i+1]; A[4*o+2] = t.cov[6*i+2];
        B[4*o] = t.cov[6*i+3]; B[4*o+1] = t.cov[6*i+4]; B[4*o+2] = t.cov[6*i+5];
        C[4*o] = t.rgba[4*i]; C[4*o+1] = t.rgba[4*i+1]; C[4*o+2] = t.rgba[4*i+2]; C[4*o+3] = t.rgba[4*i+3]; opac[o] = t.rgba[4*i+3];
      }
    }
    const nt = {pos: makeTex(gl.RGBA32F, P, gl.RGBA, gl.FLOAT, W, H), a: makeTex(gl.RGBA32F, A, gl.RGBA, gl.FLOAT, W, H),
                b: makeTex(gl.RGBA32F, B, gl.RGBA, gl.FLOAT, W, H), c: makeTex(gl.RGBA8, C, gl.RGBA, gl.UNSIGNED_BYTE, W, H)};
    pend = {tex: nt, n: N, positions, opac};
    worker.postMessage({pos: positions.slice()});
    const cam = V3.cam, cp = Math.cos(cam.pitch), f = [cp * Math.cos(cam.yaw), cp * Math.sin(cam.yaw), Math.sin(cam.pitch)];
    sortBusy = true; lastSortKey = ''; worker.postMessage({f, c: cam.p.slice(), id: version});
  }

  function upload() {
    if (!sorted || sorted.length !== n) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ARRAY_BUFFER, sorted, gl.DYNAMIC_DRAW);
  }

  function requestSort(cam) {
    if (!n || sortBusy || pend) return;
    const cp = Math.cos(cam.pitch), f = [cp * Math.cos(cam.yaw), cp * Math.sin(cam.yaw), Math.sin(cam.pitch)];
    const key = cam.p.map(v => v.toFixed(1)).join(',') + f.map(v => v.toFixed(2)).join(',');
    if (key === lastSortKey && sorted) return;
    lastSortKey = key; sortBusy = true; worker.postMessage({f, c: cam.p, id: version});
  }

  let streaming = false;
  async function stream() {
    if (!enabled || !gl || !V3.started || streaming) return;
    streaming = true; try { await streamOnce(); } finally { streaming = false; }
  }
  async function streamOnce() {
    const O = V3.O, c = V3.cam.p, want = [];
    for (const [lid, L] of Object.entries(layers)) {
      if (!L.visible) continue;
      for (const t of L.meta.tiles) {
        const cx = (t.bbox[0] + t.bbox[3]) / 2 - O[0], cy = (t.bbox[1] + t.bbox[4]) / 2 - O[1];
        const d = Math.hypot(cx - c[0], cy - c[1]), key = lid + '/' + t.name;
        if (d < (loaded.has(key) ? radius * 1.3 : radius)) want.push({key, d: loaded.has(key) ? d * 0.8 : d, n: t.n});   // hysteresis
      }
    }
    want.sort((a, b) => a.d - b.d);
    let sum = 0; const keep = new Set();
    for (const w of want) { if (sum + w.n > budget * (loaded.has(w.key) ? 1.15 : 1)) continue; sum += w.n; keep.add(w.key); }
    let changed = false;
    for (const k of [...loaded.keys()]) if (!keep.has(k)) { loaded.delete(k); changed = true; }
    for (const k of keep) {
      if (loaded.has(k)) continue;
      const buf = await (await fetch('/splat/tile/' + k)).arrayBuffer();
      const m = buf.byteLength / (12 + 24 + 4);
      loaded.set(k, {n: m, pos: new Float32Array(buf, 0, 3 * m), cov: new Float32Array(buf, 12 * m, 6 * m), rgba: new Uint8Array(buf, 36 * m, 4 * m)});
      changed = true;
    }
    if (changed) rebuild();
    if ($('splatInfo')) $('splatInfo').textContent = n ? `${(n / 1e6).toFixed(2)} M gaussians loaded (${loaded.size} tiles)` : 'no splat near the camera';
  }

  function draw(g, s) {
    if (!prog) init(g);
    if (!enabled || !n) return;
    requestSort(s.cam);
    if (!sorted || sorted.length !== n) return;
    gl.useProgram(prog.p);
    gl.uniformMatrix4fv(prog.u.uView, false, s.view); gl.uniformMatrix4fv(prog.u.uProj, false, s.proj);
    const fy = s.H / 2 / Math.tan(s.fov / 2);
    gl.uniform2f(prog.u.uFocal, fy, fy); gl.uniform2f(prog.u.uViewport, s.W, s.H);
    [['uPos', tex.pos], ['uCovA', tex.a], ['uCovB', tex.b], ['uCol', tex.c]].forEach(([nm, t], i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(prog.u[nm], i); });
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    gl.bindVertexArray(vao); gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, n); lastErr = gl.getError(); drawn++;
    gl.bindVertexArray(null); gl.depthMask(true); gl.disable(gl.BLEND); gl.activeTexture(gl.TEXTURE0);
  }

  function pick(px, py) {     // nearest fairly opaque gaussian along the ray through a screen point -> {p (rel O), dist}
    if (!enabled || !n || !positions) return null;
    const r = V3.rayDir(px, py), c = V3.cam.p; let best = Infinity, bi = -1;
    for (let i = 0; i < n; i++) {
      if (opac[i] < 90) continue;
      const vx = positions[3*i] - c[0], vy = positions[3*i+1] - c[1], vz = positions[3*i+2] - c[2];
      const t = vx * r[0] + vy * r[1] + vz * r[2]; if (t < 0.3 || t >= best) continue;
      const d2 = vx * vx + vy * vy + vz * vz - t * t, tol = 0.006 * t + 0.04;
      if (d2 < tol * tol) { best = t; bi = i; }
    }
    return bi < 0 ? null : {p: [positions[3*bi], positions[3*bi+1], positions[3*bi+2]], dist: best};
  }

  function setLayers(proj) {
    const old = layers; layers = {};
    for (const L of proj.layers) if (L.kind === 'splat' && proj.splats[L.id]) layers[L.id] = {visible: L.visible, meta: proj.splats[L.id]};
    for (const k of [...loaded.keys()]) if (!layers[k.split('/')[0]] || !layers[k.split('/')[0]].visible) loaded.delete(k);
    rebuild(); stream();
  }

  V3.drawHooks.push(draw);
  setInterval(() => { if (TAB === '3d') stream(); }, 800);
  return {debug: () => ({n, sorted: sorted ? sorted.length : -1, version, sortBusy, tiles: loaded.size, enabled, prog: !!prog, lastErr, drawn, texH: n ? Math.ceil(n / 2048) : 0, maxTex: gl && gl.getParameter(gl.MAX_TEXTURE_SIZE)}), setLayers, stream, pick, get enabled() { return enabled; }, set enabled(v) { enabled = v; V3.redraw(); },
          set budget(v) { budget = v; stream(); }, set radius(v) { radius = v; stream(); }, get count() { return n; }};
})();
