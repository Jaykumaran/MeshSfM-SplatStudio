'use strict';
// 3D view: streams the collision mesh in 50 m chunks (full detail near the camera, coarse further out),
// fly / drive navigation, picking (depth pass), lane + route lines, mesh-edit masks and patches.
const V3 = (() => {
  const cv = $('c3'), hud = $('hud'), hx = hud.getContext('2d');
  const gl = cv.getContext('webgl2', {antialias: true, alpha: false, preserveDrawingBuffer: false});
  let W = 0, H = 0, DPR = 1;
  let split = null;          // side by side (Align): {left: 'splat' | 'mesh', right: ...} - same camera, one kind per half
  let O = [0, 0, 0];                 // mesh origin (absolute z-up); all GPU coordinates are relative to it
  let chunkMeta = [];
  const chunks = new Map();          // key -> chunk
  const patches = new Map();         // pid -> patch
  const cam = {p: [0, 0, 5], yaw: 0, pitch: -0.15};
  const drive = {on: false, s: 0, v: 0, look: 0, pitch: -0.05};
  let shade = 0, needDraw = true, started = false, version = 0;
  const keys = new Set();
  let tool = 'nav';
  const SKY = [0.62, 0.70, 0.78];
  const LOD_R = 900;
  let ROUTE3 = [];                   // [x, y, z rel O, arc]
  let lineBuf = null, lineCount = 0;
  let pick = {fbo: null, tex: null, depth: null, w: 0, h: 0, valid: false};
  let onStatus = [], hoverInfo = null, freezePick = false, propTool = null;
  const drawHooks = [];

  if (!gl) { $('hint3').textContent = 'WebGL2 is not available in this browser.'; }

  // --------------------------------------------------------------------------------- shaders
  const VS = `#version 300 es
  layout(location=0) in vec3 aPos; layout(location=1) in vec3 aCol;
  uniform mat4 uMVP; out vec3 vCol; out vec3 vPos;
  void main(){ vCol = aCol; vPos = aPos; gl_Position = uMVP * vec4(aPos, 1.0); }`;
  const FS = `#version 300 es
  precision highp float;
  in vec3 vCol; in vec3 vPos;
  uniform int uMode; uniform vec4 uOver; uniform vec4 uTint; uniform vec3 uCam; uniform float uFog; uniform vec3 uSky;
  out vec4 o;
  void main(){
    vec3 c = vCol;
    if (uMode < 3) {
      vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
      float l = 0.30 + 0.70 * abs(dot(n, normalize(vec3(0.35, 0.25, 0.90))));
      if (uMode == 1) c = vec3(0.80, 0.80, 0.78) * l;
      else if (uMode == 2) c = vCol * (0.55 + 0.45 * l);
    }
    c = mix(c, uTint.rgb, uTint.a);
    if (uOver.a > 0.0) c = mix(c, uOver.rgb, uOver.a);
    float d = length(vPos - uCam);
    float f = uFog > 0.0 ? clamp((d - uFog * 0.35) / (uFog * 0.65), 0.0, 1.0) * 0.85 : 0.0;
    o = vec4(mix(c, uSky, f), 1.0);
  }`;
  const FS_PICK = `#version 300 es
  precision highp float;
  in vec3 vCol; in vec3 vPos; uniform vec3 uCam; uniform float uFar; out vec4 o;
  void main(){
    float d = clamp(length(vPos - uCam) / uFar, 0.0, 0.9999);
    vec3 e = fract(d * vec3(1.0, 255.0, 65025.0));
    e -= e.yzz * vec3(1.0 / 255.0, 1.0 / 255.0, 0.0);
    o = vec4(e, 1.0);
  }`;
  function program(vs, fs) {
    const mk = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const n = gl.getActiveUniform(p, i).name; u[n] = gl.getUniformLocation(p, n); }
    return {p, u};
  }
  const prog = gl && program(VS, FS), progPick = gl && program(VS, FS_PICK);

  // ---------------------------------------------------------------------------------- math
  const FAR = 3000, NEAR = 0.05;
  function persp(fovy, asp) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (NEAR - FAR);
    return new Float32Array([f / asp, 0, 0, 0, 0, f, 0, 0, 0, 0, (FAR + NEAR) * nf, -1, 0, 0, 2 * FAR * NEAR * nf, 0]);
  }
  function basis() {
    const cp = Math.cos(cam.pitch), f = [cp * Math.cos(cam.yaw), cp * Math.sin(cam.yaw), Math.sin(cam.pitch)];
    let r = [f[1], -f[0], 0]; const rl = Math.hypot(r[0], r[1]) || 1; r = [r[0] / rl, r[1] / rl, 0];
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    return {f, r, u};
  }
  function viewMat() {
    const {f, r, u} = basis(), e = cam.p;
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return new Float32Array([r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, e), -dot(u, e), dot(f, e), 1]);
  }
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; }
    return o;
  }
  const fovRad = () => (+$('fov').value) * Math.PI / 180;
  const aspect = () => (split ? W / 2 : W) / Math.max(H, 1);
  function mvp() { return mul(persp(fovRad(), aspect()), viewMat()); }
  function sideOf(px) { return split && px >= W / 2 ? 1 : 0; }
  function projectSide(p, side) {  // split view: world (rel O) -> screen px inside the given half
    const s = project(p); if (!s) return null;
    return split ? [s[0] + (side ? W / 2 : 0), s[1], s[2]] : s;
  }
  function project(p, M) {  // world (rel O) -> screen px, or null if behind
    M = M || mvp();
    const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13], w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
    if (w <= 1e-6) return null;
    return [(x / w * 0.5 + 0.5) * (split ? W / 2 : W), (1 - (y / w * 0.5 + 0.5)) * H, w];
  }
  function rayDir(px, py) {
    const {f, r, u} = basis(), t = Math.tan(fovRad() / 2), a = aspect(), Wh = split ? W / 2 : W;
    if (split && px >= Wh) px -= Wh;
    const x = (2 * px / Wh - 1) * t * a, y = (1 - 2 * py / H) * t;
    const d = [f[0] + x * r[0] + y * u[0], f[1] + x * r[1] + y * u[1], f[2] + x * r[2] + y * u[2]], l = Math.hypot(...d);
    return [d[0] / l, d[1] / l, d[2] / l];
  }

  // --------------------------------------------------------------------------- chunk loading
  function parse(buf) {
    const dv = new DataView(buf);
    if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'RSCK') throw new Error('bad chunk');
    const hl = dv.getUint32(4, true), hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
    let off = 8 + hl;
    const take = (Type, n, bytes) => { const a = new Type(buf, off, n); off += Math.ceil(n * bytes / 4) * 4; return a; };
    const V = take(Float32Array, hdr.nv * 3, 4), C = take(Uint8Array, hdr.nv * 3, 1), T = take(Uint32Array, hdr.nt * 3, 4);
    const Hh = (hdr.lod === 0 && hdr.nt !== undefined && off < buf.byteLength) ? take(Int16Array, hdr.nt, 2) : null;
    return {hdr, V, C, T, H: Hh};
  }
  function makeVAO(V, C) {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, V, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, C, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    const ib = gl.createBuffer(), pb = gl.createBuffer();
    gl.bindVertexArray(null);
    return {vao, vb, cb, ib, pb};
  }
  function freeObj(o) { if (!o || !o.gl) return; ['vb', 'cb', 'ib', 'pb'].forEach(k => gl.deleteBuffer(o.gl[k])); gl.deleteVertexArray(o.gl.vao); o.gl = null; }
  const bit = (bits, i) => bits ? (bits[i >> 3] >> (i & 7)) & 1 : 0;
  function rebuild(ch) {
    if (!ch.gl) return;
    const T = ch.T, n = T.length / 3;
    let cnt = 0, pc = 0;
    for (let i = 0; i < n; i++) { if (!bit(ch.del, i)) cnt++; if (bit(ch.prop, i) && !bit(ch.del, i)) pc++; }
    const idx = new Uint32Array(cnt * 3), pidx = new Uint32Array(pc * 3); let a = 0, b = 0;
    for (let i = 0; i < n; i++) {
      const d = bit(ch.del, i);
      if (!d) { idx[a++] = T[3 * i]; idx[a++] = T[3 * i + 1]; idx[a++] = T[3 * i + 2]; }
      if (!d && bit(ch.prop, i)) { pidx[b++] = T[3 * i]; pidx[b++] = T[3 * i + 1]; pidx[b++] = T[3 * i + 2]; }
    }
    gl.bindVertexArray(ch.gl.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ch.gl.ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ch.gl.pb); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, pidx, gl.STATIC_DRAW);
    ch.nIdx = idx.length; ch.nProp = pidx.length; if (!freezePick) pick.valid = false; needDraw = true;
  }
  async function fetchMasks(ch) {
    const [d, p] = await Promise.all([fetch(`/mesh/mask/${ch.key}?which=del`), fetch(`/mesh/mask/${ch.key}?which=prop`)]);
    ch.del = new Uint8Array(await d.arrayBuffer()); ch.prop = new Uint8Array(await p.arrayBuffer());
  }
  let inflight = 0;
  const pending = new Set();
  async function loadChunk(key, lod) {
    pending.add(key); inflight++;
    try {
      const buf = await (await fetch(`/mesh/chunk/${key}?lod=${lod}`)).arrayBuffer();
      const d = parse(buf);
      const ch = {key, lod, V: d.V, C: d.C, T: d.T, H: d.H, nt: d.hdr.nt, gl: makeVAO(d.V, d.C), cen: null};
      if (!lod) await fetchMasks(ch);
      rebuild(ch);
      const old = chunks.get(key); if (old) freeObj(old);
      chunks.set(key, ch);
    } catch (e) { console.error(key, e); }
    inflight--; pending.delete(key); needDraw = true; pick.valid = false;
  }
  // vertex edits (smoothing, fills): patch the chunk's vertex / colour buffers in place
  async function applyDelta(ch) {
    if (!ch || ch.lod || !ch.gl) return;
    const buf = await (await fetch(`/mesh/vdelta/${ch.key}`)).arrayBuffer();
    const dv = new DataView(buf), hl = dv.getUint32(4, true), hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
    if (!hdr.n || !ch.gl) return;
    let off = 8 + hl;
    const idx = new Uint32Array(buf, off, hdr.n); off += hdr.n * 4;
    const P = new Float32Array(buf, off, hdr.n * 3); off += hdr.n * 12;
    const C = new Uint8Array(buf, off, hdr.n * 3);
    for (let j = 0; j < hdr.n; j++) { const i = idx[j]; for (let k = 0; k < 3; k++) { ch.V[3 * i + k] = P[3 * j + k]; ch.C[3 * i + k] = C[3 * j + k]; } }
    gl.bindBuffer(gl.ARRAY_BUFFER, ch.gl.vb); gl.bufferData(gl.ARRAY_BUFFER, ch.V, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, ch.gl.cb); gl.bufferData(gl.ARRAY_BUFFER, ch.C, gl.STATIC_DRAW);
    ch.cen = null; ch.bb = null; pick.valid = false; needDraw = true;
  }
  async function refreshGeom(keys) {
    await Promise.all(keys.map(async k => { const ch = chunks.get(k); if (!ch) return; if (ch.lod) return loadChunk(k, 1); await fetchMasks(ch); rebuild(ch); await applyDelta(ch); }));
    await syncPatches();
  }
  function chunkDist(c) {
    const b = c.bbox, x = cam.p[0], y = cam.p[1];
    const dx = Math.max(b[0] - x, 0, x - b[3]), dy = Math.max(b[1] - y, 0, y - b[4]);
    return Math.hypot(dx, dy);
  }
  function stream() {
    if (!chunkMeta.length) return;
    const rFull = +$('rFull').value, budget = (+$('budget').value) * 1e6;
    const order = chunkMeta.map(c => [c, chunkDist(c)]).sort((a, b) => a[1] - b[1]);
    let used = 0;
    const want = new Map();
    for (const [c, d] of order) {          // hysteresis: what is loaded stays until clearly out of range (no popping)
      const cur = chunks.get(c.key), fullNow = cur && !cur.lod;
      if (d < (fullNow ? rFull * 1.25 : rFull) && used + c.nt <= budget * (fullNow ? 1.1 : 1)) { want.set(c.key, 0); used += c.nt; }
      else if (d < (cur ? LOD_R * 1.15 : LOD_R)) want.set(c.key, fullNow && d < rFull * 1.6 ? 0 : 1);
    }
    for (const [k, ch] of chunks) if (!want.has(k)) { freeObj(ch); chunks.delete(k); needDraw = true; }
    for (const [c] of order) {
      if (inflight >= 2) break;
      const w = want.get(c.key); if (w === undefined || pending.has(c.key)) continue;
      const ch = chunks.get(c.key);
      if (!ch || ch.lod !== w) loadChunk(c.key, w);
    }
    let full = 0, lod = 0, tris = 0;
    for (const ch of chunks.values()) { ch.lod ? lod++ : full++; tris += ch.lod ? ch.nt : ch.nIdx / 3; }
    $('loadInfo').textContent = `${full} chunks full detail, ${lod} coarse; ${(tris / 1e6).toFixed(1)} M triangles drawn${pending.size ? ', loading ' + pending.size : ''}`;
  }

  // -------------------------------------------------------------------------------- patches
  async function syncPatches() {
    const list = await api('/mesh/patches');
    const want = new Map(list.filter(p => p.active).map(p => [p.pid, p]));
    for (const [pid, p] of patches) if (!want.has(pid)) { freeObj(p); patches.delete(pid); }
    for (const [pid, info] of want) {
      if (patches.has(pid)) continue;
      const d = parse(await (await fetch(`/mesh/patch/${pid}`)).arrayBuffer());
      const o = {pid, kind: info.kind, preview: info.preview, V: d.V, T: d.T, gl: makeVAO(d.V, d.C), n: d.T.length};
      gl.bindVertexArray(o.gl.vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.gl.ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, d.T, gl.STATIC_DRAW); gl.bindVertexArray(null);
      patches.set(pid, o);
    }
    pick.valid = false; needDraw = true;
  }

  // ---------------------------------------------------------------------------------- lines
  const LCOL = {left_edge: [1, 0.23, 0.23], right_edge: [0.18, 0.53, 1], centre_line: [1, 0.83, 0], lane_divider: [1, 1, 1], stop_line: [1, 0.55, 0], other: [1, 0.3, 1], drive_path: [0.24, 0.81, 0.56]};
  async function refreshLines() {
    const r = await api('/route3d?annotated=0');
    const pos = [], col = [];
    const add = (pts, c, lift) => { for (let i = 1; i < pts.length; i++) { pos.push(pts[i - 1][0], pts[i - 1][1], pts[i - 1][2] + lift, pts[i][0], pts[i][1], pts[i][2] + lift); col.push(...c, ...c); } };
    add(r.route, [255, 154, 60], 0.12);
    for (const ln of r.lines) add(ln.pts, (LCOL[ln.type] || [1, 1, 1]).map(v => v * 255), 0.06);
    if (lineBuf) freeObj(lineBuf);
    lineBuf = {gl: makeVAO(new Float32Array(pos), new Uint8Array(col))}; lineCount = pos.length / 3; needDraw = true;
  }

  // -------------------------------------------------------------------------------- drawing
  let fillPoly = [];   // 3D fill contour (rel O)
  let fillBuf = null;
  function setFillPoly(pts) {
    fillPoly = pts;
    if (fillBuf) { freeObj(fillBuf); fillBuf = null; }
    if (pts.length > 1) {
      const pos = [], col = [];
      for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; pos.push(a[0], a[1], a[2] + 0.08, b[0], b[1], b[2] + 0.08); col.push(62, 207, 142, 62, 207, 142); }
      fillBuf = {gl: makeVAO(new Float32Array(pos), new Uint8Array(col)), n: pos.length / 3};
    }
    needDraw = true;
  }
  function uniforms(P, M) {
    gl.uniformMatrix4fv(P.u.uMVP, false, M);
    gl.uniform3f(P.u.uCam, ...cam.p);
    if (P.u.uFar) gl.uniform1f(P.u.uFar, FAR);
    if (P.u.uSky) { gl.uniform3f(P.u.uSky, ...SKY); gl.uniform1f(P.u.uFog, $('fog').checked ? LOD_R : 0); gl.uniform1i(P.u.uMode, shade); gl.uniform4f(P.u.uOver, 0, 0, 0, 0); gl.uniform4f(P.u.uTint, 0, 0, 0, 0); }
  }
  function drawGeometry(P, M, isPick) {
    for (const ch of chunks.values()) {
      if (!ch.gl) continue;
      gl.bindVertexArray(ch.gl.vao);
      if (ch.lod) { gl.drawElements(gl.TRIANGLES, ch.nt * 3, gl.UNSIGNED_INT, 0); continue; }
      gl.drawElements(gl.TRIANGLES, ch.nIdx, gl.UNSIGNED_INT, 0);
    }
    if ($('showPatches').checked) for (const p of patches.values()) {
      if (p.preview && !isPick) continue;   // previews are drawn last, on top of the red removals
      if (!isPick) gl.uniform4f(P.u.uTint, ...($('tintPatches').checked ? [0.3, 0.55, 1.0, 0.18] : [0, 0, 0, 0]));
      gl.bindVertexArray(p.gl.vao); gl.drawElements(gl.TRIANGLES, p.n, gl.UNSIGNED_INT, 0);
    }
    if (!isPick) gl.uniform4f(P.u.uTint, 0, 0, 0, 0);
  }
  function draw() {
    needDraw = false;
    if (!gl || TAB !== '3d' || !W) return;
    const M = mvp();
    if (split) { drawSplit(M); return; }
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(...SKY, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    gl.useProgram(prog.p); uniforms(prog, M);
    if (window.SHOW_MESH !== false) drawGeometry(prog, M, false);
    for (const f of drawHooks) { try { f(gl, {M, view: viewMat(), proj: persp(fovRad(), W / Math.max(H, 1)), cam, W: cv.width, H: cv.height, fov: fovRad()}); } catch (e) { console.error(e); } }
    gl.useProgram(prog.p); uniforms(prog, M); gl.depthMask(true); gl.disable(gl.BLEND);
    // pending removals (proposal) in red on top, or Road segmentation preview in light colour + half transparency
    gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(-1, -2);
    if (propTool === 'road_seg') {
      const hex = ($('segColor') && $('segColor').value) || '#8cd2ff';
      const cr = parseInt(hex.slice(1, 3), 16) / 255, cg = parseInt(hex.slice(3, 5), 16) / 255, cb = parseInt(hex.slice(5, 7), 16) / 255;
      const ca = $('segAlpha') ? +$('segAlpha').value : 0.5;
      gl.uniform4f(prog.u.uOver, cr, cg, cb, ca);
    } else {
      gl.uniform4f(prog.u.uOver, 1, 0.12, 0.1, 0.85);
    }
    for (const ch of chunks.values()) {
      if (ch.lod || !ch.nProp) continue;
      gl.bindVertexArray(ch.gl.vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ch.gl.pb);
      gl.drawElements(gl.TRIANGLES, ch.nProp, gl.UNSIGNED_INT, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ch.gl.ib);
    }
    gl.uniform4f(prog.u.uOver, 0, 0, 0, 0); gl.uniform4f(prog.u.uTint, 0.15, 0.95, 0.45, 0.45); gl.polygonOffset(-2, -6);
    if ($('showPatches').checked) for (const p of patches.values()) if (p.preview) { gl.bindVertexArray(p.gl.vao); gl.drawElements(gl.TRIANGLES, p.n, gl.UNSIGNED_INT, 0); }
    gl.uniform4f(prog.u.uTint, 0, 0, 0, 0); gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.uniform1i(prog.u.uMode, 3); gl.uniform1f(prog.u.uFog, 0);
    if ($('showLanes').checked && lineBuf) { gl.bindVertexArray(lineBuf.gl.vao); gl.drawArrays(gl.LINES, 0, lineCount); }
    if (fillBuf) { gl.disable(gl.DEPTH_TEST); gl.bindVertexArray(fillBuf.gl.vao); gl.drawArrays(gl.LINES, 0, fillBuf.n); gl.enable(gl.DEPTH_TEST); }
    gl.bindVertexArray(null);
    drawHud();
  }
  function drawSplit(M) {          // two halves, same camera: one kind of layer in each
    const hw = Math.floor(cv.width / 2);
    gl.enable(gl.SCISSOR_TEST); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    [split.left, split.right].forEach((kind, side) => {
      const x0 = side ? hw : 0, w = side ? cv.width - hw : hw;
      gl.viewport(x0, 0, w, cv.height); gl.scissor(x0, 0, w, cv.height);
      gl.clearColor(...SKY, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(prog.p); uniforms(prog, M); gl.depthMask(true); gl.disable(gl.BLEND);
      if (kind === 'mesh') drawGeometry(prog, M, false);
      else for (const f of drawHooks) { try { f(gl, {M, view: viewMat(), proj: persp(fovRad(), aspect()), cam, W: w, H: cv.height, fov: fovRad()}); } catch (e) { console.error(e); } }
    });
    gl.disable(gl.SCISSOR_TEST); gl.viewport(0, 0, cv.width, cv.height); gl.bindVertexArray(null);
    drawHud();
  }
  function renderPick() {
    const w = cv.width, h = cv.height;
    if (!pick.fbo || pick.w !== w || pick.h !== h) {
      if (pick.fbo) { gl.deleteFramebuffer(pick.fbo); gl.deleteTexture(pick.tex); gl.deleteRenderbuffer(pick.depth); }
      pick.tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, pick.tex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      pick.depth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, pick.depth); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      pick.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, pick.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pick.tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pick.depth);
      pick.w = w; pick.h = h;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, pick.fbo);
    gl.viewport(0, 0, w, h); gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(progPick.p); uniforms(progPick, mvp());
    if (split) {                     // only the mesh half is pickable here (the splat half uses SPLAT.pick)
      const hw = Math.floor(w / 2), side = split.left === 'mesh' ? 0 : split.right === 'mesh' ? 1 : -1;
      if (side >= 0) { gl.viewport(side ? hw : 0, 0, side ? w - hw : hw, h); drawGeometry(progPick, null, true); }
      gl.viewport(0, 0, w, h);
    } else drawGeometry(progPick, null, true);
    gl.bindVertexArray(null);
    pick.valid = true;
  }
  const dec = (r, g, b) => (r / 255 + g / 255 / 255 + b / 255 / 65025) * FAR;
  function pickAt(px, py) {   // CSS px -> {p (rel O), dist} or null
    if (!gl || !W) return null;
    if (!pick.valid) renderPick(); else gl.bindFramebuffer(gl.FRAMEBUFFER, pick.fbo);
    const x = Math.round(px * DPR), y = cv.height - 1 - Math.round(py * DPR), out = new Uint8Array(4);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (out[0] === 255 && out[1] === 255 && out[2] === 255) return null;
    const d = dec(out[0], out[1], out[2]), r = rayDir(px, py);
    return {p: [cam.p[0] + r[0] * d, cam.p[1] + r[1] * d, cam.p[2] + r[2] * d], dist: d};
  }
  function depthBuffer() {   // full distance buffer (device px, bottom-up rows)
    if (!pick.valid) renderPick(); else gl.bindFramebuffer(gl.FRAMEBUFFER, pick.fbo);
    const out = new Uint8Array(cv.width * cv.height * 4);
    gl.readPixels(0, 0, cv.width, cv.height, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return {buf: out, w: cv.width, h: cv.height, at: (px, py) => {
      const x = Math.round(px * DPR), y = cv.height - 1 - Math.round(py * DPR);
      if (x < 0 || y < 0 || x >= cv.width || y >= cv.height) return -1;
      const i = 4 * (y * cv.width + x); if (out[i] === 255 && out[i + 1] === 255 && out[i + 2] === 255) return Infinity;
      return dec(out[i], out[i + 1], out[i + 2]);
    }};
  }
  function centroids(ch) {
    if (ch.cen) return ch.cen;
    const V = ch.V, T = ch.T, n = T.length / 3, c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) c[3 * i + k] = (V[3 * T[3 * i] + k] + V[3 * T[3 * i + 1] + k] + V[3 * T[3 * i + 2] + k]) / 3;
    return (ch.cen = c);
  }

  // ------------------------------------------------------------------------------------ HUD
  const hudHooks = [];
  function drawHud() {
    hx.setTransform(DPR, 0, 0, DPR, 0, 0); hx.clearRect(0, 0, W, H);
    // compass
    const cx = W - 60, cy = H - 60;
    hx.fillStyle = '#0d0f13aa'; hx.beginPath(); hx.arc(cx, cy, 34, 0, 7); hx.fill();
    hx.save(); hx.translate(cx, cy); hx.rotate(-(northYaw() - cam.yaw));
    hx.fillStyle = '#ff5d5d'; hx.beginPath(); hx.moveTo(0, -26); hx.lineTo(6, 0); hx.lineTo(-6, 0); hx.fill();
    hx.fillStyle = '#ddd'; hx.beginPath(); hx.moveTo(0, 26); hx.lineTo(6, 0); hx.lineTo(-6, 0); hx.fill();
    hx.fillStyle = '#fff'; hx.font = 'bold 11px system-ui'; hx.fillText('N', -4, -28); hx.restore();
    if (tool === 'nav' && !drive.on) { hx.strokeStyle = '#ffffffaa'; hx.lineWidth = 1; hx.beginPath(); hx.moveTo(W / 2 - 8, H / 2); hx.lineTo(W / 2 + 8, H / 2); hx.moveTo(W / 2, H / 2 - 8); hx.lineTo(W / 2, H / 2 + 8); hx.stroke(); }
    if (fillPoly.length) {
      const M = mvp();
      fillPoly.forEach((p, i) => { const s = project(p, M); if (s) { hx.fillStyle = '#3ecf8e'; hx.beginPath(); hx.arc(s[0], s[1], 4, 0, 7); hx.fill(); if (!i) { hx.fillText('start', s[0] + 6, s[1] - 6); } } });
    }
    hudHooks.forEach(f => f(hx));
    updateInfo();
  }
  // north / bearings from the georeference (model axes are rotated relative to east / north)
  function northYaw() { const g = META && META.geo; return g ? Math.atan2(g.A[0][0], -g.A[0][1]) : Math.PI / 2; }
  function bearing(yaw) {
    const g = META && META.geo, c = Math.cos(yaw), s = Math.sin(yaw);
    const e = g ? g.A[0][0] * c + g.A[0][1] * s : c, n = g ? g.A[1][0] * c + g.A[1][1] * s : s;
    return ((Math.atan2(e, n) * 180 / Math.PI) % 360 + 360) % 360;
  }
  function routeAtArc(s) {
    if (!ROUTE3.length) return null;
    let lo = 0, hi = ROUTE3.length - 1; s = Math.max(ROUTE3[0][3], Math.min(ROUTE3[hi][3], s));
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ROUTE3[m][3] < s) lo = m; else hi = m; }
    const a = ROUTE3[lo], b = ROUTE3[hi], t = (s - a[3]) / Math.max(b[3] - a[3], 1e-9);
    const p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
    const i0 = Math.max(0, lo - 2), i1 = Math.min(ROUTE3.length - 1, hi + 2);
    const dx = ROUTE3[i1][0] - ROUTE3[i0][0], dy = ROUTE3[i1][1] - ROUTE3[i0][1], dz = ROUTE3[i1][2] - ROUTE3[i0][2];
    return {p, yaw: Math.atan2(dy, dx), slope: Math.atan2(dz, Math.hypot(dx, dy))};
  }
  function arcOf(x, y) {
    let bi = 0, bd = Infinity;
    for (let i = 0; i < ROUTE3.length; i++) { const d = (ROUTE3[i][0] - x) ** 2 + (ROUTE3[i][1] - y) ** 2; if (d < bd) { bd = d; bi = i; } }
    return ROUTE3.length ? ROUTE3[bi][3] : 0;
  }
  function updateInfo() {
    const ax = cam.p[0] + O[0], ay = cam.p[1] + O[1];
    window.CAM3D = {x: ax, y: ay, yaw: cam.yaw};
    const arc = drive.on ? drive.s : arcOf(cam.p[0], cam.p[1]);
    const head = bearing(cam.yaw);
    let t = `${drive.on ? 'DRIVE ' + (drive.v * 3.6).toFixed(0) + ' km/h' : 'FLY'}   route ${arc.toFixed(1)} m\n`;
    const g = META && META.geo ? toGps(ax, ay) : null;
    if (g) t += `camera ${g[0].toFixed(6)}, ${g[1].toFixed(6)}\n`;
    t += `heading ${head.toFixed(0)}°  elevation ${(cam.p[2] + O[2]).toFixed(2)} m (model frame)`;
    if (hoverInfo) t += `\ncursor  ${hoverInfo}`;
    $('info3').textContent = t;
    if (TAB === '3d') {
      $('sArc').textContent = `${arc.toFixed(1)} m`;
      if (document.activeElement !== $('scrub')) $('scrub').value = Math.round(arc);
    }
  }

  // ----------------------------------------------------------------------------- navigation
  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (TAB === '3d' && started) {
      const typing = ['input', 'select', 'textarea'].includes((document.activeElement.tagName || '').toLowerCase()) && document.activeElement.type !== 'range' && document.activeElement.type !== 'checkbox';
      const k = c => !typing && keys.has(c);
      const fast = k('Shift') ? 5 : 1, sp = (+$('speed').value) * fast;
      if (drive.on) {
        const acc = 6 * fast;
        if (k('w') || k('ArrowUp')) drive.v += acc * dt; else if (k('s') || k('ArrowDown')) drive.v -= acc * dt; else drive.v *= Math.pow(0.6, dt);
        if (k(' ')) drive.v *= Math.pow(0.02, dt);
        drive.v = Math.max(-8, Math.min(40 * fast, drive.v)); if (Math.abs(drive.v) < 0.02 && !k('w') && !k('s')) drive.v = 0;
        drive.s = Math.max(0, Math.min(ROUTE3.length ? ROUTE3[ROUTE3.length - 1][3] : 0, drive.s + drive.v * dt));
        const look = (k('a') || k('ArrowLeft') ? 1 : 0) - (k('d') || k('ArrowRight') ? 1 : 0);
        drive.look += (look * 1.2 - drive.look) * Math.min(1, dt * 6);
        if (k('z')) $('eyeH').value = Math.max(0.3, +$('eyeH').value - 1.5 * dt), $('eyeHV').textContent = (+$('eyeH').value).toFixed(1) + ' m';
        if (k('x')) $('eyeH').value = Math.min(12, +$('eyeH').value + 1.5 * dt), $('eyeHV').textContent = (+$('eyeH').value).toFixed(1) + ' m';
        const r = routeAtArc(drive.s);
        if (r) { cam.p = [r.p[0], r.p[1], r.p[2] + (+$('eyeH').value)]; cam.yaw = r.yaw + drive.look; cam.pitch = r.slope + drive.pitch; }
        if (drive.v || look || k('z') || k('x')) moved();
      } else {
        const {f, r} = basis(); let mv = false;
        const step = sp * dt, add = (v, s) => { cam.p[0] += v[0] * s; cam.p[1] += v[1] * s; cam.p[2] += v[2] * s; mv = true; };
        if (k('w')) add(f, step); if (k('s')) add(f, -step);
        if (k('d')) add(r, step); if (k('a')) add(r, -step);
        if (k('e')) add([0, 0, 1], step); if (k('q')) add([0, 0, 1], -step);
        if (k('ArrowLeft')) { cam.yaw += 1.6 * dt; mv = true; } if (k('ArrowRight')) { cam.yaw -= 1.6 * dt; mv = true; }
        if (k('ArrowUp')) { cam.pitch = Math.min(1.55, cam.pitch + 1.1 * dt); mv = true; } if (k('ArrowDown')) { cam.pitch = Math.max(-1.55, cam.pitch - 1.1 * dt); mv = true; }
        if (mv) moved();
      }
    }
    if (needDraw) draw();
    requestAnimationFrame(tick);
  }
  let streamT = 0;
  function moved() {
    needDraw = true; pick.valid = false;
    if (performance.now() - streamT > 250) { streamT = performance.now(); stream(); }
  }
  // mouse look (left drag in Navigate, right drag in every tool)
  let look = null;
  cv.addEventListener('contextmenu', e => e.preventDefault());
  cv.addEventListener('mousedown', e => {
    if (e.button === 2 && !drive.on) {          // right-drag: orbit around the point under the mouse
      const r = cv.getBoundingClientRect(), h = pickAt(e.clientX - r.left, e.clientY - r.top);
      const {f} = basis(), piv = h ? h.p : [cam.p[0] + f[0] * 20, cam.p[1] + f[1] * 20, cam.p[2] + f[2] * 20];
      const d = [cam.p[0] - piv[0], cam.p[1] - piv[1], cam.p[2] - piv[2]], R = Math.hypot(...d);
      orbit = {x: e.clientX, y: e.clientY, piv, R, az: Math.atan2(d[1], d[0]), el: Math.asin(Math.max(-1, Math.min(1, d[2] / Math.max(R, 1e-6))))};
      e.preventDefault(); return;
    }
    if (e.button === 1 || (e.button === 0 && tool === 'nav')) { look = {x: e.clientX, y: e.clientY}; e.preventDefault(); }
  });
  let orbit = null;
  window.addEventListener('mousemove', e => {
    if (!orbit) return;
    const dx = e.clientX - orbit.x, dy = e.clientY - orbit.y; orbit.x = e.clientX; orbit.y = e.clientY;
    orbit.az -= dx * 0.006; orbit.el = Math.max(-0.2, Math.min(1.5, orbit.el + dy * 0.006));
    const ce = Math.cos(orbit.el), p = orbit.piv;
    cam.p = [p[0] + orbit.R * ce * Math.cos(orbit.az), p[1] + orbit.R * ce * Math.sin(orbit.az), p[2] + orbit.R * Math.sin(orbit.el)];
    cam.yaw = Math.atan2(p[1] - cam.p[1], p[0] - cam.p[0]); cam.pitch = -orbit.el; moved();
  });
  window.addEventListener('mouseup', () => { orbit = null; });
  window.addEventListener('mousemove', e => {
    if (look) {
      const dx = e.clientX - look.x, dy = e.clientY - look.y; look = {x: e.clientX, y: e.clientY};
      if (drive.on) { drive.look -= dx * 0.004; drive.pitch = Math.max(-1.2, Math.min(1.2, drive.pitch - dy * 0.004)); }
      else { cam.yaw -= dx * 0.004; cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch - dy * 0.004)); }
      moved();
    }
  });
  window.addEventListener('mouseup', () => { look = null; });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.ctrlKey) { $('fov').value = Math.max(30, Math.min(110, +$('fov').value + Math.sign(e.deltaY) * 3)); $('fovV').textContent = $('fov').value + '°'; moved(); return; }
    if (drive.on) { drive.s += -Math.sign(e.deltaY) * 2; moved(); return; }
    const {f} = basis(), s = -Math.sign(e.deltaY) * Math.max(0.5, (+$('speed').value) * 0.4) * (e.shiftKey ? 5 : 1);
    cam.p = [cam.p[0] + f[0] * s, cam.p[1] + f[1] * s, cam.p[2] + f[2] * s]; moved();
  }, {passive: false});
  cv.addEventListener('dblclick', e => {
    if (tool !== 'nav' || drive.on) return;
    const r = cv.getBoundingClientRect(), h = pickAt(e.clientX - r.left, e.clientY - r.top); if (!h) return;
    const {f} = basis(), back = Math.min(8, h.dist * 0.6);
    cam.p = [h.p[0] - f[0] * back, h.p[1] - f[1] * back, h.p[2] - f[2] * back]; moved();
  });
  // hover: cursor position + GPS (throttled)
  let hoverT = 0;
  cv.addEventListener('mousemove', e => {
    if (look || performance.now() - hoverT < 120) return; hoverT = performance.now();
    const r = cv.getBoundingClientRect(), h = pickAt(e.clientX - r.left, e.clientY - r.top);
    if (!h) { hoverInfo = null; $('sCur').textContent = '-'; $('sGps').textContent = '-'; needDraw = true; return; }
    const ax = h.p[0] + O[0], ay = h.p[1] + O[1], az = h.p[2] + O[2];
    hoverInfo = `${gpsText(ax, ay)}  z ${az.toFixed(2)} m  ${h.dist.toFixed(1)} m away`;
    $('sCur').textContent = `x ${ax.toFixed(2)}  y ${ay.toFixed(2)}  z ${az.toFixed(2)} | route ${arcOf(h.p[0], h.p[1]).toFixed(1)} m`;
    $('sGps').textContent = gpsText(ax, ay);
    needDraw = true;
  });
  window.addEventListener('keydown', e => {
    if (TAB !== '3d') return;
    const tag = (e.target.tagName || '').toLowerCase();
    if ((tag === 'input' && e.target.type !== 'range' && e.target.type !== 'checkbox') || tag === 'select') return;
    if (e.ctrlKey || e.metaKey) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    keys.add(k === 'shift' ? 'Shift' : k); if (e.shiftKey) keys.add('Shift');
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'PageUp', 'PageDown'].includes(e.key)) e.preventDefault();
    if (k === 't') toggleDrive();
    if (k === 'n') { shade = (shade + 1) % 3; $('bShade').textContent = ['Shading: colour', 'Shading: lit grey', 'Shading: colour + light'][shade]; needDraw = true; }
    if (drive.on && (e.key === 'PageDown' || e.key === 'PageUp')) { drive.s += (e.key === 'PageDown' ? 1 : -1) * (e.shiftKey ? 100 : 20); moved(); }
  });
  window.addEventListener('keyup', e => { const k = e.key.length === 1 ? e.key.toLowerCase() : e.key; keys.delete(k); if (e.key === 'Shift') keys.delete('Shift'); });
  window.addEventListener('blur', () => keys.clear());
  function toggleDrive(on) {
    drive.on = on === undefined ? !drive.on : on;
    if (drive.on) { drive.s = arcOf(cam.p[0], cam.p[1]); drive.v = 0; drive.look = 0; drive.pitch = -0.05; }
    $('bDrive').classList.toggle('toggled', drive.on); $('bDrive').textContent = drive.on ? 'Driving (T = fly)' : 'Drive mode';
    moved();
  }
  $('bDrive').onclick = () => toggleDrive();
  $('bShade').onclick = () => { shade = (shade + 1) % 3; $('bShade').textContent = ['Shading: colour', 'Shading: lit grey', 'Shading: colour + light'][shade]; needDraw = true; };
  $('bOnMap').onclick = () => showOnMap(cam.p[0] + O[0], cam.p[1] + O[1]);
  [['fov', 'fovV', '°'], ['speed', 'speedV', ' m/s'], ['rFull', 'rFullV', ' m'], ['budget', 'budgetV', ' M tris'], ['eyeH', 'eyeHV', ' m']].forEach(([id, out, unit]) =>
    $(id).addEventListener('input', () => { $(out).textContent = $(id).value + unit; moved(); }));
  ['showLanes', 'showPatches', 'tintPatches', 'fog'].forEach(id => $(id).addEventListener('input', () => { pick.valid = false; needDraw = true; }));

  // ---------------------------------------------------------------------------------- setup
  function resize() {
    if (TAB !== '3d') return;
    const r = $('v3Wrap').getBoundingClientRect(); DPR = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR); hud.width = cv.width; hud.height = cv.height;
    pick.valid = false; needDraw = true;
  }
  window.addEventListener('resize', resize);
  let startPromise = null;
  function start() { return startPromise || (startPromise = doStart()); }
  async function doStart() {
    started = true;
    await READY;
    const m = await api('/mesh/meta');
    if (m.error) { $('hint3').textContent = m.error; return; }
    O = m.origin; chunkMeta = m.chunks;
    ROUTE3 = META.route.map(r => [r[0] - O[0], r[1] - O[1], r[3] - O[2], r[2]]);
    if (!placed) { const r = routeAtArc(0); cam.p = [r.p[0] - 10 * Math.cos(r.yaw), r.p[1] - 10 * Math.sin(r.yaw), r.p[2] + 5]; cam.yaw = r.yaw; cam.pitch = -0.3; }
    refreshLines(); syncPatches(); stream();
    setInterval(() => { if (TAB === '3d') stream(); }, 600);
  }
  let placed = false;
  function placeAt(x, y, yaw) {   // absolute world x, y
    placed = true;
    const go = () => {
      const px = x - O[0], py = y - O[1];
      const arc = arcOf(px, py), back = routeAtArc(Math.max(0, arc - 10)), at = routeAtArc(arc);
      if (!back) return;
      cam.p = [back.p[0], back.p[1], back.p[2] + 2.5];          // 10 m back along the road, 2.5 m up
      const dx = px - cam.p[0], dy = py - cam.p[1], dz = at.p[2] - cam.p[2];
      cam.yaw = Math.hypot(dx, dy) > 1 ? Math.atan2(dy, dx) : back.yaw;
      cam.pitch = Math.atan2(dz, Math.max(Math.hypot(dx, dy), 1)); toggleDrive(false); moved();
    };
    start().then(go);
  }
  // status polling: masks / patches / proposal follow the server
  async function poll() {
    try {
      const st = await api(`/mesh/status?since=${version}`);
      if (typeof st.version !== 'number') throw new Error(st.error || 'bad status');
      const newTool = st.proposal ? st.proposal.tool : null;
      if (newTool !== propTool) { propTool = newTool; needDraw = true; }
      if (st.version !== version) {
        version = st.version;
        const geom = new Set(st.geom || []);
        for (const k of st.changed) {
          const ch = chunks.get(k); if (!ch) continue;
          if (ch.lod) loadChunk(k, 1); else { await fetchMasks(ch); rebuild(ch); if (geom.has(k)) await applyDelta(ch); }
        }
        await syncPatches();
      }
      onStatus.forEach(f => f(st));
    } catch (e) { /* server restarting */ }
    setTimeout(poll, 700);
  }
  tabHooks.push(t => { if (t === '3d') { resize(); start(); moved(); } else keys.clear(); });
  gotoHooks.push((x, y) => { if (TAB === '3d') placeAt(x, y, cam.yaw); });
  $('scrub').addEventListener('input', e => {
    if (TAB !== '3d') return;
    const r = routeAtArc(+e.target.value); if (!r) return;
    if (drive.on) { drive.s = +e.target.value; } else { cam.p = [r.p[0] - 12 * Math.cos(r.yaw), r.p[1] - 12 * Math.sin(r.yaw), r.p[2] + 6]; cam.yaw = r.yaw; cam.pitch = -0.35; }
    moved();
  });
  requestAnimationFrame(tick);
  READY.then(() => poll());   // status (pending changes, jobs) also while only the map is open

  return {
    get O() { return O; }, cam, chunks, patches, refreshGeom, project, pickAt, depthBuffer, centroids, rebuild, mvp, bit,
    drawHooks, rayDir, setTool(t) { tool = t; needDraw = true; }, get tool() { return tool; }, onStatus, hudHooks, setFillPoly,
    redraw() { needDraw = true; }, freeze(v) { freezePick = v; if (!v) pick.valid = false; }, invalidatePick() { pick.valid = false; }, refreshLines, placeAt, arcOf, routeAtArc,
    get started() { return started; }, start, get drive() { return drive; },
    get split() { return split; }, setSplit(v) { split = v || null; pick.valid = false; needDraw = true; }, sideOf, projectSide, get W() { return W; }, get H() { return H; },
  };
})();
