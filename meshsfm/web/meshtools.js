'use strict';
// Mesh tools in the 3D view (brush, lasso, pick blob, fill contour, inspect) + automatic cleaning UI.
const M3 = (() => {
  const cv = $('c3');
  const HINTS = {
    nav: 'Navigate: drag to look around, W/A/S/D fly, E/Q up/down, arrows turn (360°), wheel forward/back, double-click flies to a point, T drive mode.',
    brush: 'Brush: drag over floaters to erase (Alt+drag restores). [ / ] radius. "Protect ground" keeps the surface. Right-drag looks around.',
    smooth: 'Smooth: drag over spikes / bumps / pits: the surface is evened out and blended into the neighbours (colours too), holes under the brush are closed. [ / ] radius. Ctrl+Z undoes a whole stroke. Right-drag looks around.',
    lasso: 'Lasso: drag an outline. Visible triangles inside turn red; Enter applies, Esc drops.',
    rect: 'Box select: drag a rectangle over anything (road too).',
    poly: 'Polyline select: click corners, double-click or Enter closes; Backspace removes the last corner, Esc cancels.',
    blob: 'Pick blob: click a floating piece / ghost obstacle. The connected piece above the ground turns red; Enter applies.',
    fill: 'Fill area: click points on the surface around a hole (contour). Enter generates the patch, Backspace removes the last point, Esc clears.',
    inspect: 'Inspect: click anywhere for position, GPS and height above the ground.',
    lanes: 'Edit lanes: click a line to select, drag keynodes along the surface, Alt+click / double-click a line inserts, Shift+click adds a keynode at the nearer end (or starts a new line), Del deletes, 1-6 type. Right-drag looks around.',
  };
  let sm = null;   // smoothing stroke: {id, pts (rel O), sent, busy}
  let tool = 'nav', stroke = null, lasso = null, rect = null, polyPts = [], fillPts = [], mouse = {x: 0, y: 0, hit: null}, prop = null, busy = false;

  function setTool(t) {
    tool = t; V3.setTool(t); lasso = null; polyPts = [];
    if (t !== 'fill' && fillPts.length) { fillPts = []; V3.setFillPoly([]); }
    document.querySelectorAll('button.tool3').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
    cv.style.cursor = {nav: 'grab', brush: 'none', smooth: 'none', lasso: 'crosshair', rect: 'crosshair', poly: 'crosshair', blob: 'pointer', fill: 'crosshair', inspect: 'help', lanes: 'default'}[t];
    $('hint3').textContent = HINTS[t]; V3.redraw();
  }
  document.querySelectorAll('button.tool3').forEach(b => b.onclick = () => setTool(b.dataset.tool));

  // ------------------------------------------------------------------------- selection helpers
  const protectH = () => $('protect').checked ? +$('protectH').value * 100 : -1e9;  // cm, like the server's H
  function bboxOf(ch) {
    if (ch.bb) return ch.bb;
    const c = V3.centroids(ch); const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < c.length; i += 3) for (let k = 0; k < 3; k++) { b[k] = Math.min(b[k], c[i + k]); b[k + 3] = Math.max(b[k + 3], c[i + k]); }
    return (ch.bb = b);
  }
  function paint(px, py) {
    const h = V3.pickAt(px, py); mouse.hit = h; if (!h) return;
    const r = +$('brushR').value, r2 = r * r, ph = protectH(), restore = stroke.restore;
    for (const ch of V3.chunks.values()) {
      if (ch.lod || !ch.del) continue;
      const b = bboxOf(ch);
      if (h.p[0] < b[0] - r || h.p[0] > b[3] + r || h.p[1] < b[1] - r || h.p[1] > b[4] + r || h.p[2] < b[2] - r || h.p[2] > b[5] + r) continue;
      const c = V3.centroids(ch), n = c.length / 3; let changed = false;
      let set = stroke.sets.get(ch.key); if (!set) stroke.sets.set(ch.key, set = new Set());
      for (let i = 0; i < n; i++) {
        const dx = c[3 * i] - h.p[0], dy = c[3 * i + 1] - h.p[1], dz = c[3 * i + 2] - h.p[2];
        if (dx * dx + dy * dy + dz * dz > r2) continue;
        if (!restore && ch.H && ch.H[i] <= ph) continue;
        const isDel = V3.bit(ch.del, i);
        if (isDel === (restore ? 0 : 1)) continue;
        ch.del[i >> 3] ^= (1 << (i & 7)); set.add(i); changed = true;
      }
      if (changed) V3.rebuild(ch);
    }
  }
  async function endStroke() {
    const s = stroke; stroke = null; V3.freeze(false);
    const sets = {}; let n = 0;
    for (const [k, v] of s.sets) if (v.size) { sets[k] = [...v]; n += v.size; }
    if (!n) return;
    await api('/mesh/edit', {[s.restore ? 'undel' : 'del']: sets, label: `brush ${s.restore ? 'restore' : 'erase'} (${n} triangles)`});
    toast(`${s.restore ? 'Restored' : 'Erased'} ${n.toLocaleString()} triangles (Ctrl+Z undoes)`);
  }
  // smoothing brush: the stroke goes to the server in pieces while dragging (one undo step per stroke)
  function smAdd(px, py) {
    const h = V3.pickAt(px, py); mouse.hit = h; if (!h) return;
    const last = sm.pts[sm.pts.length - 1], r = +$('brushR').value;
    if (last && Math.hypot(h.p[0] - last[0], h.p[1] - last[1]) < r / 4) return;
    sm.pts.push(h.p);
  }
  function smParams() {
    return {radius: +$('brushR').value, strength: +$('smS').value, range: +$('smH').value, colour: $('smC').value, close_holes: $('smHole').checked};
  }
  const abs = pts => { const O = V3.O; return pts.map(p => [p[0] + O[0], p[1] + O[1], p[2] + O[2]]); };
  async function smSend(s, body) {
    s.busy = true;
    try {
      const r = await api('/mesh/smooth', Object.assign({params: smParams(), stroke: s.id}, body));
      if (r.keys && r.keys.length) await V3.refreshGeom(r.keys);
      s.busy = false; return r;
    } catch (e) { s.busy = false; toast('smoothing failed: ' + e); return {}; }
  }
  function smFlush() {      // new stroke points (+ the previous one, so the stroke stays continuous)
    const s = sm; if (!s || s.busy || s.pts.length <= s.sent) return;
    const pts = s.pts.slice(Math.max(0, s.sent - 1)); s.sent = s.pts.length;
    smSend(s, {pts: abs(pts)}).then(r => { s.moved += r.changed || 0; });
  }
  async function smEnd() {
    const s = sm; sm = null; clearInterval(s.timer);
    while (s.busy) await new Promise(r => setTimeout(r, 30));
    if (s.pts.length > s.sent) { const r = await smSend(s, {pts: abs(s.pts.slice(Math.max(0, s.sent - 1)))}); s.moved += r.changed || 0; }
    if (!s.pts.length) return;
    let holes = 0;
    if ($('smHole').checked) { const r = await smSend(s, {pts: abs(s.pts), final: true, close_only: true}); holes = r.holes || 0; }
    toast(s.moved || holes ? `Smoothed${holes ? ' and closed holes' : ''} (Ctrl+Z undoes the whole stroke)` : 'Nothing to smooth there (raise "height range"?)');
  }
  async function applySelection(poly, kind) {
    const act = $('selAct') ? $('selAct').value : 'remove';
    if (act === 'remove') {
      const {out, n} = lassoSelect(poly);
      if (!n) { toast('Nothing inside the selection (fly closer: far chunks are coarse previews)'); return; }
      await api('/mesh/propose_ids', {del: out, label: `${kind} selection (${n} triangles)`});
      toast(`${n.toLocaleString()} triangles selected (red): Enter applies, Esc drops`); return;
    }
    // reshape / recolour: the selection outline projected onto the surface -> the contour engine
    const O = V3.O, ring = [];
    const N = Math.max(8, poly.length); const pts = poly.length >= 8 ? poly : poly.flatMap((p, i) => { const q = poly[(i + 1) % poly.length]; return [0, .25, .5, .75].map(t => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]); });
    for (const [x, y] of pts) { const h = V3.pickAt(x, y); if (h) ring.push([h.p[0] + O[0], h.p[1] + O[1]]); }
    if (ring.length < 3) { toast('The selection must lie on the surface'); return; }
    if (act === 'recolour') { const r = await api('/mesh/recolour', {poly: ring}); toast(r.ok ? 'Recolouring the selection from the orthophoto (Ctrl+Z undoes)' : r.error); return; }
    const prm = Object.assign(fillParams(), act === 'flatten' ? {mode: 'level', remove: 'ground'} : act === 'smooth' ? {mode: 'smooth', remove: 'ground'} :
                              {mode: 'smooth', remove: 'ground', flat_h: 0.02, colour: 'ortho'});
    const r = await api('/mesh/fill', {poly: ring, params: prm});
    toast(r.ok ? `${act} of the ${kind} selection: computing - shown green / red, Enter applies` : (r.error || 'failed'));
  }
  function lassoSelect(poly) {
    const W = cv.clientWidth, H = cv.clientHeight;
    const off = document.createElement('canvas'); off.width = W; off.height = H;
    const o = off.getContext('2d'); o.beginPath(); poly.forEach((p, i) => i ? o.lineTo(p[0], p[1]) : o.moveTo(p[0], p[1])); o.closePath(); o.fill();
    const mask = o.getImageData(0, 0, W, H).data;
    const through = $('lassoThrough').checked, db = through ? null : V3.depthBuffer(), M = V3.mvp(), cam = V3.cam.p, ph = protectH();
    let xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const out = {}; let n = 0;
    for (const ch of V3.chunks.values()) {
      if (ch.lod || !ch.del) continue;
      const c = V3.centroids(ch), cnt = c.length / 3, sel = [];
      for (let i = 0; i < cnt; i++) {
        if (V3.bit(ch.del, i)) continue;       // explicit selections take everything inside (road included)
        const px = c[3 * i], py = c[3 * i + 1], pz = c[3 * i + 2];
        const w = M[3] * px + M[7] * py + M[11] * pz + M[15]; if (w <= 0.05) continue;
        const sx = ((M[0] * px + M[4] * py + M[8] * pz + M[12]) / w * 0.5 + 0.5) * W;
        if (sx < x0 || sx > x1) continue;
        const sy = (1 - ((M[1] * px + M[5] * py + M[9] * pz + M[13]) / w * 0.5 + 0.5)) * H;
        if (sy < y0 || sy > y1 || !mask[4 * ((sy | 0) * W + (sx | 0)) + 3]) continue;
        if (!through) {
          const d = Math.hypot(px - cam[0], py - cam[1], pz - cam[2]), z = db.at(sx, sy);
          if (z >= 0 && d > z + 0.15 + 0.01 * d) continue;
        }
        sel.push(i);
      }
      if (sel.length) { out[ch.key] = sel; n += sel.length; }
    }
    return {out, n};
  }

  // ------------------------------------------------------------------------------------ mouse
  cv.addEventListener('mousedown', async e => {
    if (e.button !== 0 || tool === 'nav') return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (tool === 'lanes') { Lanes3D.down(e, x, y); return; }
    if (tool === 'brush') { stroke = {restore: e.altKey, sets: new Map()}; V3.freeze(true); paint(x, y); return; }
    if (tool === 'smooth') { sm = {id: 'sm' + Date.now(), pts: [], sent: 0, busy: false, moved: 0, timer: setInterval(smFlush, 160)}; smAdd(x, y); return; }
    if (tool === 'lasso') { lasso = [[x, y]]; return; }
    if (tool === 'rect') { rect = [x, y, x, y]; return; }
    if (tool === 'poly') { if (e.detail >= 2 && polyPts.length >= 3) { const pp = polyPts; polyPts = []; V3.redraw(); await applySelection(pp, 'polyline'); return; } polyPts.push([x, y]); V3.redraw(); return; }
    if (tool === 'inspect' && V3.split) {           // Align side by side: the half that was clicked decides the layer
      const side = V3.sideOf(x), kind = side ? V3.split.right : V3.split.left;
      const hs = kind === 'mesh' ? V3.pickAt(x, y) : (typeof SPLAT !== 'undefined' ? SPLAT.pick(x, y) : null);
      if (!hs) { toast(`Nothing under the cursor on the ${kind}`); return; }
      window.dispatchEvent(new CustomEvent('pick3d', {detail: {p: hs.p, side}})); return;
    }
    let h = window.SHOW_MESH !== false ? V3.pickAt(x, y) : null, src = 'mesh';
    if (tool === 'inspect' && typeof SPLAT !== 'undefined') {      // the splat counts too (nearer of the two, if shown)
      const s = SPLAT.pick(x, y); if (s && (!h || s.dist < h.dist - 0.05)) { h = s; src = 'splat'; }
    }
    if (!h) { toast('Nothing under the cursor'); return; }
    if (tool === 'fill') { fillPts.push(h.p); V3.setFillPoly(fillPts.slice()); return; }
    if (tool === 'blob' || tool === 'inspect') {
      let best = null, bd = 0.5;
      for (const ch of V3.chunks.values()) {
        if (ch.lod || !ch.del) continue;
        const b = bboxOf(ch); if (h.p[0] < b[0] - 1 || h.p[0] > b[3] + 1 || h.p[1] < b[1] - 1 || h.p[1] > b[4] + 1) continue;
        const c = V3.centroids(ch);
        for (let i = 0; i < c.length / 3; i++) { if (V3.bit(ch.del, i)) continue; const d = Math.hypot(c[3 * i] - h.p[0], c[3 * i + 1] - h.p[1], c[3 * i + 2] - h.p[2]); if (d < bd) { bd = d; best = {ch, i}; } }
      }
      const O = V3.O, ax = h.p[0] + O[0], ay = h.p[1] + O[1], az = h.p[2] + O[2];
      if (tool === 'inspect') {
        window.dispatchEvent(new CustomEvent('pick3d', {detail: {p: h.p}}));
        const g = await api(`/ground?x=${ax}&y=${ay}`);
        const lines = [`point   x ${ax.toFixed(3)}  y ${ay.toFixed(3)}  z ${az.toFixed(3)} (z-up)`, 
          `GPS     ${gpsText(ax, ay)}`, `ground  z ${g.z.toFixed(3)}  ->  ${(az - g.z).toFixed(2)} m above ground`, `route   ${V3.arcOf(h.p[0], h.p[1]).toFixed(1)} m`];
        if (best) lines.push(`mesh    ${best.ch.key} triangle ${best.i}${best.ch.H ? ', ' + (best.ch.H[best.i] / 100).toFixed(2) + ' m above ground grid' : ''}`);
        lines.unshift(`on      ${src}`);
        window.INSPECT_PIN = {p: h.p, src, gps: gpsText(ax, ay), h: (az - g.z)}; V3.redraw();
        $('info3').textContent = lines.join('\n');
        return;
      }
      if (!best) { toast('No full-detail triangle there (fly closer)'); return; }
      const res = await api('/mesh/blob', {key: best.ch.key, tri: best.i, hprotect: $('protect').checked ? +$('protectH').value : -100});
      res.ok ? toast(res.msg + ': Enter applies, Esc drops') : toast(res.error);
    }
  });
  window.addEventListener('mousemove', e => {
    if (TAB !== '3d') return;
    const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
    if (Lanes3D.dragging) { Lanes3D.move(mouse.x, mouse.y); return; }
    if (stroke) { paint(mouse.x, mouse.y); V3.redraw(); return; }
    if (sm) { smAdd(mouse.x, mouse.y); V3.redraw(); return; }
    if (lasso) { lasso.push([mouse.x, mouse.y]); V3.redraw(); return; }
    if (rect) { rect[2] = mouse.x; rect[3] = mouse.y; V3.redraw(); return; }
    if (tool === 'poly' && polyPts.length) V3.redraw();
    if ((tool === 'brush' || tool === 'smooth') && mouse.x >= 0 && mouse.y >= 0 && mouse.x < r.width && mouse.y < r.height) { mouse.hit = V3.pickAt(mouse.x, mouse.y); V3.redraw(); }
  });
  window.addEventListener('mouseup', async e => {
    if (e.button !== 0) return;
    if (Lanes3D.up()) return;
    if (stroke) await endStroke();
    if (sm) await smEnd();
    if (rect) {
      const [x0, y0, x1, y1] = rect; rect = null; V3.redraw();
      if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) return;
      await applySelection([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 'box'); return;
    }
    if (lasso) {
      const poly = lasso; lasso = null; V3.redraw();
      if (poly.length < 3) return;
      await applySelection(poly, 'lasso');
    }
  });
  V3.hudHooks.push(hx => {
    if (TAB !== '3d') return;
    if ((tool === 'brush' || tool === 'smooth') && mouse.hit) {
      const f = cv.clientHeight / (2 * Math.tan((+$('fov').value) * Math.PI / 360)), rp = (+$('brushR').value) * f / Math.max(mouse.hit.dist, 0.1);
      hx.strokeStyle = tool === 'smooth' ? '#4da3ff' : stroke && stroke.restore ? '#3ecf8e' : '#ff5d5d'; hx.lineWidth = 2;
      hx.beginPath(); hx.arc(mouse.x, mouse.y, rp, 0, 7); hx.stroke(); hx.beginPath(); hx.arc(mouse.x, mouse.y, 2, 0, 7); hx.stroke();
    }
    if (rect) {
      hx.strokeStyle = '#ffb020'; hx.fillStyle = 'rgba(255,176,32,.12)'; hx.lineWidth = 2; hx.setLineDash([6, 4]);
      const x = Math.min(rect[0], rect[2]), y = Math.min(rect[1], rect[3]), w = Math.abs(rect[2] - rect[0]), hh = Math.abs(rect[3] - rect[1]);
      hx.fillRect(x, y, w, hh); hx.strokeRect(x, y, w, hh); hx.setLineDash([]);
    }
    if (tool === 'poly' && polyPts.length) {
      hx.strokeStyle = '#ffb020'; hx.lineWidth = 2; hx.fillStyle = 'rgba(255,176,32,.12)'; hx.beginPath();
      polyPts.forEach((p, i) => i ? hx.lineTo(p[0], p[1]) : hx.moveTo(p[0], p[1])); hx.lineTo(mouse.x, mouse.y); hx.closePath(); hx.fill();
      hx.setLineDash([6, 4]); hx.stroke(); hx.setLineDash([]);
      hx.fillStyle = '#ffb020'; polyPts.forEach(p => { hx.beginPath(); hx.arc(p[0], p[1], 4, 0, 7); hx.fill(); });
    }
    const pin = window.INSPECT_PIN;
    if (pin && tool === 'inspect' && !V3.split) {           // Google-Maps style pin at the inspected point
      const s = V3.project(pin.p);
      if (s) {
        const [px, py] = s, R = 13;
        hx.save(); hx.shadowColor = 'rgba(0,0,0,.45)'; hx.shadowBlur = 6; hx.shadowOffsetY = 2;
        hx.fillStyle = '#ea4335'; hx.beginPath(); hx.moveTo(px, py);
        hx.bezierCurveTo(px - 4, py - 12, px - R, py - 18, px - R, py - 28); hx.arc(px, py - 28, R, Math.PI, 0); hx.bezierCurveTo(px + R, py - 18, px + 4, py - 12, px, py);
        hx.fill(); hx.restore();
        hx.fillStyle = '#7a1b12'; hx.beginPath(); hx.arc(px, py - 28, 4.8, 0, 7); hx.fill();
        hx.fillStyle = 'rgba(255,255,255,.9)'; hx.beginPath(); hx.arc(px, py, 2.5, 0, 7); hx.fill();
        const txt = `${pin.gps}  ·  ${pin.src}  ·  ${pin.h.toFixed(2)} m above ground`;
        hx.font = '600 13px Roboto, system-ui, sans-serif'; const w = hx.measureText(txt).width + 18;
        hx.fillStyle = 'rgba(20,22,28,.88)'; hx.beginPath(); hx.roundRect ? hx.roundRect(px + 16, py - 52, w, 26, 7) : hx.rect(px + 16, py - 52, w, 26); hx.fill();
        hx.fillStyle = '#fff'; hx.fillText(txt, px + 25, py - 34);
      }
    }
    if (lasso && lasso.length > 1) {
      hx.strokeStyle = '#ffb020'; hx.lineWidth = 2; hx.setLineDash([6, 4]); hx.beginPath();
      lasso.forEach((p, i) => i ? hx.lineTo(p[0], p[1]) : hx.moveTo(p[0], p[1])); hx.closePath(); hx.stroke(); hx.setLineDash([]);
    }
  });

  // ------------------------------------------------------------------------------------- keys
  window.addEventListener('keydown', e => {
    if (TAB !== '3d') return;
    const tag = (e.target.tagName || '').toLowerCase();
    if ((tag === 'input' && e.target.type !== 'range' && e.target.type !== 'checkbox') || tag === 'select') return;
    const k = e.key, ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && (k === 'z' || k === 'Z') && !e.shiftKey) { e.preventDefault(); api3.undo(); return; }
    if (ctrl && (k === 'y' || k === 'Y' || ((k === 'z' || k === 'Z') && e.shiftKey))) { e.preventDefault(); api3.redo(); return; }
    if (tool === 'lanes' && Lanes3D.key(e)) return;
    if (ctrl) return;
    const tk = {v: 'nav', b: 'brush', m: 'smooth', l: 'lasso', r: 'rect', y: 'poly', o: 'blob', g: 'fill', i: 'inspect', p: 'lanes'}[k.toLowerCase()];
    if (tk) { setTool(tk); return; }
    if (k === '[' || k === ']') { const el = $('brushR'); el.value = Math.max(0.1, Math.min(5, +el.value * (k === ']' ? 1.25 : 0.8))); $('brushRV').textContent = (+el.value).toFixed(2) + ' m'; V3.redraw(); return; }
    if (k === 'h' || k === 'H' || k === '?') { $('help').style.display = $('help').style.display === 'flex' ? 'none' : 'flex'; return; }
    if (k === 'Enter' && tool === 'poly' && polyPts.length >= 3) { const pp = polyPts; polyPts = []; applySelection(pp, 'polyline'); return; }
    if (k === 'Backspace' && tool === 'poly' && polyPts.length) { e.preventDefault(); polyPts.pop(); V3.redraw(); return; }
    if (k === 'Escape' && tool === 'poly' && polyPts.length) { polyPts = []; V3.redraw(); return; }
    if (k === 'Enter') { if (tool === 'fill' && fillPts.length >= 3) submit3dFill(); else if (prop) accept(); return; }
    if (k === 'Backspace' && tool === 'fill') { e.preventDefault(); fillPts.pop(); V3.setFillPoly(fillPts.slice()); return; }
    if (k === 'Escape') {
      if ($('help').style.display === 'flex') { $('help').style.display = 'none'; return; }
      if (tool === 'fill' && fillPts.length) { fillPts = []; V3.setFillPoly([]); return; }
      if (prop) { discard(); return; }
      setTool('nav');
    }
  });

  // ------------------------------------------------------------------------------- fill / auto
  function fillParams() {
    return {mode: $('fillMode').value, colour: $('fillColour').value, remove: $('fillRemove').value,
            clear_h: +$('fillClear').value, flat_h: +$('fillFlat').value, blend: +$('fillBlend').value, cell: +$('fillCell').value};
  }
  async function requestFill(polyAbs) {
    const act = $('fillAction') ? $('fillAction').value : 'fill';
    if (act !== 'fill') {
      const hb = $('cropBand').checked;
      const r = await api('/mesh/crop', {poly: polyAbs, mode: act, zmin: hb ? +$('cropH0').value : null, zmax: hb ? +$('cropH1').value : null});
      if (!r.ok) { toast(r.error || 'crop failed'); return; }
      toast(`Crop: ${r.delete.toLocaleString()} triangles marked red - Apply (Enter) or Drop (Esc)`); return;
    }
    const r = await api('/mesh/fill', {poly: polyAbs, params: fillParams()});
    if (!r.ok) { toast(r.error || 'fill failed'); return; }
    toast(TAB === 'map' ? 'Generating the patch... it shows in 3D (T) as green; Apply in the side panel or Enter in 3D' : 'Generating the patch...');
  }
  function submit3dFill() {
    const O = V3.O; requestFill(fillPts.map(p => [p[0] + O[0], p[1] + O[1]]));
    fillPts = []; V3.setFillPoly([]);
  }
  function autoParams(t) {
    const base = {arc0: +$('arc0').value, arc1: +$('arc1').value};
    return Object.assign(base, {
      ghost: {hmin: +$('gH0').value, hmax: +$('gH1').value, margin: +$('gM').value},
      floaters: {gap: +$('fG').value, top: +$('fT').value, maxlat: +$('fL').value},
      islands: {min_tris: +$('iN').value},
      road: {margin: +$('rdM').value, blend: +$('rdB').value, cell: +$('rdC').value},
      skirt: {width: +$('skW').value, below: +$('skB').value, only_missing: $('skMiss').checked},
    }[t]);
  }
  document.querySelectorAll('button.auto').forEach(b => b.onclick = async () => {
    const r = await api('/mesh/propose', {tool: b.dataset.tool, params: autoParams(b.dataset.tool)});
    r.ok ? toast(`Computing ${b.dataset.tool}...`) : toast(r.error);
  });
  $('rWhole').onclick = () => { $('arc0').value = 0; $('arc1').value = Math.ceil(META.route[META.route.length - 1][2]); };
  $('rHere').onclick = () => {
    const a = V3.started ? V3.arcOf(V3.cam.p[0], V3.cam.p[1]) : 0, d = +$('rHereD').value;
    $('arc0').value = Math.max(0, Math.round(a - d)); $('arc1').value = Math.round(a + d);
  };
  async function accept() { const r = await api('/mesh/accept', {}); toast(r.ok ? `Applied (${(r.changed || 0).toLocaleString()} triangles / vertices changed)` : 'Nothing to apply'); }
  async function discard() { await api('/mesh/discard', {}); toast('Dropped'); }
  async function undo() { const r = await api('/mesh/undo', {}); toast(r.ok ? `Undone: ${r.label}` : 'Nothing to undo'); }
  async function redo() { const r = await api('/mesh/redo', {}); toast(r.ok ? `Redone: ${r.label}` : 'Nothing to redo'); }
  $('pAccept').onclick = accept; $('pDiscard').onclick = discard;
  $('bExport').onclick = async () => {
    if (!confirm('Write the edited mesh (per part + merged) to the project exports/ folder?')) return;
    const r = await api('/mesh/export', {obj: $('exportObj').checked}); toast(r.ok ? 'Exporting...' : r.error);
  };
  $('bRecolour').onclick = async () => { const r = await api('/mesh/recolour', {}); toast(r.ok ? 'Re-projecting ground colours...' : r.error); };
  function segParams() {
    const hex = ($('segColor') && $('segColor').value) || '#8cd2ff';
    const rgb = [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
    return {
      arc0: +$('arc0').value || 0,
      arc1: +$('arc1').value || 1e9,
      margin: $('segMargin') ? +$('segMargin').value : 0.0,
      hmax: $('segMaxH') ? +$('segMaxH').value : 0.6,
      hmin: -0.8,
      rgb,
      alpha: $('segAlpha') ? +$('segAlpha').value : 0.5,
      blend: ($('segBlend') && $('segBlend').checked) ? 0.45 : 0.0,
      save_raw: $('segSaveRaw') ? $('segSaveRaw').checked : true,
      save_labelled: $('segSaveLabelled') ? $('segSaveLabelled').checked : true,
      save_road_only: true,
      save_zup: true,
      save_parts: $('segSaveParts') ? $('segSaveParts').checked : false,
    };
  }
  if ($('segColor')) $('segColor').oninput = () => V3.redraw();
  if ($('segAlpha')) $('segAlpha').oninput = () => V3.redraw();
  if ($('bSegPreview')) $('bSegPreview').onclick = async () => {
    if (typeof dirty !== 'undefined' && dirty && typeof save === 'function') await save();
    const r = await api('/mesh/propose', {tool: 'road_seg', params: segParams()});
    r.ok ? toast('Computing Road segmentation preview (shown in 3D with light colour & half transparency)...') : toast(r.error);
  };
  if ($('bSegExport')) $('bSegExport').onclick = async () => {
    const p = segParams();
    if (!p.save_raw && !p.save_labelled) { toast('Check at least one save option (raw edited mesh or Road-labelled PLY)'); return; }
    const msg = [
      'Segment Road inside lane lines and save PLY files?',
      '',
      `1. Raw edited mesh/cloud (with current studio edits applied): ${p.save_raw ? 'YES' : 'NO'}`,
      `2. Road-segmented labelled PLY (class Road=1, RGBA=[${p.rgb.join(', ')}, ${Math.round(p.alpha * 255)}] + road-only PLY): ${p.save_labelled ? 'YES' : 'NO'}`,
      `3. Per-part PLYs (9 parts): ${p.save_parts ? 'YES' : 'NO'}`,
    ].join('\n');
    if (!confirm(msg)) return;
    if (typeof dirty !== 'undefined' && dirty && typeof save === 'function') await save();
    const r = await api('/mesh/segment_road', p);
    r.ok ? toast('Segmenting road & saving PLY files...') : toast(r.error);
  };

  // ------------------------------------------------------------------------------ status panel
  let lastResult = null;
  V3.onStatus.push(st => {
    prop = st.proposal;
    $('pAccept').disabled = $('pDiscard').disabled = !prop;
    $('propInfo').innerHTML = prop ? (prop.tool === 'road_seg'
      ? `<b>${prop.label}</b><br>segmented <b>${prop.stats.delete.toLocaleString()}</b> triangles as <b>class Road (1)</b> (shown in light colour, half transparency)`
      : `<b>${prop.label}</b><br>remove <b>${prop.stats.delete.toLocaleString()}</b> triangles (red)` +
        (prop.stats.moved ? `, reshape <b>${prop.stats.moved.toLocaleString()}</b> vertices (shown as the result)` : '') +
        (prop.stats.patch_tris ? `, close holes with <b>${prop.stats.patch_tris.toLocaleString()}</b> triangles (green)` : '') + (prop.stats.cell ? `, cell ${prop.stats.cell} m` : ''))
      : 'None. Previews and selections show here (red = will be removed, green = new surface).';
    busy = st.busy;
    $('jobInfo').innerHTML = st.busy ? `<span style="color:#ffb020">${st.msg} ${(st.progress * 100).toFixed(0)}%</span>` :
      st.error ? `<span style="color:#ff5d5d">error: ${st.error}</span>` : '';
    if (st.result && st.result !== lastResult && JSON.stringify(st.result) !== JSON.stringify(lastResult)) {
      lastResult = st.result;
      if (st.result.road_seg_export) {
        const rs = st.result;
        toast(`Road segmented & saved in ${rs.seconds}s: ${rs.road_triangles.toLocaleString()} Road tris (${rs.road_vertices.toLocaleString()} Road verts)`);
        const filesHtml = (rs.files || []).map(f => `<div>&bull; <code>${f.split('/').slice(-2).join('/')}</code></div>`).join('');
        $('jobInfo').innerHTML = `<span style="color:#3ecf8e">Saved ${rs.road_triangles.toLocaleString()} Road triangles (${rs.seconds}s)</span>`;
        if ($('segResult')) $('segResult').innerHTML = `<div><b>Saved (${rs.seconds}s):</b> ${rs.road_triangles.toLocaleString()} / ${rs.triangles.toLocaleString()} tris labelled <b>Road (1)</b>${filesHtml}</div>`;
      } else if (st.result.route) {
        toast(`Mesh exported: ${st.result.triangles.toLocaleString()} triangles`);
        const sp = q => '…/' + String(q).split('/').filter(Boolean).slice(-2).join('/');
        $('jobInfo').innerHTML = `exported to <span title="${st.result.parts}">${sp(st.result.parts)}</span> and <span title="${st.result.route}">${sp(st.result.route)}</span>`;
      }
    }
    const s = st.summary;
    $('meshSummary').innerHTML = `smoothed / reshaped <b>${(s.moved || 0).toLocaleString()}</b> vertices; removed <b>${s.deleted.toLocaleString()}</b> of ${(s.total / 1e6).toFixed(1)} M triangles; ${s.patches} generated patches (${s.patch_tris.toLocaleString()} triangles)`;
    $('undoList').innerHTML = s.undo.length ? 'history: ' + s.undo.slice(-6).map(x => `<div>&bull; ${x}</div>`).join('') : '';
  });

  cv.addEventListener('dblclick', e => { if (tool !== 'lanes') return; const r = cv.getBoundingClientRect(); Lanes3D.dbl(e.clientX - r.left, e.clientY - r.top); });
  // undo / redo: lanes while editing lanes, mesh otherwise
  const api3 = {
    requestFill, placeAt: (x, y, yaw) => V3.placeAt(x, y, yaw),
    undo: () => tool === 'lanes' ? (window.undo(), V3.redraw()) : undo(),
    redo: () => tool === 'lanes' ? (window.redo(), V3.redraw()) : redo(),
    get tool() { return tool; },
  };
  return api3;
})();
document.querySelector('button.tool3[data-tool="nav"]').classList.add('on');
$('hint3').textContent = 'Navigate: drag to look around, W/A/S/D fly, E/Q up/down, arrows turn (360°), wheel forward/back, double-click flies to a point, T drive mode.';
