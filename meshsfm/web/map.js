
'use strict';
// ------------------------------------------------------------------------------------------ setup
const TYPES = [
  {id:'left_edge',    name:'Left edge',    color:'#ff3b3b', dash:[]},
  {id:'right_edge',   name:'Right edge',   color:'#2f86ff', dash:[]},
  {id:'centre_line',  name:'Centre line',  color:'#ffd400', dash:[]},
  {id:'lane_divider', name:'Lane divider', color:'#ffffff', dash:[12,9]},
  {id:'stop_line',    name:'Stop line',    color:'#ff8c00', dash:[]},
  {id:'other',        name:'Other',        color:'#ff4dff', dash:[5,5]},
  {id:'drive_path',   name:'Drive path',   color:'#3ecf8e', dash:[14,6]},
];
const TYPE = Object.fromEntries(TYPES.map(t => [t.id, t]));
const cv = $('c'), ctx = cv.getContext('2d');
let W = 0, H = 0, DPR = 1;
let ROUTE = [];            // ROUTE: [x, z, arc, height]
let lines = [], reviewed = new Set(), nextId = 1;
let view = {cx: 0, cz: 0, scale: 20, rot: 0};   // scale = screen px per metre
let tool = 'edit';
let sel = {line: null, nodes: new Set()};
let dirty = false, lastSave = null, needDraw = true;
let undoStack = [], redoStack = [];
const SAMPLE = 0.1;  // display sampling of curves (m)

// ------------------------------------------------------------------------------------- geometry
function w2s(x, z) {
  const mx = x - view.cx, my = -(z - view.cz), c = Math.cos(view.rot), s = Math.sin(view.rot);
  return [W / 2 + view.scale * (c * mx - s * my), H / 2 + view.scale * (s * mx + c * my)];
}
function s2w(px, py) {
  const sx = (px - W / 2) / view.scale, sy = (py - H / 2) / view.scale, c = Math.cos(view.rot), s = Math.sin(view.rot);
  const mx = c * sx + s * sy, my = -s * sx + c * sy;
  return [view.cx + mx, view.cz - my];
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// centripetal Catmull-Rom through keynodes (identical to lane_annotator.py catmull_rom)
function curve(nodes, smooth, step) {
  const P = nodes, n = P.length, out = [], seg = [];
  if (n === 0) return {pts: out, seg};
  out.push(P[0].slice()); seg.push(0);
  if (n === 1) return {pts: out, seg};
  if (!smooth || n === 2) {
    for (let i = 0; i < n - 1; i++) {
      const a = P[i], b = P[i + 1], k = Math.max(2, Math.ceil(dist(a, b) / step));
      for (let j = 1; j <= k; j++) { out.push([a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k]); seg.push(i); }
    }
    return {pts: out, seg};
  }
  const E = [[2 * P[0][0] - P[1][0], 2 * P[0][1] - P[1][1]], ...P, [2 * P[n - 1][0] - P[n - 2][0], 2 * P[n - 1][1] - P[n - 2][1]]];
  for (let i = 0; i < n - 1; i++) {
    const p0 = E[i], p1 = E[i + 1], p2 = E[i + 2], p3 = E[i + 3];
    const t0 = 0, t1 = t0 + Math.sqrt(Math.max(dist(p0, p1), 1e-6)), t2 = t1 + Math.sqrt(Math.max(dist(p1, p2), 1e-6)), t3 = t2 + Math.sqrt(Math.max(dist(p2, p3), 1e-6));
    const k = Math.max(2, Math.ceil(dist(p1, p2) / step));
    for (let j = 1; j <= k; j++) {
      const t = t1 + (t2 - t1) * j / k, r = [];
      for (let d = 0; d < 2; d++) {
        const A1 = (t1 - t) / (t1 - t0) * p0[d] + (t - t0) / (t1 - t0) * p1[d];
        const A2 = (t2 - t) / (t2 - t1) * p1[d] + (t - t1) / (t2 - t1) * p2[d];
        const A3 = (t3 - t) / (t3 - t2) * p2[d] + (t - t2) / (t3 - t2) * p3[d];
        const B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2;
        const B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3;
        r.push((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2);
      }
      out.push(r); seg.push(i);
    }
  }
  return {pts: out, seg};
}
const cache = new Map();
function samples(ln) {
  const key = JSON.stringify(ln.nodes) + (ln.smooth ? 1 : 0);
  const c = cache.get(ln.id);
  if (c && c.key === key) return c;
  const s = curve(ln.nodes, ln.smooth, SAMPLE);
  let L = 0; for (let i = 1; i < s.pts.length; i++) L += dist(s.pts[i - 1], s.pts[i]);
  s.key = key; s.len = L; s.bbox = bboxOf(s.pts);
  cache.set(ln.id, s);
  return s;
}
function bboxOf(P) {
  let a = [Infinity, Infinity, -Infinity, -Infinity];
  for (const p of P) { a[0] = Math.min(a[0], p[0]); a[1] = Math.min(a[1], p[1]); a[2] = Math.max(a[2], p[0]); a[3] = Math.max(a[3], p[1]); }
  return a;
}
function douglasPeucker(P, tol) {
  if (P.length < 3) return P.map(p => p.slice());
  const keep = new Uint8Array(P.length); keep[0] = keep[P.length - 1] = 1;
  const st = [[0, P.length - 1]];
  while (st.length) {
    const [a, b] = st.pop(); if (b <= a + 1) continue;
    const ax = P[b][0] - P[a][0], az = P[b][1] - P[a][1], L = Math.hypot(ax, az) + 1e-12;
    let best = -1, bi = -1;
    for (let i = a + 1; i < b; i++) { const d = Math.abs(ax * (P[i][1] - P[a][1]) - az * (P[i][0] - P[a][0])) / L; if (d > best) { best = d; bi = i; } }
    if (best > tol) { keep[bi] = 1; st.push([a, bi], [bi, b]); }
  }
  return P.filter((_, i) => keep[i]).map(p => p.slice());
}
// route: nearest point, arc and lateral offset (+ = right of the driving direction)
function routeInfo(x, z) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < ROUTE.length; i++) { const d = (ROUTE[i][0] - x) ** 2 + (ROUTE[i][1] - z) ** 2; if (d < bd) { bd = d; bi = i; } }
  const i0 = Math.max(0, bi - 1), i1 = Math.min(ROUTE.length - 1, bi + 1);
  let fx = ROUTE[i1][0] - ROUTE[i0][0], fz = ROUTE[i1][1] - ROUTE[i0][1]; const L = Math.hypot(fx, fz) || 1; fx /= L; fz /= L;
  const rx = x - ROUTE[bi][0], rz = z - ROUTE[bi][1];
  return {i: bi, arc: ROUTE[bi][2] + rx * fx + rz * fz, lat: rx * fz - rz * fx, fx, fz};
}
function routeAt(arc) {
  let lo = 0, hi = ROUTE.length - 1;
  arc = Math.max(0, Math.min(ROUTE[hi][2], arc));
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ROUTE[m][2] < arc) lo = m; else hi = m; }
  const t = (arc - ROUTE[lo][2]) / Math.max(ROUTE[hi][2] - ROUTE[lo][2], 1e-9);
  return {x: ROUTE[lo][0] + t * (ROUTE[hi][0] - ROUTE[lo][0]), z: ROUTE[lo][1] + t * (ROUTE[hi][1] - ROUTE[lo][1]),
          fx: ROUTE[hi][0] - ROUTE[lo][0], fz: ROUTE[hi][1] - ROUTE[lo][1]};
}

// --------------------------------------------------------------------------------------- tiles
const tiles = new Map(); let tileTick = 0;
let tileSet = {};
function tileImg(L, i, j) {
  const key = `${L}/${i}_${j}`;
  if (!tileSet[L] || !tileSet[L].has(`${i}_${j}`)) return null;
  let t = tiles.get(key);
  if (!t) {
    const img = new Image(); t = {img, ok: false, used: 0};
    img.onload = () => { t.ok = true; needDraw = true; };
    img.src = `/tile/${L}/${i}_${j}.jpg`;
    tiles.set(key, t);
    if (tiles.size > 500) {  // drop least recently used
      const arr = [...tiles.entries()].sort((a, b) => a[1].used - b[1].used);
      for (const [k] of arr.slice(0, 100)) tiles.delete(k);
    }
  }
  t.used = tileTick;
  return t.ok ? t.img : null;
}
function drawTiles() {
  tileTick++;
  const px0 = META.px0, T = META.tile;
  let L = Math.floor(Math.log2(1 / (px0 * view.scale)));
  L = Math.max(0, Math.min(META.levels - 1, L));
  const corners = [[0, 0], [W, 0], [0, H], [W, H]].map(p => s2w(p[0], p[1]));
  const xs = corners.map(c => c[0]), zs = corners.map(c => c[1]);
  const opac = +$('vOpacity').value;
  ctx.save();
  ctx.globalAlpha = opac;
  ctx.filter = `brightness(${$('vBright').value}) contrast(${$('vContrast').value}) saturate(${$('vSat').value})`;
  for (const lev of [Math.min(L + 2, META.levels - 1), Math.min(L + 1, META.levels - 1), L]) {
    const pxL = px0 * 2 ** lev, tw = T * pxL;
    const i0 = Math.floor((Math.min(...xs) - META.x0) / tw), i1 = Math.floor((Math.max(...xs) - META.x0) / tw);
    const j0 = Math.floor((META.ztop - Math.max(...zs)) / tw), j1 = Math.floor((META.ztop - Math.min(...zs)) / tw);
    if ((i1 - i0 + 1) * (j1 - j0 + 1) > 400) continue;
    const c = Math.cos(view.rot), s = Math.sin(view.rot), k = view.scale * pxL;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const img = tileImg(lev, i, j); if (!img) continue;
      const X0 = META.x0 + i * tw - view.cx, Y0 = j * tw + view.cz - META.ztop;
      ctx.setTransform(DPR * k * c, DPR * k * s, -DPR * k * s, DPR * k * c,
                       DPR * (W / 2 + view.scale * (c * X0 - s * Y0)), DPR * (H / 2 + view.scale * (s * X0 + c * Y0)));
      ctx.drawImage(img, 0, 0, T + 0.5, T + 0.5);
    }
  }
  ctx.restore();
  return L;
}

// ------------------------------------------------------------------------------------- drawing
let mouse = {x: 0, y: 0, w: [0, 0], inside: false};
let drag = null;          // active mouse interaction
let drawing = null;       // pen / free draw in progress: {line, atStart, pts?}
let measure = [];         // measure tool points
let joinFirst = null;     // join tool: first chosen end
let hover = null;         // hovered thing for highlights
let snapTarget = null;
let fillPts = [];          // fill tool contour (world x, y)
let curLevel = 0;

function draw() {
  needDraw = false;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = '#26282c'; ctx.fillRect(0, 0, W, H);
  if (!META) return;
  curLevel = drawTiles();
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  if ($('vRoute').checked) drawRoute();
  const lw = +$('vWidth').value;
  const visLines = lines.filter(l => l.visible !== false);
  const vb = viewBBox(60);
  for (const ln of visLines) if (ln !== sel.line) drawLine(ln, lw, false, vb);
  if (sel.line && sel.line.visible !== false) drawLine(sel.line, lw, true, vb);
  // pen preview
  if (drawing && tool === 'pen' && drawing.line.nodes.length) {
    const end = drawing.atStart ? drawing.line.nodes[0] : drawing.line.nodes[drawing.line.nodes.length - 1];
    const a = w2s(end[0], end[1]), b = snapTarget ? w2s(snapTarget[0], snapTarget[1]) : [mouse.x, mouse.y];
    ctx.setLineDash([6, 5]); ctx.strokeStyle = TYPE[drawing.line.type].color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
  }
  if (drawing && tool === 'free' && drawing.pts && drawing.pts.length > 1) {
    ctx.strokeStyle = TYPE[drawing.line ? drawing.line.type : $('newType').value].color; ctx.lineWidth = 2; ctx.beginPath();
    drawing.pts.forEach((p, i) => { const s = w2s(p[0], p[1]); i ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); }); ctx.stroke();
  }
  if (fillPts.length) {
    ctx.beginPath(); fillPts.forEach((p, i) => { const s = w2s(p[0], p[1]); i ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); });
    if (tool === 'fill' && mouse.inside) ctx.lineTo(mouse.x, mouse.y);
    ctx.closePath(); ctx.fillStyle = '#3ecf8e33'; ctx.fill(); ctx.strokeStyle = '#3ecf8e'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]);
    fillPts.forEach(p => { const s = w2s(p[0], p[1]); ctx.fillStyle = '#3ecf8e'; ctx.beginPath(); ctx.arc(s[0], s[1], 4, 0, 7); ctx.fill(); });
  }
  if (window.CAM3D) {
    const s = w2s(CAM3D.x, CAM3D.y), a = Math.atan2(-Math.sin(CAM3D.yaw), Math.cos(CAM3D.yaw)) + view.rot;
    ctx.save(); ctx.translate(s[0], s[1]); ctx.rotate(a); ctx.fillStyle = '#ff9a3c'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(16, 0); ctx.lineTo(-8, -9); ctx.lineTo(-3, 0); ctx.lineTo(-8, 9); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
    label('3D camera', s[0] + 12, s[1] + 22, '#ff9a3c');
  }
  if (snapTarget) { const s = w2s(snapTarget[0], snapTarget[1]); ring(s, 11, '#3ecf8e'); }
  if (joinFirst) { const e = endPoint(joinFirst.line, joinFirst.atStart), s = w2s(e[0], e[1]); ring(s, 12, '#ffb020'); }
  if (hover && hover.kind === 'end' && (tool === 'join' || tool === 'pen' || tool === 'free')) { const s = w2s(hover.p[0], hover.p[1]); ring(s, 10, '#ffffffaa'); }
  if (hover && hover.kind === 'seg' && (tool === 'insert' || tool === 'split')) { const s = w2s(hover.p[0], hover.p[1]); ring(s, 6, '#ffffff'); }
  // measure
  if (tool === 'measure' && measure.length) {
    const pts = measure.length === 1 ? [measure[0], mouse.w] : measure;
    const a = w2s(...pts[0]), b = w2s(...pts[1]);
    ctx.strokeStyle = '#3ecf8e'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    label(`${dist(pts[0], pts[1]).toFixed(2)} m`, (a[0] + b[0]) / 2 + 8, (a[1] + b[1]) / 2 - 8, '#3ecf8e');
  }
  // box select
  if (drag && drag.kind === 'box') {
    ctx.strokeStyle = '#4da3ff'; ctx.fillStyle = '#4da3ff22'; ctx.lineWidth = 1;
    const x = Math.min(drag.x0, mouse.x), y = Math.min(drag.y0, mouse.y), w = Math.abs(mouse.x - drag.x0), h = Math.abs(mouse.y - drag.y0);
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  }
  drawMini();
  updateStatus();
}
function viewBBox(padPx) {
  const c = [[-padPx, -padPx], [W + padPx, -padPx], [-padPx, H + padPx], [W + padPx, H + padPx]].map(p => s2w(p[0], p[1]));
  return [Math.min(...c.map(p => p[0])), Math.min(...c.map(p => p[1])), Math.max(...c.map(p => p[0])), Math.max(...c.map(p => p[1]))];
}
function ring(s, r, col) { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s[0], s[1], r, 0, 7); ctx.stroke(); }
function label(t, x, y, col) {
  ctx.font = '12px system-ui'; const w = ctx.measureText(t).width;
  ctx.fillStyle = '#0d0f13d0'; ctx.fillRect(x - 3, y - 12, w + 6, 16); ctx.fillStyle = col || '#fff'; ctx.fillText(t, x, y);
}
function drawRoute() {
  ctx.strokeStyle = '#ff9a3c88'; ctx.lineWidth = 1; ctx.setLineDash([4, 6]); ctx.beginPath();
  const vb = viewBBox(50); let on = false;
  for (const r of ROUTE) {
    if (r[0] < vb[0] || r[0] > vb[2] || r[1] < vb[1] || r[1] > vb[3]) { on = false; continue; }
    const s = w2s(r[0], r[1]); on ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); on = true;
  }
  ctx.stroke(); ctx.setLineDash([]);
  const every = view.scale > 8 ? 10 : view.scale > 2 ? 50 : view.scale > 0.6 ? 100 : 500;
  for (let a = 0; a <= ROUTE[ROUTE.length - 1][2]; a += every) {
    const r = routeAt(a); if (r.x < vb[0] || r.x > vb[2] || r.z < vb[1] || r.z > vb[3]) continue;
    const s = w2s(r.x, r.z); ctx.fillStyle = '#ff9a3c'; ctx.beginPath(); ctx.arc(s[0], s[1], 2.5, 0, 7); ctx.fill();
    ctx.font = '11px system-ui'; ctx.fillStyle = '#ffcf9e'; ctx.fillText(`${a} m`, s[0] + 5, s[1] - 4);
  }
}
function drawLine(ln, lw, selected, vb) {
  const t = TYPE[ln.type] || TYPE.other, S = samples(ln);
  if (S.bbox[2] < vb[0] || S.bbox[0] > vb[2] || S.bbox[3] < vb[1] || S.bbox[1] > vb[3]) return;
  // decimate for drawing: about one point per 2 screen px
  const stride = Math.max(1, Math.floor(2 / (view.scale * SAMPLE)));
  const pts = [];
  for (let i = 0; i < S.pts.length; i += stride) pts.push(w2s(S.pts[i][0], S.pts[i][1]));
  if ((S.pts.length - 1) % stride) pts.push(w2s(...S.pts[S.pts.length - 1]));
  const path = () => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); };
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  path(); ctx.strokeStyle = selected ? '#000c' : '#0009'; ctx.lineWidth = lw + (selected ? 4 : 2.5); ctx.stroke();
  if (selected) { path(); ctx.strokeStyle = '#ffffff55'; ctx.lineWidth = lw + 7; ctx.stroke(); }
  path(); ctx.setLineDash(t.dash); ctx.strokeStyle = t.color; ctx.lineWidth = lw + (selected ? 1 : 0); ctx.stroke(); ctx.setLineDash([]);
  if ($('vArrows').checked && pts.length > 1) {
    let acc = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = dist(pts[i - 1], pts[i]); acc += d;
      if (acc > 140 && d > 0.5) {
        acc = 0; const a = Math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0]);
        ctx.save(); ctx.translate(pts[i][0], pts[i][1]); ctx.rotate(a); ctx.fillStyle = t.color; ctx.strokeStyle = '#000a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(-4, -5); ctx.lineTo(-4, 5); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
      }
    }
  }
  if (selected || ($('vNodes').checked && view.scale > 3)) {  // other lines' keynodes only when zoomed in
    const big = selected ? 1 : 0.75;
    ln.nodes.forEach((n, i) => {
      if (n[0] < vb[0] || n[0] > vb[2] || n[1] < vb[1] || n[1] > vb[3]) return;
      const s = w2s(n[0], n[1]), end = i === 0 || i === ln.nodes.length - 1, r = (end ? 5.5 : 4) * big;
      const on = selected && sel.nodes.has(i);
      ctx.fillStyle = on ? '#ffffff' : (end ? t.color : '#11151b'); ctx.strokeStyle = on ? '#4da3ff' : t.color; ctx.lineWidth = on ? 2.5 : 1.6;
      ctx.beginPath(); if (end) ctx.arc(s[0], s[1], r, 0, 7); else ctx.rect(s[0] - r, s[1] - r, 2 * r, 2 * r); ctx.fill(); ctx.stroke();
    });
  }
  if ($('vLabels').checked || selected) { const s = w2s(...ln.nodes[0]); label(`${ln.name || t.name}`, s[0] + 8, s[1] - 8, t.color); }
  if (ln.locked) { const s = w2s(...ln.nodes[0]); label('locked', s[0] + 8, s[1] + 12, '#aaa'); }
}

// ----------------------------------------------------------------------------------- minimap
let miniFit = null;
function drawMini() {
  const m = $('mini'), c = m.getContext('2d'); c.clearRect(0, 0, m.width, m.height);
  if (!ROUTE.length) return;
  if (!miniFit) {
    const b = bboxOf(ROUTE.map(r => [r[0], r[1]])), s = 0.9 * Math.min(m.width / (b[2] - b[0]), m.height / (b[3] - b[1]));
    miniFit = {b, s, ox: (m.width - s * (b[2] - b[0])) / 2, oy: (m.height - s * (b[3] - b[1])) / 2};
  }
  const f = miniFit, P = (x, z) => [f.ox + (x - f.b[0]) * f.s, m.height - (f.oy + (z - f.b[1]) * f.s)];
  c.lineWidth = 3;
  for (let i = 1; i < ROUTE.length; i++) {
    const seg = Math.floor(ROUTE[i][2] / 25);
    c.strokeStyle = reviewed.has(seg) ? '#3ecf8e' : '#5b6578';
    c.beginPath(); c.moveTo(...P(ROUTE[i - 1][0], ROUTE[i - 1][1])); c.lineTo(...P(ROUTE[i][0], ROUTE[i][1])); c.stroke();
  }
  const st = P(ROUTE[0][0], ROUTE[0][1]); c.fillStyle = '#fff'; c.fillText('start', st[0] + 4, st[1]);
  const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(p => P(...s2w(p[0], p[1])));
  c.strokeStyle = '#4da3ff'; c.lineWidth = 1.5; c.beginPath(); corners.forEach((p, i) => i ? c.lineTo(...p) : c.moveTo(...p)); c.closePath(); c.stroke();
  const cc = P(view.cx, view.cz); c.fillStyle = '#4da3ff'; c.beginPath(); c.arc(cc[0], cc[1], 3.5, 0, 7); c.fill();
}
$('mini').addEventListener('mousedown', e => {
  e.stopPropagation(); if (!miniFit) return;
  const r = $('mini').getBoundingClientRect(), f = miniFit, m = $('mini');
  const x = (e.clientX - r.left) * m.width / r.width, y = (e.clientY - r.top) * m.height / r.height;
  const wx = f.b[0] + (x - f.ox) / f.s, wz = f.b[1] + (m.height - y - f.oy) / f.s;
  const ri = routeInfo(wx, wz); goToArc(ri.arc);
});

// --------------------------------------------------------------------------------- hit testing
function hitNode(px, py, onlyLine) {
  let best = null, bd = 10;
  const cand = onlyLine ? [onlyLine] : [sel.line, ...lines.filter(l => l !== sel.line)].filter(Boolean);
  for (const ln of cand) {
    if (ln.visible === false || ln.locked) continue;
    if (!onlyLine && ln !== sel.line && !($('vNodes').checked && view.scale > 3)) continue;
    ln.nodes.forEach((n, i) => { const s = w2s(n[0], n[1]), d = Math.hypot(s[0] - px, s[1] - py); if (d < bd) { bd = d; best = {line: ln, idx: i}; } });
    if (best && best.line === sel.line) return best;  // prefer the selected line
  }
  return best;
}
function hitSeg(px, py, tolPx = 7, onlyUnlocked = true) {
  let best = null, bd = tolPx;
  const w = s2w(px, py), r = tolPx / view.scale;
  for (const ln of lines) {
    if (ln.visible === false || (onlyUnlocked && ln.locked) || ln.nodes.length < 2) continue;
    const S = samples(ln);
    if (w[0] < S.bbox[0] - r || w[0] > S.bbox[2] + r || w[1] < S.bbox[1] - r || w[1] > S.bbox[3] + r) continue;
    for (let i = 1; i < S.pts.length; i++) {
      const a = S.pts[i - 1], b = S.pts[i];
      if (Math.min(a[0], b[0]) > w[0] + r || Math.max(a[0], b[0]) < w[0] - r || Math.min(a[1], b[1]) > w[1] + r || Math.max(a[1], b[1]) < w[1] - r) continue;
      const vx = b[0] - a[0], vz = b[1] - a[1], L2 = vx * vx + vz * vz || 1e-12;
      const t = Math.max(0, Math.min(1, ((w[0] - a[0]) * vx + (w[1] - a[1]) * vz) / L2));
      const p = [a[0] + t * vx, a[1] + t * vz], d = dist(p, w) * view.scale;
      if (d < bd) { bd = d; best = {line: ln, seg: S.seg[i], p}; }
    }
  }
  return best;
}
function hitEnd(px, py, exclude) {
  let best = null, bd = 12;
  for (const ln of lines) {
    if (ln.visible === false || ln.locked || ln === exclude || ln.nodes.length < 1) continue;
    for (const atStart of [true, false]) {
      const p = endPoint(ln, atStart), s = w2s(p[0], p[1]), d = Math.hypot(s[0] - px, s[1] - py);
      if (d < bd) { bd = d; best = {kind: 'end', line: ln, atStart, p}; }
    }
  }
  return best;
}
const endPoint = (ln, atStart) => atStart ? ln.nodes[0] : ln.nodes[ln.nodes.length - 1];
function snapAt(px, py, exclude) {
  if (!$('snapEnds').checked) return null;
  const h = hitEnd(px, py, exclude); return h ? h.p.slice() : null;
}

// ---------------------------------------------------------------------------------- undo/redo
const snapshot = () => JSON.stringify({lines: lines.map(cleanLine), reviewed: [...reviewed], nextId});
function cleanLine(l) { return {id: l.id, type: l.type, name: l.name, nodes: l.nodes, smooth: l.smooth, visible: l.visible !== false, locked: !!l.locked}; }
function pushUndo() { undoStack.push(snapshot()); if (undoStack.length > 300) undoStack.shift(); redoStack = []; markDirty(); }
function restore(s) {
  const o = JSON.parse(s), selId = sel.line && sel.line.id;
  lines = o.lines; reviewed = new Set(o.reviewed); nextId = o.nextId;
  sel.line = lines.find(l => l.id === selId) || null; sel.nodes.clear(); drawing = null;
  markDirty(); refreshPanel();
}
function undo() { if (!undoStack.length) return; redoStack.push(snapshot()); restore(undoStack.pop()); toast('Undo'); }
function redo() { if (!redoStack.length) return; undoStack.push(snapshot()); restore(redoStack.pop()); toast('Redo'); }
function markDirty() { dirty = true; needDraw = true; updateSaveState(); scheduleCoverage(); }

// ------------------------------------------------------------------------------------ editing
function newLine(type, nodes) {
  const t = type || $('newType').value;
  const n = lines.filter(l => l.type === t).length + 1;
  const ln = {id: nextId++, type: t, name: `${TYPE[t].name} ${n}`, nodes: nodes || [], smooth: true, visible: true, locked: false};
  lines.push(ln); return ln;
}
function selectLine(ln, keepNodes) { sel.line = ln; if (!keepNodes) sel.nodes.clear(); refreshPanel(); needDraw = true; }
function deleteNodes(ln, idxs) {
  if (!ln || !idxs.size) return;
  pushUndo();
  ln.nodes = ln.nodes.filter((_, i) => !idxs.has(i)); sel.nodes.clear();
  if (ln.nodes.length < 2) { lines = lines.filter(l => l !== ln); if (sel.line === ln) sel.line = null; toast('Line removed (fewer than 2 keynodes left)'); }
  refreshPanel();
}
function deleteLine(ln) { if (!ln) return; pushUndo(); lines = lines.filter(l => l !== ln); if (sel.line === ln) sel.line = null; sel.nodes.clear(); refreshPanel(); }
function insertNode(hit) { pushUndo(); hit.line.nodes.splice(hit.seg + 1, 0, hit.p.slice()); selectLine(hit.line); sel.nodes.add(hit.seg + 1); return hit.seg + 1; }
function splitAt(ln, idx) {
  if (idx <= 0 || idx >= ln.nodes.length - 1) { toast('Pick an inner keynode or a point on the line'); return; }
  const b = newLine(ln.type, ln.nodes.slice(idx).map(p => p.slice()));
  b.smooth = ln.smooth; b.name = (ln.name || '') + ' (b)';
  ln.nodes = ln.nodes.slice(0, idx + 1);
  lines.splice(lines.indexOf(b), 1); lines.splice(lines.indexOf(ln) + 1, 0, b);
  selectLine(b); toast('Split into two lines');
}
function joinLines(a, aStart, b, bStart) {
  pushUndo();
  let A = a.nodes.slice(), B = b.nodes.slice();
  if (aStart) A.reverse(); if (!bStart) B.reverse();
  if (dist(A[A.length - 1], B[0]) < 0.05) B.shift();
  a.nodes = A.concat(B);
  lines = lines.filter(l => l !== b); selectLine(a); toast('Joined');
}
function reverseLine(ln) { if (!ln) return; pushUndo(); ln.nodes.reverse(); sel.nodes.clear(); toast('Reversed'); }
function simplifyLine(ln, tol) {
  if (!ln) return; pushUndo();
  const before = ln.nodes.length, S = curve(ln.nodes, ln.smooth, 0.05).pts;
  ln.nodes = douglasPeucker(S, tol); sel.nodes.clear(); toast(`Simplified: ${before} -> ${ln.nodes.length} keynodes`); refreshPanel();
}
function densifyLine(ln, step, mode, around) {
  // Adds keynodes ON the current curve (existing keynodes stay), every `step` m:
  //   'span'   between the first and last selected keynode
  //   'around' within +-around m of the (single) selected keynode
  //   'all'    the whole line
  if (!ln) return;
  const S = curve(ln.nodes, ln.smooth, 0.02), P = S.pts, n = ln.nodes.length;
  const cum = new Float64Array(P.length), nodeL = new Float64Array(n);
  for (let j = 1; j < P.length; j++) { cum[j] = cum[j - 1] + dist(P[j - 1], P[j]); if (j === P.length - 1 || S.seg[j + 1] !== S.seg[j]) nodeL[S.seg[j] + 1] = cum[j]; }
  const selIdx = [...sel.nodes].sort((a, b) => a - b);
  if (mode === 'auto') mode = selIdx.length >= 2 ? 'span' : selIdx.length === 1 ? 'around' : 'all';
  let a, b;
  if (mode === 'span') { if (selIdx.length < 2) { toast('Select two keynodes (Shift+click) to densify between them'); return; } a = nodeL[selIdx[0]]; b = nodeL[selIdx[selIdx.length - 1]]; }
  else if (mode === 'around') { if (selIdx.length !== 1) { toast('Select exactly one keynode to densify around it'); return; } a = Math.max(0, nodeL[selIdx[0]] - around); b = Math.min(cum[P.length - 1], nodeL[selIdx[0]] + around); }
  else { a = 0; b = cum[P.length - 1]; }
  const at = s => { let lo = 0, hi = P.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] < s) lo = m; else hi = m; } const t = (s - cum[lo]) / Math.max(cum[hi] - cum[lo], 1e-12); return [P[lo][0] + t * (P[hi][0] - P[lo][0]), P[lo][1] + t * (P[hi][1] - P[lo][1])]; };
  const start = mode === 'around' ? nodeL[selIdx[0]] - Math.floor((nodeL[selIdx[0]] - a) / step) * step : a;
  const add = [];
  for (let s = start; s <= b + 1e-9; s += step) if (s > 1e-6 && s < cum[P.length - 1] - 1e-6 && !nodeL.some(L => Math.abs(L - s) < 0.4 * step)) add.push(s);
  if (!add.length) { toast('Already at least that dense there'); return; }
  pushUndo();
  const all = [...[...nodeL].map((L, i) => [L, ln.nodes[i]]), ...add.map(s => [s, at(s)])].sort((u, v) => u[0] - v[0]);
  const keepSel = new Set(selIdx.map(i => nodeL[i]));
  ln.nodes = all.map(e => e[1].slice());
  sel.nodes.clear(); all.forEach((e, i) => { if (keepSel.has(e[0])) sel.nodes.add(i); });
  toast(`Densify (${mode}): +${add.length} keynodes every ${step} m (${n} -> ${ln.nodes.length})`); refreshPanel();
}
function offsetCopy(ln, d, type) {
  if (!ln) return; pushUndo();
  const S = curve(ln.nodes, ln.smooth, 0.05).pts;
  const off = S.map((p, i) => {
    const a = S[Math.max(0, i - 2)], b = S[Math.min(S.length - 1, i + 2)];
    let fx = b[0] - a[0], fz = b[1] - a[1]; const L = Math.hypot(fx, fz) || 1; fx /= L; fz /= L;
    return [p[0] + d * fz, p[1] - d * fx];   // + d = right of the line direction
  });
  const nl = newLine(type, douglasPeucker(off, 0.02)); nl.smooth = true; selectLine(nl); toast(`Parallel copy at ${d} m`);
}
function addNodeToDrawing(p) {
  const ln = drawing.line;
  if (drawing.atStart) ln.nodes.unshift(p); else ln.nodes.push(p);
  needDraw = true;
}
function finishDrawing() {
  if (!drawing) return;
  const ln = drawing.line;
  if (ln.nodes.length < 2 && drawing.created) { lines = lines.filter(l => l !== ln); if (sel.line === ln) sel.line = null; }
  drawing = null; snapTarget = null; refreshPanel(); markDirty();
}
function continueSelected() {
  const ln = sel.line; if (!ln) { toast('Select a line first'); return; }
  const a = w2s(...ln.nodes[0]), b = w2s(...ln.nodes[ln.nodes.length - 1]);
  const atStart = Math.hypot(a[0] - mouse.x, a[1] - mouse.y) < Math.hypot(b[0] - mouse.x, b[1] - mouse.y);
  setTool('pen'); pushUndo(); drawing = {line: ln, atStart, created: false};
  toast(`Continuing "${ln.name}" from its ${atStart ? 'start' : 'end'}: click to add keynodes, Enter to finish`);
}

// --------------------------------------------------------------------------------------- mouse
function setTool(t) {
  if (drawing && t !== tool) finishDrawing();
  tool = t; measure = []; joinFirst = null;
  document.querySelectorAll('button.tool').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
  cv.style.cursor = {edit: 'default', pen: 'crosshair', free: 'crosshair', insert: 'copy', erase: 'not-allowed', split: 'crosshair', join: 'alias', measure: 'crosshair', fill: 'crosshair'}[t];
  updateHint(); needDraw = true;
}
const HINTS = {
  edit: 'Click a line to select. Drag keynodes. Shift+click / Shift+drag to select several. Double-click a line to insert a keynode. Drag empty map to pan.',
  pen: 'Click to place keynodes (start on a line end to continue it). Enter / double-click / Esc to finish, Backspace removes the last.',
  free: 'Drag to draw; released strokes become keynodes. Start at a line end to extend it. Right-drag pans.',
  insert: 'Click on a line to insert a keynode.',
  erase: 'Click keynodes to delete them.',
  split: 'Click a keynode or a point on a line to split it into two lines.',
  join: 'Click near the end of a line, then near the end of another line.',
  measure: 'Click two points to measure. A third click starts again.',
  fill: 'Fill area: click points around a hole (contour). Enter = generate the patch (shown in 3D), Backspace = remove last point, Esc = clear.',
};
function updateHint() { $('hint').textContent = HINTS[tool]; }

let spaceDown = false;
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('mousedown', e => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, w = s2w(x, y);
  mouse = {x, y, w, inside: true};
  if (e.button === 2) { drag = {kind: 'rot', x0: x, y0: y, rot0: view.rot, moved: false}; return; }      // right-drag: rotate the map
  if (e.button === 1 || spaceDown) { drag = {kind: 'pan', x0: x, y0: y, cx: view.cx, cz: view.cz, moved: false}; return; }
  if (e.button !== 0) return;
  if (tool === 'edit') {
    const hn = hitNode(x, y);
    if (hn) {
      if (e.shiftKey) { if (sel.line !== hn.line) selectLine(hn.line); sel.nodes.has(hn.idx) ? sel.nodes.delete(hn.idx) : sel.nodes.add(hn.idx); needDraw = true; refreshPanel(); return; }
      if (sel.line !== hn.line || !sel.nodes.has(hn.idx)) { selectLine(hn.line); sel.nodes.add(hn.idx); }
      drag = {kind: 'nodes', line: hn.line, grab: hn.idx, w0: w, orig: [...sel.nodes].map(i => [i, hn.line.nodes[i].slice()]), moved: false};
      refreshPanel(); return;
    }
    const hs = hitSeg(x, y);
    if (hs && e.altKey) { const i = insertNode(hs); drag = {kind: 'nodes', line: hs.line, grab: i, w0: w, orig: [[i, hs.line.nodes[i].slice()]], moved: true, noUndo: true}; return; }
    if (hs) { selectLine(hs.line); return; }
    if (e.shiftKey) { drag = {kind: 'box', x0: x, y0: y}; return; }
    drag = {kind: 'pan', x0: x, y0: y, cx: view.cx, cz: view.cz, moved: false, deselect: true}; return;
  }
  if (tool === 'pen') { drag = {kind: 'penclick', x0: x, y0: y, cx: view.cx, cz: view.cz, moved: false}; return; }
  if (tool === 'free') {
    const he = hitEnd(x, y);
    pushUndo();
    drawing = {pts: [snapAt(x, y) || w], line: he ? he.line : null, atStart: he ? he.atStart : false};
    drag = {kind: 'free'}; return;
  }
  if (tool === 'insert') { const hs = hitSeg(x, y); if (hs) insertNode(hs); return; }
  if (tool === 'erase') { const hn = hitNode(x, y); if (hn) { selectLine(hn.line); deleteNodes(hn.line, new Set([hn.idx])); } return; }
  if (tool === 'split') {
    const hn = hitNode(x, y);
    if (hn) { pushUndo(); splitAt(hn.line, hn.idx); return; }
    const hs = hitSeg(x, y);
    if (hs) { pushUndo(); hs.line.nodes.splice(hs.seg + 1, 0, hs.p.slice()); splitAt(hs.line, hs.seg + 1); }
    return;
  }
  if (tool === 'join') {
    const he = hitEnd(x, y, joinFirst ? joinFirst.line : null);  // second click: ends of other lines only
    if (!he) { joinFirst = null; needDraw = true; return; }
    if (!joinFirst) { joinFirst = he; selectLine(he.line); toast('Now click near the end of the line to join'); }
    else if (he.line === joinFirst.line) toast('Pick an end of a different line');
    else { joinLines(joinFirst.line, joinFirst.atStart, he.line, he.atStart); joinFirst = null; }
    needDraw = true; return;
  }
  if (tool === 'measure') { if (measure.length >= 2) measure = []; measure.push(w); needDraw = true; return; }
  if (tool === 'fill') { drag = {kind: 'fillclick', x0: x, y0: y, cx: view.cx, cz: view.cz, moved: false}; return; }
});
window.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  mouse.x = x; mouse.y = y; mouse.w = s2w(x, y);
  mouse.inside = x >= 0 && y >= 0 && x < W && y < H;
  if (drag) {
    if (drag.kind === 'rot') {
      if (followRoad) setFollow(false);
      const a0 = Math.atan2(drag.y0 - H / 2, drag.x0 - W / 2), a1 = Math.atan2(y - H / 2, x - W / 2);
      view.rot = drag.rot0 + (Math.hypot(x - W / 2, y - H / 2) > 40 ? a1 - a0 : (x - drag.x0) * 0.01); needDraw = true;
    } else if (drag.kind === 'pan' || drag.kind === 'penclick' || drag.kind === 'fillclick') {
      if (Math.hypot(x - drag.x0, y - drag.y0) > 4) drag.moved = true;
      if (drag.moved) {
        const c = Math.cos(view.rot), s = Math.sin(view.rot), dx = (x - drag.x0) / view.scale, dy = (y - drag.y0) / view.scale;
        view.cx = drag.cx - (c * dx + s * dy); view.cz = drag.cz + (-s * dx + c * dy);
      }
    } else if (drag.kind === 'nodes') {
      if (!drag.moved) { drag.moved = true; }
      if (!drag.undone && !drag.noUndo) { pushUndo(); drag.undone = true; }
      let dx = mouse.w[0] - drag.w0[0], dz = mouse.w[1] - drag.w0[1];
      snapTarget = null;
      const ln = drag.line, isEnd = drag.grab === 0 || drag.grab === ln.nodes.length - 1;
      if (drag.orig.length === 1 && isEnd) {
        const sp = snapAt(x, y, ln); if (sp) { snapTarget = sp; const o = drag.orig[0][1]; dx = sp[0] - o[0]; dz = sp[1] - o[1]; }
      }
      for (const [i, o] of drag.orig) ln.nodes[i] = [o[0] + dx, o[1] + dz];
      markDirty();
    } else if (drag.kind === 'free') {
      const last = drawing.pts[drawing.pts.length - 1], s = w2s(last[0], last[1]);
      if (Math.hypot(s[0] - x, s[1] - y) > 2) drawing.pts.push(mouse.w);
    }
    needDraw = true;
  } else if (mouse.inside) {
    hover = null; snapTarget = null;
    if (tool === 'join' || tool === 'free') hover = hitEnd(x, y);
    else if (tool === 'pen') { hover = drawing ? null : hitEnd(x, y); if (drawing) { const sp = snapAt(x, y, drawing.line); snapTarget = sp; } }
    else if (tool === 'insert' || tool === 'split') { const hs = hitSeg(x, y); if (hs) hover = {kind: 'seg', p: hs.p}; }
    needDraw = true;
  }
  updateStatus();
});
window.addEventListener('mouseup', e => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.kind === 'pan' && !d.moved && d.deselect) { sel.line = null; sel.nodes.clear(); refreshPanel(); }
  if ((d.kind === 'pan' || d.kind === 'penclick' || d.kind === 'fillclick') && d.moved && followRoad) alignToRoad();
  if (d.kind === 'fillclick' && !d.moved) { fillPts.push(mouse.w.slice()); needDraw = true; }
  if (d.kind === 'penclick' && !d.moved) {
    const x = mouse.x, y = mouse.y;
    if (!drawing) {
      const he = hitEnd(x, y);
      pushUndo();
      if (he) { drawing = {line: he.line, atStart: he.atStart, created: false}; selectLine(he.line); toast(`Continuing "${he.line.name}"`); }
      else { const ln = newLine(); ln.nodes.push(snapAt(x, y) || mouse.w); drawing = {line: ln, atStart: false, created: true}; selectLine(ln); }
    } else {
      addNodeToDrawing(snapTarget || mouse.w);
      if (snapTarget) { finishDrawing(); toast('Connected to a line end'); }
    }
    markDirty();
  }
  if (d.kind === 'box') {
    const x0 = Math.min(d.x0, mouse.x), x1 = Math.max(d.x0, mouse.x), y0 = Math.min(d.y0, mouse.y), y1 = Math.max(d.y0, mouse.y);
    const inBox = ln => ln.nodes.map((n, i) => [i, w2s(n[0], n[1])]).filter(([, s]) => s[0] >= x0 && s[0] <= x1 && s[1] >= y0 && s[1] <= y1).map(([i]) => i);
    let ln = sel.line, idx = ln ? inBox(ln) : [];
    if (!idx.length) { let best = null, bc = 0; for (const l of lines) { if (l.visible === false || l.locked) continue; const k = inBox(l); if (k.length > bc) { bc = k.length; best = l; idx = k; } } ln = best; }
    if (ln) { selectLine(ln, ln === sel.line); idx.forEach(i => sel.nodes.add(i)); toast(`${sel.nodes.size} keynodes selected`); }
  }
  if (d.kind === 'free' && drawing) {
    const raw = drawing.pts;
    const end = snapAt(mouse.x, mouse.y, drawing.line); if (end) raw.push(end);
    let nodes = douglasPeucker(raw, +$('freeTol').value);
    if (raw.length < 2 || (nodes.length === 2 && dist(nodes[0], nodes[1]) < 0.05)) { undoStack.pop(); drawing = null; needDraw = true; return; }
    if (drawing.line) {
      const ln = drawing.line;
      if (drawing.atStart) { nodes.reverse(); ln.nodes = nodes.slice(0, -1).concat(ln.nodes); }
      else ln.nodes = ln.nodes.concat(nodes.slice(1));
      selectLine(ln);
    } else { const ln = newLine(null, nodes); selectLine(ln); }
    toast(`Free draw: ${nodes.length} keynodes`);
    drawing = null; markDirty(); refreshPanel();
  }
  if (d.kind === 'nodes') { snapTarget = null; refreshPanel(); }
  needDraw = true;
});
cv.addEventListener('dblclick', e => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (tool === 'pen' && drawing) {
    // the double-click also placed a duplicate node: drop it
    const ln = drawing.line, n = ln.nodes, k = drawing.atStart ? 0 : n.length - 1, k2 = drawing.atStart ? 1 : n.length - 2;
    if (n.length > 2 && dist(n[k], n[k2]) * view.scale < 4) n.splice(k, 1);
    finishDrawing(); return;
  }
  if (tool === 'edit' && !hitNode(x, y)) { const hs = hitSeg(x, y); if (hs) insertNode(hs); }
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  zoomAt(x, y, Math.pow(1.0015, -e.deltaY * (e.deltaMode === 1 ? 30 : 1)));
}, {passive: false});
function zoomAt(x, y, f) {
  const before = s2w(x, y);
  view.scale = Math.max(0.05, Math.min(400, view.scale * f));
  const after = s2w(x, y); view.cx += before[0] - after[0]; view.cz += before[1] - after[1];
  needDraw = true; updateStatus();
}

// ------------------------------------------------------------------------------------ keyboard
window.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' && e.target.type !== 'range' && e.target.type !== 'checkbox' || tag === 'select' || tag === 'textarea') { if (e.key === 'Escape') e.target.blur(); return; }
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  if ($('help').style.display === 'flex' && (k === 'Escape' || k === 'h' || k === 'H' || k === '?')) { $('help').style.display = 'none'; return; }
  if (ctrl && (k === 's' || k === 'S')) { e.preventDefault(); save(); return; }
  if (TAB !== 'map') return;
  if (ctrl && !e.shiftKey && (k === 'z' || k === 'Z')) { e.preventDefault(); undo(); return; }
  if (ctrl && (k === 'y' || k === 'Y' || (e.shiftKey && (k === 'z' || k === 'Z')))) { e.preventDefault(); redo(); return; }
  if (ctrl && (k === 'a' || k === 'A')) { e.preventDefault(); if (sel.line) { sel.line.nodes.forEach((_, i) => sel.nodes.add(i)); needDraw = true; refreshPanel(); } return; }
  if (ctrl) return;
  if (k === ' ') { spaceDown = true; cv.style.cursor = 'grab'; e.preventDefault(); return; }
  const toolKeys = {v: 'edit', p: 'pen', f: 'free', i: 'insert', d: 'erase', x: 'split', j: 'join', m: 'measure', g: 'fill'};
  if (toolKeys[k]) { setTool(toolKeys[k]); return; }
  if (k >= '1' && k <= '6') {
    const t = TYPES[+k - 1].id;
    if (sel.line) { pushUndo(); sel.line.type = t; refreshPanel(); toast(`Type: ${TYPE[t].name}`); }
    else { $('newType').value = t; toast(`New lines: ${TYPE[t].name}`); }
    return;
  }
  switch (k) {
    case 'Escape':
      if (tool === 'fill' && fillPts.length) { fillPts = []; } else if (drawing) finishDrawing(); else if (measure.length || joinFirst) { measure = []; joinFirst = null; } else { sel.line = null; sel.nodes.clear(); refreshPanel(); }
      needDraw = true; break;
    case 'Enter': if (drawing) finishDrawing(); else if (tool === 'fill') submitFill(); break;
    case 'Backspace': case 'Delete':
      e.preventDefault();
      if (tool === 'fill') { fillPts.pop(); needDraw = true; break; }
      if (drawing && tool === 'pen') { const n = drawing.line.nodes; if (n.length) { drawing.atStart ? n.shift() : n.pop(); needDraw = true; } if (!n.length) finishDrawing(); break; }
      if (e.shiftKey) deleteLine(sel.line); else if (sel.line && sel.nodes.size) deleteNodes(sel.line, sel.nodes);
      break;
    case 'R': reverseLine(sel.line); break;
    case 'c': case 'C': continueSelected(); break;
    case 'n': case 'N': $('vNodes').checked = !$('vNodes').checked; needDraw = true; break;
    case 'k': case 'K': toggleReviewed(); break;
    case 't': case 'T': open3dHere(); break;
    case 'h': case 'H': case '?': $('help').style.display = 'flex'; break;
    case 'a': setFollow(true); break;
    case 'A': setFollow(false); view.rot = 0; needDraw = true; break;
    case 'q': case 'Q': setFollow(false); view.rot -= (e.shiftKey ? 1 : 5) * Math.PI / 180; needDraw = true; break;
    case 'e': case 'E': setFollow(false); view.rot += (e.shiftKey ? 1 : 5) * Math.PI / 180; needDraw = true; break;
    case '+': case '=': zoomAt(W / 2, H / 2, 1.25); break;
    case '-': case '_': zoomAt(W / 2, H / 2, 0.8); break;
    case 'PageDown': case ']': case '}': e.preventDefault(); jump(e.shiftKey || k === '}' ? 100 : 20); break;
    case 'PageUp': case '[': case '{': e.preventDefault(); jump(e.shiftKey || k === '{' ? -100 : -20); break;
    case 'ArrowUp': case 'ArrowDown': case 'ArrowLeft': case 'ArrowRight': {
      e.preventDefault();
      const step = 120 / view.scale, c = Math.cos(view.rot), s = Math.sin(view.rot);
      const [dx, dy] = {ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0]}[k];
      view.cx += step * (c * dx + s * dy); view.cz -= step * (-s * dx + c * dy); if (followRoad) alignToRoad(); needDraw = true; break;
    }
  }
});
window.addEventListener('keyup', e => { if (e.key === ' ') { spaceDown = false; setTool(tool); } });

// ----------------------------------------------------------------------------- route navigation
function alignToRoad() {
  const ri = routeInfo(view.cx, view.cz);
  const phi = Math.atan2(-ri.fz, ri.fx);  // screen-space direction of the road, before rotation
  view.rot = -Math.PI / 2 - phi; needDraw = true;
}
function goToArc(a) {
  const r = routeAt(a); view.cx = r.x; view.cz = r.z;
  if (followRoad) alignToRoad();
  needDraw = true; updateStatus();
}
function jump(d) { const ri = routeInfo(view.cx, view.cz); goToArc(ri.arc + d); }
function toggleReviewed() {
  const seg = Math.floor(routeInfo(view.cx, view.cz).arc / 25);
  pushUndo(); reviewed.has(seg) ? reviewed.delete(seg) : reviewed.add(seg);
  toast(`${seg * 25}-${seg * 25 + 25} m ${reviewed.has(seg) ? 'marked reviewed' : 'not reviewed'}`); drawScrubBg();
}
$('scrub').addEventListener('input', e => goToArc(+e.target.value));
$('goto').addEventListener('keydown', e => { if (e.key === 'Enter') { goToArc(+e.target.value); e.target.blur(); } });

// coverage of left/right edges along the route, for the bottom bar
let coverage = {left: new Uint8Array(0), right: new Uint8Array(0)}, covTimer = null;
function scheduleCoverage() { clearTimeout(covTimer); covTimer = setTimeout(computeCoverage, 500); }
function computeCoverage() {
  if (!ROUTE.length) return;
  const n = Math.ceil(ROUTE[ROUTE.length - 1][2]) + 1;
  coverage = {left: new Uint8Array(n), right: new Uint8Array(n)};
  for (const ln of lines) {
    const key = ln.type === 'left_edge' ? 'left' : ln.type === 'right_edge' ? 'right' : null;
    if (!key || ln.nodes.length < 2) continue;
    const S = samples(ln);
    for (let i = 0; i < S.pts.length; i += 5) { const a = Math.round(routeInfo(S.pts[i][0], S.pts[i][1]).arc); if (a >= 0 && a < n) coverage[key][a] = 1; }
  }
  const L = coverage.left.reduce((s, v) => s + v, 0), R = coverage.right.reduce((s, v) => s + v, 0);
  $('sCover').innerHTML = `annotated: left <b>${L} m</b>, right <b>${R} m</b> of ${n - 1} m, reviewed <b>${reviewed.size * 25} m</b>`;
  drawScrubBg();
}
function drawScrubBg() {
  const c = $('scrubbg'), r = c.getBoundingClientRect();
  c.width = Math.max(10, r.width); c.height = 24; const g = c.getContext('2d');
  g.fillStyle = '#14171c'; g.fillRect(0, 0, c.width, c.height);
  if (!ROUTE.length) return;
  const n = ROUTE[ROUTE.length - 1][2], X = a => a / n * c.width;
  g.fillStyle = '#2b8a5c'; for (const s of reviewed) g.fillRect(X(s * 25), 0, Math.max(1, X(25)), 24);
  for (const [key, y, col] of [['left', 3, '#ff3b3b'], ['right', 17, '#2f86ff']]) {
    g.fillStyle = col; const a = coverage[key];
    for (let i = 0; i < a.length; i++) if (a[i]) { let j = i; while (j < a.length && a[j]) j++; g.fillRect(X(i), y, Math.max(1, X(j - i)), 4); i = j; }
  }
  g.fillStyle = '#ffffff55'; for (let a = 0; a < n; a += 100) g.fillRect(X(a), 9, 1, 6);
}

// ------------------------------------------------------------------------------------- status
function updateStatus() {
  if (!ROUTE.length) return;
  const rc = routeInfo(view.cx, view.cz);
  $('sArc').textContent = `${rc.arc.toFixed(1)} m`;
  if (document.activeElement !== $('scrub')) $('scrub').value = Math.round(rc.arc);
  if (TAB !== 'map') return;
  if (mouse.inside) { const ri = routeInfo(mouse.w[0], mouse.w[1]); $('sCur').textContent = `x ${mouse.w[0].toFixed(2)}  y ${mouse.w[1].toFixed(2)} | route ${ri.arc.toFixed(1)} m, ${ri.lat >= 0 ? 'right' : 'left'} ${Math.abs(ri.lat).toFixed(2)} m`; $('sGps').textContent = gpsText(mouse.w[0], mouse.w[1]); }
  $('sScale').textContent = `${(100 / view.scale).toFixed(2)} m / 100 px, tiles L${curLevel}`;
}
function updateSaveState() {
  $('savestate').innerHTML = dirty ? '<span style="color:#ffb020">unsaved changes</span>' : (lastSave ? `saved ${lastSave}` : 'no changes');
}
let toastTimer = null;
function toast(t) { const el = $('toast'); el.textContent = t; el.style.display = 'block'; clearTimeout(toastTimer); toastTimer = setTimeout(() => el.style.display = 'none', 2200); }

// ---------------------------------------------------------------------------------------- panel
function fillTypeSelect(el) { el.innerHTML = TYPES.map((t, i) => `<option value="${t.id}">${i + 1}. ${t.name}</option>`).join(''); }
['newType', 'pType', 'pOffType'].forEach(id => fillTypeSelect($(id)));
$('pOffType').value = 'lane_divider';
function refreshPanel() {
  const box = $('lines');
  box.innerHTML = '';
  lines.forEach(ln => {
    const t = TYPE[ln.type] || TYPE.other, S = samples(ln), d = document.createElement('div');
    d.className = 'li' + (ln === sel.line ? ' sel' : '');
    d.innerHTML = `<span class="sw" style="background:${t.color}"></span><span class="nm" title="${ln.name}">${ln.name}</span><span class="meta">${ln.nodes.length} n, ${S.len.toFixed(0)} m</span>` +
      `<button class="ic ${ln.visible === false ? 'off' : ''}" data-a="vis" title="Show / hide">&#128065;</button><button class="ic ${ln.locked ? '' : 'off'}" data-a="lock" title="Lock (no editing)">&#128274;</button>`;
    d.onclick = ev => {
      const a = ev.target.dataset.a;
      if (a === 'vis') { ln.visible = ln.visible === false; markDirty(); refreshPanel(); return; }
      if (a === 'lock') { ln.locked = !ln.locked; markDirty(); refreshPanel(); return; }
      selectLine(ln); if (!lineInView(ln)) focusLine(ln);
    };
    d.ondblclick = () => focusLine(ln);
    box.appendChild(d);
  });
  if (!lines.length) box.innerHTML = '<div class="muted">No lines yet.</div>';
  const ln = sel.line;
  $('props').style.display = ln ? '' : 'none'; $('noSel').style.display = ln ? 'none' : '';
  if (ln) {
    $('pName').value = ln.name || ''; $('pType').value = ln.type; $('pSmooth').value = ln.smooth ? '1' : '0';
    const S = samples(ln), a0 = routeInfo(...ln.nodes[0]).arc, a1 = routeInfo(...ln.nodes[ln.nodes.length - 1]).arc;
    $('pInfo').textContent = `${ln.nodes.length} keynodes, ${S.len.toFixed(1)} m long, route ${a0.toFixed(0)} -> ${a1.toFixed(0)} m` + (sel.nodes.size ? ` | ${sel.nodes.size} selected` : '');
  }
  needDraw = true;
}
function lineInView(ln) { const vb = viewBBox(0), b = samples(ln).bbox; return !(b[2] < vb[0] || b[0] > vb[2] || b[3] < vb[1] || b[1] > vb[3]); }
function focusLine(ln) {
  const b = samples(ln).bbox; view.cx = (b[0] + b[2]) / 2; view.cz = (b[1] + b[3]) / 2;
  view.scale = Math.max(0.05, Math.min(60, 0.8 * Math.min(W / Math.max(b[2] - b[0], 1), H / Math.max(b[3] - b[1], 1))));
  needDraw = true;
}
$('pName').addEventListener('change', e => { if (sel.line) { pushUndo(); sel.line.name = e.target.value; refreshPanel(); } });
$('pType').addEventListener('change', e => { if (sel.line) { pushUndo(); sel.line.type = e.target.value; refreshPanel(); } });
$('pSmooth').addEventListener('change', e => { if (sel.line) { pushUndo(); sel.line.smooth = e.target.value === '1'; refreshPanel(); } });
$('pFocus').onclick = () => sel.line && focusLine(sel.line);
$('pReverse').onclick = () => reverseLine(sel.line);
$('pContinue').onclick = continueSelected;
$('pSimplify').onclick = () => simplifyLine(sel.line, +$('pSimpTol').value);
$('pDensify').onclick = () => densifyLine(sel.line, Math.max(0.05, +$('pDensStep').value), $('pDensMode').value, +$('pDensAround').value);
$('pOffset').onclick = () => offsetCopy(sel.line, +$('pOffDist').value, $('pOffType').value);
$('pDelNodes').onclick = () => deleteNodes(sel.line, sel.nodes);
$('pDelete').onclick = () => { if (sel.line && confirm(`Delete "${sel.line.name}"?`)) deleteLine(sel.line); };
$('freeTol').oninput = e => $('freeTolV').textContent = `${(+e.target.value).toFixed(2)} m`;
['vBright', 'vContrast', 'vSat', 'vOpacity', 'vWidth', 'vRoute', 'vNodes', 'vArrows', 'vLabels'].forEach(id => $(id).addEventListener('input', () => { needDraw = true; saveSettings(); }));
$('vReset').onclick = () => { localStorage.removeItem(VIEWKEY()); location.reload(); };
document.querySelectorAll('button.tool').forEach(b => b.onclick = () => setTool(b.dataset.tool));
$('bUndo').onclick = () => TAB === 'map' ? undo() : M3.undo(); $('bRedo').onclick = () => TAB === 'map' ? redo() : M3.redo();
let followRoad = false;
const VIEWKEY = () => 'meshsfm_map_' + ((META && META.project && META.project.root) || '');
function setFollow(v) {
  followRoad = v; $('bAlign').classList.toggle('toggled', v);
  $('bAlign').innerHTML = v ? 'Following road <kbd>A</kbd>' : 'Follow road <kbd>A</kbd>';
  if (v) alignToRoad(); needDraw = true;
}
$('bAlign').onclick = () => setFollow(!followRoad);
$('bNorth').onclick = () => { setFollow(false); view.rot = 0; needDraw = true; };
$('bReview').onclick = toggleReviewed;
$('bSave').onclick = save;
$('bHelp').onclick = () => $('help').style.display = 'flex';
$('help').onclick = e => { if (e.target.id === 'help') $('help').style.display = 'none'; };
$('bImport').onclick = importAuto; $('bnImport').onclick = () => { $('banner').style.display = 'none'; importAuto(); };
$('bnClose').onclick = () => $('banner').style.display = 'none';
$('bDownload').onclick = () => {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(payload(), null, 1)], {type: 'application/json'}));
  a.download = 'annotation.json'; a.click();
};
$('bUpload').onclick = () => $('fileIn').click();
$('fileIn').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { const o = JSON.parse(await f.text()); pushUndo(); loadAnnotation(o); toast(`Loaded ${lines.length} lines from ${f.name}`); }
  catch (err) { alert('Could not read that file: ' + err); }
  e.target.value = '';
};
function saveSettings() {
  const o = {}; ['vBright', 'vContrast', 'vSat', 'vOpacity', 'vWidth'].forEach(id => o[id] = $(id).value);
  ['vRoute', 'vNodes', 'vArrows', 'vLabels', 'snapEnds', 'autosave'].forEach(id => o[id] = $(id).checked);
  o.followRoad = followRoad;
  o.view = view; localStorage.setItem(VIEWKEY(), JSON.stringify(o));
}
function loadSettings() {
  try {
    const o = JSON.parse(localStorage.getItem(VIEWKEY()) || 'null'); if (!o) return false;
    setTimeout(() => setFollow(o.followRoad === true), 0);
    for (const [k, v] of Object.entries(o)) { if (k === 'view' || k === 'followRoad') continue; const el = $(k); if (!el) continue; if (el.type === 'checkbox') el.checked = v; else el.value = v; }
    if (o.view) view = o.view; return !!o.view;
  } catch (e) { return false; }
}

// -------------------------------------------------------------------------------------- server
function payload() { return {version: 1, frame: 'splat (y-down), x/z keynodes in metres', lines: lines.map(cleanLine), reviewed: [...reviewed].sort((a, b) => a - b), nextId}; }
function loadAnnotation(o) {
  lines = (o.lines || []).map(l => ({...l, nodes: l.nodes.map(p => [+p[0], +p[1]]), smooth: l.smooth !== false}));
  reviewed = new Set(o.reviewed || []);
  nextId = Math.max(o.nextId || 1, 1 + Math.max(0, ...lines.map(l => l.id || 0)));
  lines.forEach(l => { if (!l.id) l.id = nextId++; });
  sel.line = null; sel.nodes.clear(); refreshPanel(); scheduleCoverage();
}
let saving = false;
async function save() {
  if (saving) return; saving = true; $('savestate').textContent = 'saving + exporting...';
  try {
    const r = await fetch('/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload())});
    const o = await r.json();
    dirty = false; lastSave = o.saved.split(' ')[1];
    $('exportInfo').innerHTML = o.export_error ? `<span style="color:#ff5d5d">export failed: ${o.export_error}</span>` :
      `Exported ${o.lines} lines. Both edges along ${o.both_edges_m} m (left ${o.left_m} m, right ${o.right_m} m).`;
    toast('Saved and exported'); if (V3.started) V3.refreshLines();
  } catch (err) { toast('Save failed: ' + err); }
  saving = false; updateSaveState();
}
setInterval(() => { if (dirty && $('autosave').checked && !drag && !drawing) save(); }, 60000);
window.addEventListener('beforeunload', e => { saveSettings(); if (dirty) { e.preventDefault(); e.returnValue = ''; } });
async function importAuto() {
  if (lines.some(l => l.type === 'left_edge' || l.type === 'right_edge') && !confirm('There are already edge lines. Import the auto-detected edges as extra lines anyway?')) return;
  const tol = +(prompt('Keynode tolerance in metres (smaller = more keynodes, closer to the detection):', '0.04') || 0.04);
  $('banner').style.display = 'none';
  const r = await fetch(`/auto?tol=${tol}`), arr = await r.json();
  pushUndo();
  for (const a of arr) { const ln = newLine(a.type, a.nodes); ln.name = a.name; ln.smooth = true; }
  refreshPanel(); toast(`Imported ${arr.length} auto-detected lines (${arr.map(a => a.nodes.length).join(' + ')} keynodes)`);
}

// ---------------------------------------------------------------------------------------- start
function resize() {
  if (TAB !== 'map') return;
  const r = $('mapWrap').getBoundingClientRect(); DPR = window.devicePixelRatio || 1;
  W = r.width; H = r.height; cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  needDraw = true; drawScrubBg();
}
window.addEventListener('resize', resize);
function loop() { if (needDraw) draw(); requestAnimationFrame(loop); }
(async () => {
  resize();
  await READY;
  ROUTE = META.route;
  tileSet = Object.fromEntries(Object.entries(META.tiles).map(([L, v]) => [L, new Set(v)]));
  $('scrub').max = Math.ceil(ROUTE[ROUTE.length - 1][2]);
  const hadView = loadSettings();
  if (!hadView) {          // north up, fitted to the whole drive path
    const xs = ROUTE.map(r => r[0]), ys = ROUTE.map(r => r[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    view.cx = (x0 + x1) / 2; view.cz = (y0 + y1) / 2; view.rot = 0;
    view.scale = Math.max(2, Math.min(60, 0.8 * Math.min(W / Math.max(x1 - x0, 10), H / Math.max(y1 - y0, 10))));
  }
  loadAnnotation(await (await fetch('/annotation')).json());
  if (!lines.length && META.auto_available) $('banner').style.display = 'block';
  dirty = false; updateSaveState(); setTool('edit'); updateStatus(); computeCoverage();
  setInterval(saveSettings, 5000);
  loop();
})();

// ----------------------------------------------------------------------------- studio integration
function submitFill() {
  if (fillPts.length < 3) { toast('Click at least 3 points around the area'); return; }
  M3.requestFill(fillPts.map(p => p.slice())); fillPts = []; needDraw = true;
}
function open3dHere() {
  const ri = routeInfo(view.cx, view.cz);
  setTab('3d'); M3.placeAt(view.cx, view.cz, Math.atan2(ri.fz, ri.fx));
}
$('b3dHere').onclick = open3dHere;
gotoHooks.push((x, y) => { if (TAB === 'map') { view.cx = x; view.cz = y; needDraw = true; updateStatus(); } });
tabHooks.push(t => { if (t === 'map') { resize(); needDraw = true; } });
function showOnMap(x, y) { setTab('map'); view.cx = x; view.cz = y; view.scale = Math.max(view.scale, 15); needDraw = true; updateStatus(); }

$('brushR').addEventListener('input', () => { $('brushRV').textContent = (+$('brushR').value).toFixed(2) + ' m'; });
$('smS').addEventListener('input', () => { $('smSV').textContent = (+$('smS').value).toFixed(2); });
