'use strict';
// Lane lines in the 3D view: drawn live on the road surface (over the canopy, so they can be edited under trees),
// edited with the "Edit lanes" tool. Uses the map's line data and undo history (map.js globals).
const Lanes3D = (() => {
  const cache = new Map();   // line id -> {key, pts [[x,y,z] rel O], seg [], nodes [[x,y,z]]}
  let fetching = false, drag = null;
  const keyOf = ln => JSON.stringify(ln.nodes) + (ln.smooth ? 1 : 0);
  const MAXD = 250;

  async function refresh() {
    if (fetching || !V3.started) return;
    const todo = lines.filter(ln => ln.visible !== false && ln.nodes.length >= 2 && (!cache.has(ln.id) || cache.get(ln.id).key !== keyOf(ln)));
    for (const id of [...cache.keys()]) if (!lines.some(l => l.id === id)) cache.delete(id);
    if (!todo.length) return;
    fetching = true;
    try {
      const pay = todo.map(ln => {
        const S = samples(ln), pts = [], seg = [];
        for (let i = 0; i < S.pts.length; i += 5) { pts.push(S.pts[i]); seg.push(S.seg[i]); }
        if ((S.pts.length - 1) % 5) { pts.push(S.pts[S.pts.length - 1]); seg.push(S.seg[S.pts.length - 1]); }
        return {ln, key: keyOf(ln), pts, seg, nodes: ln.nodes.map(p => p.slice())};
      });
      const r = await api('/heights', {lines: pay.map(p => ({pts: p.pts, nodes: p.nodes}))});
      const O = V3.O;
      pay.forEach((p, k) => {
        const h = r.heights[k];
        cache.set(p.ln.id, {key: p.key, seg: p.seg, pts: p.pts.map((q, i) => [q[0] - O[0], q[1] - O[1], h.pts[i] - O[2]]),
                            nodes: p.nodes.map((q, i) => [q[0] - O[0], q[1] - O[1], h.nodes[i] - O[2]])});
      });
      V3.redraw();
    } catch (e) { /* retry on the next tick */ }
    fetching = false;
  }
  setInterval(() => { if (TAB === '3d') refresh(); }, 200);

  const editing = () => M3.tool === 'lanes';
  // current keynode positions (xy from the live data, height from the cache; the dragged one from the surface hit)
  function nodePos(ln, i) {
    const O = V3.O, c = cache.get(ln.id), n = ln.nodes[i];
    let z = c && c.nodes[Math.min(i, c.nodes.length - 1)] ? c.nodes[Math.min(i, c.nodes.length - 1)][2] : (c ? c.pts[0][2] : 0);
    if (drag && drag.line === ln && drag.z !== undefined && sel.nodes.has(i)) z = drag.z;
    return [n[0] - O[0], n[1] - O[1], z];
  }
  V3.hudHooks.push(hx => {
    if (TAB !== '3d' || !$('showLanes').checked) return;
    const M = V3.mvp(), cam = V3.cam.p, lw = +$('vWidth').value;
    const order = lines.filter(l => l !== sel.line).concat(sel.line ? [sel.line] : []);
    for (const ln of order) {
      if (ln.visible === false) continue;
      const c = cache.get(ln.id); if (!c) continue;
      const t = TYPE[ln.type] || TYPE.other, isSel = ln === sel.line;
      const segs = []; let cur = [];
      for (const p of c.pts) {
        const d = Math.hypot(p[0] - cam[0], p[1] - cam[1]);
        const s = d < MAXD ? V3.project([p[0], p[1], p[2] + 0.04], M) : null;
        if (!s || s[2] < 0.2) { if (cur.length > 1) segs.push(cur); cur = []; continue; }
        cur.push(s);
      }
      if (cur.length > 1) segs.push(cur);
      const path = () => { hx.beginPath(); for (const sg of segs) sg.forEach((s, i) => i ? hx.lineTo(s[0], s[1]) : hx.moveTo(s[0], s[1])); };
      hx.lineJoin = 'round'; hx.lineCap = 'round';
      path(); hx.strokeStyle = '#000b'; hx.lineWidth = lw + (isSel ? 4 : 2.5); hx.stroke();
      path(); hx.setLineDash(t.dash); hx.strokeStyle = t.color; hx.lineWidth = lw + (isSel ? 1 : 0); hx.stroke(); hx.setLineDash([]);
      if (editing() && (isSel || $('vNodes').checked)) {
        ln.nodes.forEach((_, i) => {
          const p = nodePos(ln, i); if (Math.hypot(p[0] - cam[0], p[1] - cam[1]) > 90) return;
          const s = V3.project([p[0], p[1], p[2] + 0.04], M); if (!s || s[2] < 0.2) return;
          const on = isSel && sel.nodes.has(i), end = i === 0 || i === ln.nodes.length - 1;
          hx.fillStyle = on ? '#fff' : end ? t.color : '#11151b'; hx.strokeStyle = on ? '#4da3ff' : t.color; hx.lineWidth = on ? 2.5 : 1.6;
          hx.beginPath(); hx.arc(s[0], s[1], isSel ? 5.5 : 4, 0, 7); hx.fill(); hx.stroke();
        });
      }
    }
  });

  // ------------------------------------------------------------------------------- hit tests
  function hitNode(x, y) {
    const M = V3.mvp(), cam = V3.cam.p; let best = null, bd = 11;
    const cand = sel.line ? [sel.line, ...lines.filter(l => l !== sel.line)] : lines;
    for (const ln of cand) {
      if (ln.visible === false || ln.locked || (ln !== sel.line && !$('vNodes').checked)) continue;
      for (let i = 0; i < ln.nodes.length; i++) {
        const p = nodePos(ln, i); if (Math.hypot(p[0] - cam[0], p[1] - cam[1]) > 90) continue;
        const s = V3.project([p[0], p[1], p[2] + 0.04], M); if (!s || s[2] < 0.2) continue;
        const d = Math.hypot(s[0] - x, s[1] - y); if (d < bd) { bd = d; best = {line: ln, idx: i}; }
      }
      if (best && best.line === sel.line) return best;
    }
    return best;
  }
  function hitLine(x, y) {
    const M = V3.mvp(), cam = V3.cam.p, O = V3.O; let best = null, bd = 8;
    for (const ln of lines) {
      if (ln.visible === false || ln.locked) continue;
      const c = cache.get(ln.id); if (!c) continue;
      let prev = null;
      for (let i = 0; i < c.pts.length; i++) {
        const p = c.pts[i]; const s = Math.hypot(p[0] - cam[0], p[1] - cam[1]) < MAXD ? V3.project([p[0], p[1], p[2] + 0.04], M) : null;
        if (s && s[2] > 0.2 && prev) {
          const [a, ia] = prev, vx = s[0] - a[0], vy = s[1] - a[1], L2 = vx * vx + vy * vy || 1e-9;
          const t = Math.max(0, Math.min(1, ((x - a[0]) * vx + (y - a[1]) * vy) / L2)), d = Math.hypot(a[0] + t * vx - x, a[1] + t * vy - y);
          if (d < bd) { const q = c.pts[ia]; bd = d; best = {line: ln, seg: c.seg[i], p: [q[0] + t * (p[0] - q[0]) + O[0], q[1] + t * (p[1] - q[1]) + O[1]]}; }
        }
        prev = s && s[2] > 0.2 ? [s, i] : null;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------------------------- editing
  function down(e, x, y) {
    const hn = hitNode(x, y);
    if (hn) {
      if (e.shiftKey) { if (sel.line !== hn.line) selectLine(hn.line); sel.nodes.has(hn.idx) ? sel.nodes.delete(hn.idx) : sel.nodes.add(hn.idx); refreshPanel(); V3.redraw(); return; }
      if (sel.line !== hn.line || !sel.nodes.has(hn.idx)) { selectLine(hn.line); sel.nodes.add(hn.idx); }
      drag = {line: hn.line, grab: hn.idx, orig: [...sel.nodes].map(i => [i, hn.line.nodes[i].slice()]), undone: false};
      refreshPanel(); V3.redraw(); return;
    }
    const hl = hitLine(x, y);
    if (hl && e.altKey) { insertNode(hl); V3.redraw(); return; }
    if (hl) { selectLine(hl.line); V3.redraw(); return; }
    const h = V3.pickAt(x, y);
    if (e.shiftKey && h) {
      const O = V3.O, p = [h.p[0] + O[0], h.p[1] + O[1]];
      pushUndo();
      if (sel.line && !sel.line.locked) {
        const ln = sel.line, a = ln.nodes[0], b = ln.nodes[ln.nodes.length - 1];
        if (dist(p, a) < dist(p, b) && ln.nodes.length > 1) { ln.nodes.unshift(p); sel.nodes.clear(); sel.nodes.add(0); }
        else { ln.nodes.push(p); sel.nodes.clear(); sel.nodes.add(ln.nodes.length - 1); }
      } else { const ln = newLine(null, [p]); selectLine(ln); sel.nodes.add(0); toast(`New ${TYPE[ln.type].name}: Shift+click to add keynodes`); }
      refreshPanel(); markDirty(); V3.redraw(); return;
    }
    sel.line = null; sel.nodes.clear(); refreshPanel(); V3.redraw();
  }
  function move(x, y) {
    if (!drag) return false;
    const h = V3.pickAt(x, y); if (!h) return true;
    if (!drag.undone) { pushUndo(); drag.undone = true; }
    const O = V3.O, g = drag.orig.find(o => o[0] === drag.grab)[1];
    const dx = h.p[0] + O[0] - g[0], dy = h.p[1] + O[1] - g[1];
    for (const [i, o] of drag.orig) drag.line.nodes[i] = [o[0] + dx, o[1] + dy];
    drag.z = h.p[2];
    markDirty(); V3.redraw(); return true;
  }
  function up() { if (!drag) return false; drag = null; refreshPanel(); V3.redraw(); return true; }
  function dbl(x, y) { if (hitNode(x, y)) return; const hl = hitLine(x, y); if (hl) { insertNode(hl); V3.redraw(); } }
  function key(e) {
    const k = e.key;
    if (k === 'Delete' || k === 'Backspace') { e.preventDefault(); if (e.shiftKey) deleteLine(sel.line); else if (sel.line && sel.nodes.size) deleteNodes(sel.line, sel.nodes); V3.redraw(); return true; }
    if (k === 'Escape' && (sel.line || sel.nodes.size)) { sel.line = null; sel.nodes.clear(); refreshPanel(); V3.redraw(); return true; }
    if (k >= '1' && k <= '6') { const t = TYPES[+k - 1].id; if (sel.line) { pushUndo(); sel.line.type = t; refreshPanel(); } else $('newType').value = t; toast(`Type: ${TYPE[t].name}`); V3.redraw(); return true; }
    if ((e.ctrlKey || e.metaKey) && (k === 'a' || k === 'A') && sel.line) { e.preventDefault(); sel.line.nodes.forEach((_, i) => sel.nodes.add(i)); V3.redraw(); return true; }
    return false;
  }
  function nodeScreen(ln, i) { const p = nodePos(ln, i); return V3.project([p[0], p[1], p[2] + 0.04]); }
  return {down, move, up, dbl, key, refresh, nodeScreen, get dragging() { return !!drag; }};
})();
