'use strict';
// Sections (Generate / Edit / Align / Annotate / Drive), project + layers, file browser, jobs, alignment picks.
const STUDIO = (() => {
  let PROJ = null, SEC = 'gen', browseCb = null, picks = [], pickSide = 'moving', lastAlign = null;

  // ------------------------------------------------------------------------------------- sections
  function setSection(s) {
    SEC = s; document.body.dataset.sec = s;
    document.querySelectorAll('.secbtn').forEach(b => b.classList.toggle('on', b.dataset.sec === s));
    const hasMesh = META && META.mesh;
    if (s === 'annotate' || s === 'gen') setTab('map');
    else if (['edit', 'align', 'drive'].includes(s) && hasMesh) setTab('3d');
    if (s === 'drive' && hasMesh && V3.started && !V3.drive.on) $('bDrive').click();
    if (s !== 'drive' && V3.drive && V3.drive.on) $('bDrive').click();
    if (s === 'align') V3.setTool('inspect');
    updateSplit();
  }
  function layerKind(id) { const L = PROJ && PROJ.layers.find(l => l.id === id); return L ? L.kind : null; }
  function updateSplit() {           // Align side by side: moving | fixed, one kind per half (splat vs mesh)
    if (typeof V3 === 'undefined' || !V3.setSplit) return;
    const km = layerKind($('alMoving').value), kf = layerKind($('alFixed').value);
    const ok = SEC === 'align' && $('alSplit').checked && km && kf && km !== kf && [km, kf].every(k => k === 'mesh' || k === 'splat');
    V3.setSplit(ok ? {left: km, right: kf} : null);
    document.body.classList.toggle('split3', !!ok);
    if (ok) {
      window.SHOW_MESH = true; if (typeof SPLAT !== 'undefined') SPLAT.enabled = true;
      const sid = km === 'splat' ? $('alMoving').value : $('alFixed').value, L = PROJ.layers.find(l => l.id === sid);
      if (L && !L.visible) toast(`Switch on the layer "${L.name}" to see it in the split view`);
    }
    $('alSplitNote').textContent = SEC === 'align' && $('alSplit').checked && !ok ? 'side by side works for a splat and a mesh; these two are shown on top of each other' : '';
  }
  document.querySelectorAll('.secbtn').forEach(b => b.onclick = () => setSection(b.dataset.sec));

  // ------------------------------------------------------------------------------------- project / layers
  async function refresh() {
    PROJ = await api('/project');
    $('projName').textContent = PROJ.name; $('projRoot').textContent = '…/' + PROJ.root.split('/').filter(Boolean).slice(-2).join('/'); $('projRoot').title = PROJ.root;
    $('projPath').textContent = PROJ.path ? `drive path: ${PROJ.path.source}, ${PROJ.path.length.toFixed(0)} m` : 'no drive path yet (made from poses, a drawn line, or the data axis)';
    $('projPrepared').textContent = PROJ.prepared ? `prepared ${PROJ.prepared}` : 'not prepared yet';
    const box = $('layerList'); box.innerHTML = '';
    const icon = {mesh: '&#9650;', splat: '&#10022;', points: '&#8942;'};
    // mesh visibility (all mesh layers are one streamed mesh in the viewer)
    const mrow = document.createElement('div'); mrow.className = 'lrow';
    mrow.innerHTML = `<input type="checkbox" ${window.SHOW_MESH !== false ? 'checked' : ''}><b>Mesh</b><span class="muted">show / hide in 3D (U)</span>`;
    mrow.querySelector('input').onchange = e => { window.SHOW_MESH = e.target.checked; V3.redraw(); };
    box.appendChild(mrow);
    for (const L of PROJ.layers) {
      const r = document.createElement('div'); r.className = 'lrow';
      r.innerHTML = `<input type="checkbox" ${L.visible ? 'checked' : ''} title="visible"><span class="lk">${icon[L.kind]}</span>
        <span class="ln" title="${L.src}">${L.name}</span><span class="muted">${L.id}</span>
        <select title="axis convention of the file">${['z_up', 'y_up', 'y_down'].map(f => `<option ${L.frame === f ? 'selected' : ''}>${f}</option>`).join('')}</select>
        <button title="remove layer (files are not touched)">&times;</button>`;
      const [cb, , , , sel, rm] = r.children;
      cb.onchange = async () => { await api('/project/layer/update', {id: L.id, visible: cb.checked}); await refresh(); };
      sel.onchange = async () => { await api('/project/layer/update', {id: L.id, frame: sel.value}); toast('Axis changed - press Prepare'); refresh(); };
      rm.onclick = async () => { if (confirm(`Remove layer ${L.name}? (the file stays on disk)`)) { await api('/project/layer/remove', {id: L.id}); refresh(); } };
      box.appendChild(r);
    }
    for (const id of ['alMoving', 'alFixed']) {
      const s = $(id), v = s.value; s.innerHTML = PROJ.layers.map(L => `<option value="${L.id}">${L.id}: ${L.name}</option>`).join('');
      if (v) s.value = v;
    }
    if (!$('alFixed').value && PROJ.layers.length > 1) $('alFixed').selectedIndex = 1;
    updateSplit();
    const gens = (PROJ.generated || []); $('trSrc').innerHTML = gens.map(g => `<option value="${g}">${g}</option>`).join('') || '<option value="">(generate a mesh first)</option>';
    if (typeof SPLAT !== 'undefined' && V3.started) SPLAT.setLayers(PROJ);
    const S = PROJ.settings; ['chunk', 'ground_cell', 'corridor_half'].forEach(k => { if ($('set_' + k)) $('set_' + k).value = S[k]; });
    return PROJ;
  }

  $('bAddLayer').onclick = async () => {
    const src = $('addPath').value.trim(); if (!src) { toast('Choose a file or folder'); return; }
    const r = await api('/project/layer/add', {kind: $('addKind').value, src, frame: $('addFrame').value, name: $('addName').value || undefined});
    if (!r.ok) { toast(r.error || 'could not add'); return; }
    toast(`Added ${r.layer.name} - press Prepare to load it`); $('addPath').value = ''; $('addName').value = ''; refresh();
  };
  $('bPrepare').onclick = () => job('/project/prepare', {reset_path: $('resetPath').checked}, 'Preparing', () => location.reload());
  $('bSettings').onclick = async () => { await api('/project/settings', {chunk: +$('set_chunk').value, ground_cell: +$('set_ground_cell').value, corridor_half: +$('set_corridor_half').value}); toast('Saved - press Prepare to rebuild'); };

  // ------------------------------------------------------------------------------------- generate
  $('genPipe').onchange = () => { $('genHelp').innerHTML = GENHELP[$('genPipe').value]; };
  const GENHELP = {
    lidar: 'LiDAR + RGB (+ GPS/INS poses). Capture folder with <code>cameras.json</code>, <code>images/</code>, <code>lidar/</code>, optional <code>masks/</code>. Without poses KISS-ICP odometry is run.',
    rgb: 'RGB (+ GPS poses): COLMAP features (masks honoured), triangulation with the known poses (or full SfM without), dense MVS. Needs COLMAP with CUDA.',
    points: 'A coloured point cloud (.ply / .pcd, e.g. COLMAP fused.ply or a LiDAR map) - or, with separate inputs, a COLMAP model + images (its poses become the drive path).',
  };
  $('genHelp').innerHTML = GENHELP.lidar;
  $('genFrom').onchange = () => { const c = $('genFrom').value === 'capture'; $('genCapture').style.display = c ? '' : 'none'; $('genInputs').style.display = c ? 'none' : ''; };
  document.querySelectorAll('.bIn').forEach(b => b.onclick = () => browse($(b.dataset.for).value, p => { $(b.dataset.for).value = p; }));
  $('bGen').onclick = () => {
    if ($('genFrom').value === 'inputs') {
      const inputs = {}; ['images', 'masks', 'lidar', 'colmap', 'points', 'poses', 'timestamps', 'lidar_poses', 'lidar_timestamps', 'gps', 'intrinsics', 'T_cam_lidar'].forEach(k => { const v = $('in_' + k).value.trim(); if (v) inputs[k] = v; });
      if (!inputs.images && !inputs.colmap) { toast('Images are required'); return; }
      if ($('genPipe').value === 'lidar' && !inputs.lidar) { toast('The LiDAR pipeline needs the LiDAR scans folder'); return; }
      job('/gen/run', {pipeline: $('genPipe').value, inputs, name: $('genName').value || $('genPipe').value, masks: $('genMasks').checked,
                       objects: $('genObjects').checked, fast: $('genFast').checked, ref_step: $('genFast').checked ? 2 : 1}, 'Generating', () => location.reload());
      return;
    }
    const capture = $('genPath').value.trim(); if (!capture) { toast('Choose the capture folder / point cloud'); return; }
    job('/gen/run', {pipeline: $('genPipe').value, capture, name: $('genName').value || $('genPipe').value, masks: $('genMasks').checked,
                     objects: $('genObjects').checked, fast: $('genFast').checked, ref_step: $('genFast').checked ? 2 : 1}, 'Generating', () => location.reload());
  };

  $('pAsPath').onclick = async () => {
    if (typeof sel === 'undefined' || !sel.line) { toast('Select a line first'); return; }
    if (!confirm('Use the selected line as the drive path? The mesh is re-chunked along it (mesh edits are kept in a backup folder).')) return;
    $('bSave').click(); await new Promise(r => setTimeout(r, 1500));
    job('/path/from_lines', {id: sel.line.id}, 'Re-chunking along the new path', () => location.reload());
  };
  $('bExpColmap').onclick = () => {
    if (!$('trSrc').value) { toast('Generate a mesh from a capture first'); return; }
    job('/splat/export_colmap', {name: $('trSrc').value}, 'Exporting', res => { if (res && res.path) { $('expColmapInfo').innerHTML = `Exported: <code title="${res.path}">…/${res.path.split('/').filter(Boolean).slice(-2).join('/')}</code> - train it, then add the .ply as a splat layer (z up).`; } });
  };
  // collapsible side panels (click a panel title); remembered per browser
  let COLL = {}; try { COLL = JSON.parse(localStorage.getItem('meshsfm_collapsed') || '{}'); } catch (e) {}
  document.querySelectorAll('#side .sec > h3:first-child').forEach(h => {
    const key = h.textContent.trim().slice(0, 24); const sec = h.parentElement;
    if (COLL[key]) sec.classList.add('collapsed');
    h.addEventListener('click', e => { if (e.target.tagName === 'BUTTON') return; sec.classList.toggle('collapsed'); COLL[key] = sec.classList.contains('collapsed'); try { localStorage.setItem('meshsfm_collapsed', JSON.stringify(COLL)); } catch (err) {} });
  });
  // drive path drawn by the user (a line of type drive_path)
  $('dpMap').onclick = () => {
    setSection('annotate'); setTab('map'); if (typeof sel !== 'undefined') { sel.line = null; sel.nodes.clear(); }
    $('newType').value = 'drive_path'; setTool('pen'); toast('Click along the way you want to drive; Enter to finish, then "Use the drawn path"');
  };
  $('dpView').onclick = () => {
    if (!(META && META.mesh)) { toast('No mesh yet: draw it on the map'); return; }
    setTab('3d'); if (typeof sel !== 'undefined') { sel.line = null; sel.nodes.clear(); }
    $('newType').value = 'drive_path'; const b = document.querySelector('.tool3[data-tool="lanes"]'); if (b) b.click();
    toast('Shift+click points on the surface along the way you want to drive; then "Use the drawn path"');
  };
  $('dpUse').onclick = async () => {
    const cand = (typeof lines !== 'undefined' ? lines : []).filter(l => l.type === 'drive_path' && l.nodes.length >= 2);
    if (!cand.length) { toast('Draw a Drive path line first'); return; }
    const ln = cand[cand.length - 1];
    if (!confirm(`Use the drawn path (${ln.nodes.length} points) as the drive path? The mesh is re-chunked along it; earlier mesh edits are kept in a backup folder.`)) return;
    $('bSave').click(); await new Promise(r => setTimeout(r, 1500));
    job('/path/from_lines', {id: ln.id}, 'Re-chunking along the drawn path', () => location.reload());
  };
  // ------------------------------------------------------------------------------------- file browser
  async function browse(start, cb) {
    browseCb = cb; $('fsWrap').style.display = 'flex'; await ls(start || (PROJ && PROJ.root) || '~');
  }
  async function ls(p) {
    const r = await api('/fs/ls?path=' + encodeURIComponent(p));
    $('fsPath').value = r.path; const box = $('fsList'); box.innerHTML = '';
    const up = document.createElement('div'); up.className = 'fsrow'; up.textContent = '.. (up)'; up.onclick = () => ls(r.parent); box.appendChild(up);
    for (const e of r.entries) {
      const d = document.createElement('div'); d.className = 'fsrow' + (e.dir ? ' dir' : '');
      d.textContent = (e.dir ? '\u{1F4C1} ' : '') + e.name + (e.dir ? '' : `  (${(e.size / 1e6).toFixed(1)} MB)`);
      d.onclick = () => e.dir ? ls(r.path + '/' + e.name) : pick(r.path + '/' + e.name);
      box.appendChild(d);
    }
  }
  function pick(p) { $('fsWrap').style.display = 'none'; if (browseCb) browseCb(p); }
  $('fsUse').onclick = () => pick($('fsPath').value);
  $('fsClose').onclick = () => { $('fsWrap').style.display = 'none'; };
  $('fsPath').onkeydown = e => { if (e.key === 'Enter') ls($('fsPath').value); };
  $('bBrowseAdd').onclick = () => browse($('addPath').value, p => { $('addPath').value = p; });
  $('bBrowseGen').onclick = () => browse($('genPath').value, p => { $('genPath').value = p; });

  // ------------------------------------------------------------------------------------- jobs
  async function job(path, body, label, done) {
    const r = await api(path, body);
    if (!r.ok) { toast(r.error || 'failed'); return; }
    $('jobBar').style.display = 'block'; $('jobMsg').textContent = label + '...';
    const t = setInterval(async () => {
      const s = await api('/job');
      $('jobMsg').textContent = s.busy ? `${s.msg} ${s.progress ? Math.round(100 * s.progress) + '%' : ''}` : (s.error ? 'Error: ' + s.error : 'Done');
      if (!s.busy) {
        clearInterval(t); setTimeout(() => { $('jobBar').style.display = 'none'; }, s.error ? 8000 : 1500);
        if (!s.error && done) done(s.result);
      }
    }, 800);
  }

  // ------------------------------------------------------------------------------------- align
  const fmt = v => v.map(x => x.toFixed(2)).join(', ');
  let pairRes = [];                                 // residual per pair after the last solve
  function drawPicks() {
    $('alPicks').innerHTML = picks.map((p, i) => {
      const r = pairRes[i], bad = r && r.suspect;
      const badge = r ? `<span class="pres ${bad ? 'bad' : ''}" title="${bad ? 'far off after the solve: probably a mis-click, it was down-weighted' : 'distance between the two points after the solve'}">${(100 * r.residual).toFixed(0)} cm${bad ? ' - mis-click?' : ''}</span>` : '';
      return `<div class="prow"><b>${i + 1}</b><span class="pm">${p.m ? fmt(p.m) : '-'}</span>&rarr;<span class="pf">${p.f ? fmt(p.f) : 'click the fixed layer'}</span>${badge}<button data-i="${i}" title="remove this pair">&times;</button></div>`;
    }).join('') || '<span class="muted">no pairs: refines from the current placement (add pairs when the layers are far apart)</span>';
    $('alPicks').querySelectorAll('button').forEach(b => b.onclick = () => { picks.splice(+b.dataset.i, 1); pairRes = []; pickSide = picks.length && !picks[picks.length - 1].f ? 'fixed' : 'moving'; drawPicks(); });
    V3.redraw && V3.redraw();
  }
  V3.hudHooks.push(hx => {                          // pair markers: moving (orange) -> fixed (blue)
    if (SEC !== 'align') return;
    const S = V3.split, O = V3.O;
    if (S) {                                          // divider + labels of the two halves
      const W = V3.W, H = V3.H, nm = id => { const L = PROJ.layers.find(l => l.id === id); return L ? L.name : id; };
      hx.save(); hx.fillStyle = 'rgba(255,255,255,.9)'; hx.fillRect(W / 2 - 1.5, 0, 3, H);
      hx.font = '600 14px Roboto, system-ui, sans-serif'; hx.textBaseline = 'middle';
      [[`MOVING · ${nm($('alMoving').value)}`, 12, '#ff9f43'], [`FIXED · ${nm($('alFixed').value)}`, W / 2 + 12, '#2f8cff']].forEach(([t, x, c]) => {
        const w = hx.measureText(t).width + 20, y = 12; hx.fillStyle = 'rgba(13,15,19,.82)'; hx.beginPath(); hx.roundRect ? hx.roundRect(x, y, w, 28, 8) : hx.rect(x, y, w, 28); hx.fill();
        hx.fillStyle = c; hx.fillRect(x, y, 4, 28); hx.fillStyle = '#fff'; hx.fillText(t, x + 12, y + 14.5);
      });
      hx.restore();
    }
    if (!picks.length) return;
    const rel = (p, side) => { const q = [p[0] - O[0], p[1] - O[1], p[2] - O[2]]; if (!S) return V3.project(q); const r = V3.projectSide(q, side), x0 = side ? V3.W / 2 : 0; return r && r[0] > x0 + 4 && r[0] < x0 + V3.W / 2 - 4 ? r : null; };
    hx.save(); hx.font = '700 12px Roboto, system-ui, sans-serif'; hx.textAlign = 'center'; hx.textBaseline = 'middle';
    picks.forEach((p, i) => {
      const a = p.m && rel(p.m, 0), b = p.f && rel(p.f, 1), bad = pairRes[i] && pairRes[i].suspect;
      if (a && b && !S) { hx.strokeStyle = bad ? '#ff5d5d' : 'rgba(255,255,255,.85)'; hx.lineWidth = 2; hx.setLineDash([5, 4]); hx.beginPath(); hx.moveTo(a[0], a[1]); hx.lineTo(b[0], b[1]); hx.stroke(); hx.setLineDash([]); }
      [[a, '#ff9f43'], [b, '#2f8cff']].forEach(([s, c]) => {
        if (!s) return; hx.fillStyle = c; hx.strokeStyle = bad ? '#ff5d5d' : '#fff'; hx.lineWidth = 2;
        hx.beginPath(); hx.arc(s[0], s[1], 10, 0, 7); hx.fill(); hx.stroke(); hx.fillStyle = '#fff'; hx.fillText(String(i + 1), s[0], s[1] + 0.5);
      });
    });
    hx.restore();
  });
  window.addEventListener('pick3d', ev => {        // inspect clicks while in Align: point pairs (moving, then fixed; side by side: left / right)
    const info = ev.detail; if (SEC !== 'align' || !info || !info.p || !$('alPickOn').checked) return;
    const O = V3.O, P = [info.p[0] + O[0], info.p[1] + O[1], info.p[2] + O[2]];
    pairRes = [];
    const which = info.side === 0 ? 'm' : info.side === 1 ? 'f' : (pickSide === 'moving' ? 'm' : 'f');
    let open = picks[picks.length - 1];
    if (!open || (open.m && open.f) || open[which]) { open = {}; picks.push(open); }
    open[which] = P;
    const other = which === 'm' ? 'FIXED' : 'MOVING', side = V3.split ? (which === 'm' ? ' (right half)' : ' (left half)') : '';
    if (open.m && open.f) { pickSide = 'moving'; toast(`Pair ${picks.length} set` + (picks.length < 3 ? ` - ${3 - picks.length} more for a full 3D start (fewer is fine for small offsets)` : '')); }
    else { pickSide = which === 'm' ? 'fixed' : 'moving'; toast(`Pair ${picks.length}: now click the same spot on the ${other} layer${side}`); }
    drawPicks();
  });
  ['alMoving', 'alFixed', 'alSplit'].forEach(id => $(id).addEventListener('change', () => { updateSplit(); V3.redraw(); }));
  $('alUndo').onclick = () => {
    if (!picks.length) return;
    const last = picks[picks.length - 1]; if (!last) return;
    if (last.f) { delete last.f; pickSide = 'fixed'; } else { picks.pop(); pickSide = 'moving'; }
    pairRes = []; drawPicks();
  };
  $('alClear').onclick = () => { picks = []; pairRes = []; pickSide = 'moving'; drawPicks(); };
  function dirWords(d) {                            // unit vector -> "mostly along x (east)"
    const names = ['x (east)', 'y (north)', 'z (up)'], k = d.map(Math.abs), i = k.indexOf(Math.max(...k));
    return k[i] > 0.9 ? names[i] : `${names[i]} / ${names[k.indexOf([...k].sort((a, b) => b - a)[1])]}`;
  }
  $('alRun').onclick = () => {
    const done = picks.filter(p => p.m && p.f), pk = done.map(p => [...p.m, ...p.f]);
    if (!$('alIcp').checked && !pk.length) { toast('Pick pairs, or switch on surface refinement'); return; }
    job('/align/run', {moving: $('alMoving').value, fixed: $('alFixed').value, picks: pk, mode: $('alMode').value, icp: $('alIcp').checked,
                       max_dist: +$('alDist').value, pick_sigma: +$('alSigma').value, colour: $('alColour').checked ? 0.3 : 0}, 'Aligning', res => {
      lastAlign = res; const R = res.report || {}, b = res.before, a = res.after;
      pairRes = []; (R.pairs || []).forEach(p => { pairRes[picks.indexOf(done[p.i - 1])] = p; }); drawPicks();
      const gap = s => `overlap ${(100 * s.overlap).toFixed(0)}%, median gap ${s.median != null ? (100 * s.median).toFixed(1) + ' cm' : '-'}, within 5 cm ${(100 * s.within_5cm).toFixed(0)}%`;
      const U = R.uncertainty || {}, G = U.with_pairs || U.geometry || {}, W = U.weakest_translation;
      const unc = ['x', 'y', 'z'].filter(k => G[k] != null).map(k => `${k} ±${(100 * G[k]).toFixed(1)} cm`).join(', ')
        + (G['rot z'] != null ? `, heading ±${G['rot z'].toFixed(3)}&deg;` : '');
      const bad = (R.pairs || []).filter(p => p.suspect);
      let warn = '';
      if (W && (W.sigma > 0.02 || W.ratio < 0.02))
        warn = `<div class="alwarn">The surfaces barely pin the shift along ${dirWords(W.direction)}${done.length ? ' - your pairs hold it there' : ''}. For a safer result add a pair at a distinct object (pole, sign, corner) far along that direction.</div>`;
      if (bad.length) warn += `<div class="alwarn">Pair ${bad.map(p => p.i).join(', ')} ${bad.length > 1 ? 'look' : 'looks'} like a mis-click (${bad.map(p => (100 * p.residual).toFixed(0) + ' cm').join(', ')} off) and ${bad.length > 1 ? 'were' : 'was'} down-weighted. Remove or re-pick ${bad.length > 1 ? 'them' : 'it'} and solve again if you like.</div>`;
      if (a.median != null && b.median != null && a.within_5cm < b.within_5cm - 0.02)
        warn += `<div class="alwarn">The fit got worse than the current placement - check the pairs, or narrow the search distance.</div>`;
      $('alResult').innerHTML = `<b>correction ${res.shift.toFixed(3)} m, ${res.rotation_deg.toFixed(3)}&deg;${Math.abs(res.scale - 1) > 1e-5 ? `, scale ${res.scale.toFixed(5)}` : ''}</b>
        ${R.chosen ? `<span class="muted"> (start: ${R.chosen})</span>` : ''}<br>
        before: ${gap(b)}<br>after: ${gap(a)}${unc ? `<br>certainty (1&sigma;): ${unc}` : ''}${warn}`;
      $('alApply').disabled = false;
    });
  };
  $('alApply').onclick = async () => {
    if (!lastAlign) return;
    await api('/align/apply', {id: lastAlign.moving, T: lastAlign.D}); $('alApply').disabled = true;
    toast('Applied to the layer transform - rebuilding'); job('/project/prepare', {}, 'Preparing', () => location.reload());
  };
  drawPicks();

  // mesh / splat toggles (toolbar buttons, layer panel, U / J keys) kept in sync
  function syncToggles() {
    $('tgMesh').classList.toggle('on', window.SHOW_MESH !== false);
    $('tgSplat').classList.toggle('on', typeof SPLAT === 'undefined' || SPLAT.enabled); $('splatOn').checked = typeof SPLAT === 'undefined' || SPLAT.enabled;
    const mb = document.querySelector('#layerList .lrow input'); if (mb) mb.checked = window.SHOW_MESH !== false;
  }
  $('tgMesh').onclick = () => { window.SHOW_MESH = window.SHOW_MESH === false; syncToggles(); V3.redraw(); };
  $('tgSplat').onclick = () => { SPLAT.enabled = !SPLAT.enabled; syncToggles(); };
  setInterval(syncToggles, 700);
  // light / dark theme
  function setTheme(light) {
    document.body.classList.toggle('light', light); $('icSun').style.display = light ? 'none' : ''; $('icMoon').style.display = light ? '' : 'none';
    try { localStorage.setItem('meshsfm_theme', light ? 'light' : 'dark'); } catch (e) {}
    if (typeof needDraw !== 'undefined') needDraw = true;
  }
  $('bTheme').onclick = () => setTheme(!document.body.classList.contains('light'));
  try { setTheme(localStorage.getItem('meshsfm_theme') === 'light'); } catch (e) {}
  // interface size S / M / L (remembered); the canvases are resized to the space left
  function setSize(sz) {
    document.body.dataset.size = sz; document.querySelectorAll('.szbtn').forEach(b => b.classList.toggle('on', b.dataset.size === sz));
    try { localStorage.setItem('meshsfm_size', sz); } catch (e) {}
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  }
  document.querySelectorAll('.szbtn').forEach(b => b.onclick = () => setSize(b.dataset.size));
  try { setSize(localStorage.getItem('meshsfm_size') || 'medium'); } catch (e) { setSize('medium'); }
  // resizable side panel (drag the edge); remembered
  const app = $('app'), grip = $('sideGrip');
  function setSideW(w) { w = Math.max(260, Math.min(window.innerWidth * 0.6, w)); app.style.setProperty('--sideW', w + 'px'); grip.style.left = w + 'px'; window.dispatchEvent(new Event('resize')); }
  let gripDrag = false;
  grip.addEventListener('mousedown', e => { gripDrag = true; grip.classList.add('drag'); e.preventDefault(); });
  window.addEventListener('mousemove', e => { if (gripDrag) setSideW(e.clientX); });
  window.addEventListener('mouseup', () => { if (gripDrag) { gripDrag = false; grip.classList.remove('drag'); try { localStorage.setItem('meshsfm_sideW', getComputedStyle(app).getPropertyValue('--sideW')); } catch (e) {} } });
  try { setSideW(parseFloat(localStorage.getItem('meshsfm_sideW')) || 620); } catch (e) { setSideW(620); }
  // keyboard (3D): U toggles the mesh, J the splats
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA' || TAB !== '3d') return;
    if (e.key === 'u' || e.key === 'U') { window.SHOW_MESH = window.SHOW_MESH === false; refresh(); V3.redraw(); }
    if ((e.key === 'j' || e.key === 'J') && typeof SPLAT !== 'undefined') { SPLAT.enabled = !SPLAT.enabled; $('splatOn').checked = SPLAT.enabled; }
  });
  $('splatOn').onchange = e => { SPLAT.enabled = e.target.checked; };
  $('splatBudget').oninput = e => { SPLAT.budget = +e.target.value * 1e6; $('splatBudgetV').textContent = e.target.value + ' M'; };
  $('splatRadius').oninput = e => { SPLAT.radius = +e.target.value; $('splatRadiusV').textContent = e.target.value + ' m'; };

  READY.then(async () => {
    await refresh();
    setSection(META.mesh ? 'edit' : 'gen');
    if (META.mesh) { V3.start().then(() => SPLAT.setLayers(PROJ)); }
  });
  return {refresh, setSection, browse};
})();
