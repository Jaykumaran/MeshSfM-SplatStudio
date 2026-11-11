'use strict';
// Shared by the map and the 3D view: tabs, metadata, GPS conversion, help, small helpers.
const $ = id => document.getElementById(id);
let META = null;          // /meta: ortho tiles, route, georef
let TAB = 'map';
const READY = (async () => { META = await (await fetch('/meta')).json(); return META; })();

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  return r.json();
}

// ------------------------------------------------------------------------------------------ tabs
const tabHooks = [];
function setTab(t) {
  if (t === TAB) return;
  TAB = t;
  document.body.classList.toggle('tab3d', t === '3d');
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  $('mapWrap').style.display = t === 'map' ? '' : 'none';
  $('v3Wrap').style.display = t === '3d' ? '' : 'none';
  $('mapTools').style.display = t === 'map' ? '' : 'none';
  $('tools3d').style.display = t === '3d' ? '' : 'none';
  $('sScaleWrap').style.display = t === 'map' ? '' : 'none';
  tabHooks.forEach(f => f(t));
}
document.querySelectorAll('.tab').forEach(b => b.onclick = () => setTab(b.dataset.tab));

// ------------------------------------------------------------------------------------------- GPS
// World = the project frame: z up, metres; (x, y) horizontal.
function geoCorr(x, y) {
  const c = META.geo.corr_grid;
  const fx = (x - c.x0) / c.step, fy = (y - c.y0) / c.step;
  const i0 = Math.max(0, Math.min(c.nx - 2, Math.floor(fx))), j0 = Math.max(0, Math.min(c.ny - 2, Math.floor(fy)));
  const ax = Math.max(0, Math.min(1, fx - i0)), ay = Math.max(0, Math.min(1, fy - j0));
  const bil = g => g[i0][j0] * (1 - ax) * (1 - ay) + g[i0 + 1][j0] * ax * (1 - ay) + g[i0][j0 + 1] * (1 - ax) * ay + g[i0 + 1][j0 + 1] * ax * ay;
  return [bil(c.east), bil(c.north)];
}
function toGps(x, y) {
  const g = META && META.geo; if (!g) return null;
  const c = g.corr_grid ? geoCorr(x, y) : [0, 0];
  const e = g.A[0][0] * x + g.A[0][1] * y + g.t[0] + c[0], n = g.A[1][0] * x + g.A[1][1] * y + g.t[1] + c[1];
  return [g.lat0 + n / g.meridian_radius * 180 / Math.PI, g.lon0 + e / (g.normal_radius * Math.cos(g.lat0 * Math.PI / 180)) * 180 / Math.PI];
}
function fromGps(lat, lon) {
  const g = META.geo;
  const n = (lat - g.lat0) * Math.PI / 180 * g.meridian_radius, e = (lon - g.lon0) * Math.PI / 180 * g.normal_radius * Math.cos(g.lat0 * Math.PI / 180);
  const det = g.A[0][0] * g.A[1][1] - g.A[0][1] * g.A[1][0];
  let x = 0, y = 0;
  for (let it = 0; it < 4; it++) {
    const c = g.corr_grid ? geoCorr(x, y) : [0, 0];
    const be = e - g.t[0] - c[0], bn = n - g.t[1] - c[1];
    x = (g.A[1][1] * be - g.A[0][1] * bn) / det; y = (-g.A[1][0] * be + g.A[0][0] * bn) / det;
  }
  return [x, y];
}
function gpsText(x, y) {
  const p = toGps(x, y); if (!p) return 'no georef';
  return `${p[0].toFixed(6)}, ${p[1].toFixed(6)}`;
}
function parseLatLon(s) {
  const m = s.trim().match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/);
  return m ? [+m[1], +m[2]] : null;
}
const gotoHooks = [];
$('gotoGps').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const ll = parseLatLon(e.target.value);
  if (!ll) { toast('Type "lat, lon" in degrees, e.g. 22.881830, 77.657740'); return; }
  const p = fromGps(ll[0], ll[1]); gotoHooks.forEach(f => f(p[0], p[1])); e.target.blur();
});
$('sGps').title = 'Click to copy';
$('sGps').style.cursor = 'copy';
$('sGps').onclick = () => { navigator.clipboard && navigator.clipboard.writeText($('sGps').textContent); toast('GPS copied'); };

// ------------------------------------------------------------------------------------------ help
$('helpBox').innerHTML = `
  <h2>MeshSfM SplatStudio: keys and mouse</h2>
  <p>Sections (top bar): <b>1 Generate</b> project, layers, mesh generation, Prepare &middot; <b>2 Edit</b> clean / crop / smooth / fill / export &middot; <b>3 Align</b> layer to layer (picks + ICP) &middot; <b>4 Annotate</b> lanes, road segmentation, drive path &middot; <b>5 Drive</b> mesh and / or splat.</p>
  <h4>Everywhere</h4><table>
    <tr><td><kbd>Tab</kbd></td><td>Switch Map / 3D</td></tr>
    <tr><td><kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Y</kbd></td><td>Undo / redo (lanes on the map, mesh edits in 3D)</td></tr>
    <tr><td><kbd>Ctrl+S</kbd></td><td>Save lanes + export lane files</td></tr>
    <tr><td>GPS box</td><td>Type "lat, lon" + Enter to go there; click the GPS readout to copy it</td></tr>
  </table>
  <h4>Map: navigate</h4><table>
    <tr><td>Wheel, <kbd>+</kbd>/<kbd>-</kbd></td><td>Zoom</td></tr>
    <tr><td>Right / middle drag, arrows</td><td>Pan (<kbd>Space</kbd>+drag too)</td></tr>
    <tr><td><kbd>A</kbd></td><td>Follow road on / off (road kept pointing up while panning / jumping)</td></tr>
    <tr><td><kbd>Q</kbd>/<kbd>E</kbd>, <kbd>Shift+A</kbd></td><td>Rotate 5&deg; / north up (both turn follow-road off)</td></tr>
    <tr><td><kbd>PgDn</kbd>/<kbd>PgUp</kbd></td><td>&plusmn;20 m along the route (Shift 100 m)</td></tr>
    <tr><td><kbd>T</kbd></td><td>Open the 3D view here</td></tr>
  </table>
  <h4>Map: lanes</h4><table>
    <tr><td><kbd>V</kbd> Edit</td><td>Click line to select, drag keynodes, Shift+click/drag multi-select, double-click inserts, <kbd>Del</kbd> deletes</td></tr>
    <tr><td><kbd>P</kbd> Pen</td><td>Click keynodes; Enter / double-click / Esc finish; Backspace removes last; start on a line end to continue it</td></tr>
    <tr><td><kbd>F</kbd> Free draw</td><td>Drag; becomes keynodes</td></tr>
    <tr><td><kbd>I</kbd> <kbd>D</kbd> <kbd>X</kbd> <kbd>J</kbd> <kbd>M</kbd></td><td>Insert, erase node, split, join, measure</td></tr>
    <tr><td><kbd>1</kbd>-<kbd>6</kbd></td><td>Line type; <kbd>Shift+R</kbd> reverse; <kbd>C</kbd> continue; <kbd>K</kbd> reviewed</td></tr>
    <tr><td><kbd>G</kbd> Fill area</td><td>Click a contour around a hole, Enter: generates the patch (preview in 3D)</td></tr>
  </table>
  <h4>3D: move</h4><table>
    <tr><td><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></td><td>Fly forward / left / back / right (<kbd>Shift</kbd> 5&times; faster)</td></tr>
    <tr><td><kbd>E</kbd> / <kbd>Q</kbd></td><td>Up / down</td></tr>
    <tr><td>Arrows</td><td>Turn left/right (full 360&deg;) and look up/down</td></tr>
    <tr><td>Drag (Navigate) / right-drag (any tool)</td><td>Look around</td></tr>
    <tr><td>Wheel</td><td>Move forward / back; <kbd>Ctrl</kbd>+wheel = field of view</td></tr>
    <tr><td>Double-click</td><td>Fly to that point</td></tr>
    <tr><td><kbd>T</kbd></td><td>Drive mode: W/S speed along the route, A/D/arrows look, Z/X eye height, Space brake, PgUp/PgDn jump</td></tr>
    <tr><td><kbd>N</kbd></td><td>Colours / lit grey geometry</td></tr>
  </table>
  <h4>3D: mesh tools</h4><table>
    <tr><td><kbd>B</kbd> Brush</td><td>Drag to erase triangles; Alt+drag restores; <kbd>[</kbd>/<kbd>]</kbd> radius; "protect ground" keeps the surface</td></tr>
    <tr><td><kbd>M</kbd> Smooth</td><td>Drag over spikes / bumps / pits: evens the surface and blends it into the neighbours (colours too), closes holes under the stroke. Strength / height range in the side panel; one undo step per stroke</td></tr>
    <tr><td><kbd>L</kbd> Lasso</td><td>Draw an outline: visible triangles inside turn red; Enter applies, Esc drops</td></tr>
    <tr><td><kbd>O</kbd> Pick blob</td><td>Click a floater: the connected piece above the ground turns red</td></tr>
    <tr><td><kbd>G</kbd> Fill area</td><td>Click points on the surface around a hole / rough area; Enter reshapes the surface inside (levelled or smooth, feathered to the contour) and closes real holes</td></tr>
    <tr><td><kbd>P</kbd> Edit lanes</td><td>Lines drawn on the surface (visible through trees). Drag keynodes, Alt+click / double-click inserts, Shift+click adds at the nearer end or starts a new line, Del deletes, 1-6 type, Ctrl+Z undoes lane edits</td></tr>
    <tr><td><kbd>I</kbd> Inspect</td><td>Click: position, GPS, height above ground</td></tr>
    <tr><td>Automatic</td><td>Side panel: Preview shows red/green; Apply (Enter) or Drop (Esc). Every change can be undone</td></tr>
    <tr><td>Export mesh</td><td>Writes the edited parts + the merged mesh to the project's exports/</td></tr>
    <tr><td>Right-drag</td><td>Map: rotate freely &middot; 3D: orbit around the point under the mouse</td></tr>
    <tr><td><kbd>U</kbd> / <kbd>J</kbd></td><td>3D: mesh on / off, splats on / off</td></tr>
  </table>`;
$('help').onclick = e => { if (e.target.id === 'help') $('help').style.display = 'none'; };

// Tab key switches Map / 3D (not while typing)
window.addEventListener('keydown', e => {
  if (e.key !== 'Tab' || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' && e.target.type !== 'range' && e.target.type !== 'checkbox' || tag === 'select' || tag === 'textarea') return;
  e.preventDefault(); setTab(TAB === 'map' ? '3d' : 'map');
}, true);
