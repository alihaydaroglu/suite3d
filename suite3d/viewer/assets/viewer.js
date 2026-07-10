/* Suite3D portable ROI viewer.
 *
 * Runs from file://, so: no fetch(), no XHR. Everything arrives through
 * <script src> (meta.js, masks.js, traces/chunk_*.js) and <img src> (plane PNGs,
 * movie frames).
 *
 * Three rendering rules that keep 43k ROIs interactive:
 *   1. Footprints are rasterised ONCE per plane into an Int32Array label map
 *      (ROI id per pixel). Clicking is an O(1) pixel lookup.
 *   2. A repaint touches only the ROIs whose appearance CHANGED. Dragging a
 *      filter thumb flips a few hundred ROIs, so it rewrites a few thousand
 *      pixels -- not the 682k of an SS004 plane. Full repaints happen only when
 *      the plane changes. Everything is coalesced into one rAF.
 *   3. The background plane image is an <img> UNDER a transparent overlay canvas.
 *      Compositing in CSS avoids getImageData, which would throw on file:// (the
 *      canvas is tainted by a local image).
 */
"use strict";

const S3D = (window.S3D = window.S3D || {});
const META = window.S3D_META;
const MASKS = window.S3D_MASKS;

/* ---------- decode ---------- */
function dec(b64, Type) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(buf);
}

const M = {
  off: dec(MASKS.off, Uint32Array),
  z: dec(MASKS.z, Uint8Array),
  y: dec(MASKS.y, Uint16Array),
  x0: dec(MASKS.x0, Uint16Array),
  len: dec(MASKS.len, Uint8Array),
  lam: dec(MASKS.lam, Uint8Array),
};
const STAT = {};
for (const k in META.stats) STAT[k] = dec(META.stats[k], Float32Array);

const NZ = META.shape[0], NY = META.shape[1], NX = META.shape[2];
const NROI = META.n_rois;
const NRUN = M.z.length;

/* ---------- colours: golden angle, matches the figure scripts ---------- */
function roiRGB(i) {
  const h = ((i * 137.508) % 360) / 60, c = 217, x = c * (1 - Math.abs((h % 2) - 1));
  const t = [[c,x,0],[x,c,0],[0,c,x],[0,x,c],[x,0,c],[c,0,x]][Math.floor(h) % 6];
  return [t[0] + 38, t[1] + 38, t[2] + 38];
}
const LUT = new Uint8Array(NROI * 3);
for (let i = 0; i < NROI; i++) { const c = roiRGB(i); LUT[3*i]=c[0]; LUT[3*i+1]=c[1]; LUT[3*i+2]=c[2]; }

/* ---------- label maps + per-plane run index, built once ----------
 * roiOf/planeRuns let a full repaint walk only the runs on the current plane,
 * and let a partial repaint walk only one ROI's runs. */
const labels = [], alpha = [], planeRuns = [];
const roiOf = new Int32Array(NRUN);
(function rasterise() {
  for (let z = 0; z < NZ; z++) { labels.push(new Int32Array(NY * NX).fill(-1)); alpha.push(new Uint8Array(NY * NX)); }
  for (let roi = 0; roi < NROI; roi++)
    for (let r = M.off[roi]; r < M.off[roi + 1]; r++) roiOf[r] = roi;

  // paint in ascending peak_val so the strongest ROI wins a contested pixel
  const order = Array.from({length: NROI}, (_, i) => i);
  const pv = STAT.peak_val;
  if (pv) order.sort((a, b) => (pv[a] || 0) - (pv[b] || 0));
  for (const roi of order) {
    for (let r = M.off[roi]; r < M.off[roi + 1]; r++) {
      const lab = labels[M.z[r]], al = alpha[M.z[r]];
      let idx = M.y[r] * NX + M.x0[r];
      const n = M.len[r], a = M.lam[r];
      for (let k = 0; k < n; k++, idx++) { lab[idx] = roi; al[idx] = a; }
    }
  }
  const counts = new Int32Array(NZ);
  for (let r = 0; r < NRUN; r++) counts[M.z[r]]++;
  for (let z = 0; z < NZ; z++) planeRuns.push(new Int32Array(counts[z]));
  const fill = new Int32Array(NZ);
  for (let r = 0; r < NRUN; r++) { const z = M.z[r]; planeRuns[z][fill[z]++] = r; }
})();

/* ---------- curation: iscell is DERIVED, never stored ----------
 * Mirror of suite3d/viewer/curation.py:apply_curation. Keep them in step.
 * NaN stats never fail a filter -- a missing statistic must not delete ROIs. */
const state = {
  filters: {},        // {stat: [lo, hi]}
  manual: new Map(),  // roi -> 0|1
  plane: Math.floor(NZ / 2),
  bg: "corrmap",
  selected: -1,
  hover: -1,
  onlySelected: false,
  movie: null,        // key into META.movies, or null for the static background
  frame: 0,
  playing: false,
};
let visible = new Uint8Array(NROI);

function applyCuration() {
  visible.fill(1);
  for (const name in state.filters) {
    const [lo, hi] = state.filters[name], v = STAT[name];
    if (!v) continue;
    for (let i = 0; i < NROI; i++) {
      if (!visible[i]) continue;
      const x = v[i];
      if (Number.isFinite(x) && (x < lo || x > hi)) visible[i] = 0;
    }
  }
  for (const [roi, val] of state.manual) visible[roi] = val ? 1 : 0;
}

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const bg = $("bg"), ov = $("ov"), layers = $("layers");
ov.width = NX; ov.height = NY;
const octx = ov.getContext("2d");
const img = octx.createImageData(NX, NY);

/* ---------- painting ----------
 * `drawn[roi]` is the appearance code currently on the canvas. A repaint diffs
 * the desired code against it and repaints only the ROIs that moved. */
const HIDDEN = 0, NORMAL = 1, SELECTED = 2, HOVER = 3;
const drawn = new Uint8Array(NROI);

function codeOf(roi) {
  if (!visible[roi]) return HIDDEN;
  if (state.onlySelected && state.selected >= 0 && roi !== state.selected) return HIDDEN;
  if (roi === state.selected) return SELECTED;
  if (roi === state.hover) return HOVER;
  return NORMAL;
}

// A run's pixels can be stolen by an overlapping stronger ROI, so consult the
// label map rather than assuming the run owns them.
function paintRun(r, roi, code) {
  const lab = labels[M.z[r]], al = alpha[M.z[r]], d = img.data;
  let idx = M.y[r] * NX + M.x0[r];
  for (let k = M.len[r]; k > 0; k--, idx++) {
    if (lab[idx] !== roi) continue;
    const p = idx * 4;
    if (code === HIDDEN) { d[p + 3] = 0; continue; }
    if (code === SELECTED) { d[p] = 255; d[p+1] = 255; d[p+2] = 255; d[p+3] = 255; continue; }
    const b = code === HOVER ? 46 : 0;
    d[p]     = Math.min(255, LUT[3*roi]     + b);
    d[p + 1] = Math.min(255, LUT[3*roi + 1] + b);
    d[p + 2] = Math.min(255, LUT[3*roi + 2] + b);
    d[p + 3] = code === HOVER ? 220 : 40 + ((al[idx] * 0.72) | 0);
  }
}
function paintROI(roi, code) {
  const z = state.plane;
  for (let r = M.off[roi]; r < M.off[roi + 1]; r++) if (M.z[r] === z) paintRun(r, roi, code);
}

function fullPaint() {
  img.data.fill(0);
  // Seeding `drawn` only matters for the first paint -- diffPaint sweeps every ROI
  // and leaves drawn == codeOf, and codeOf does not depend on the plane. Keep it
  // anyway: it makes "the canvas shows `drawn`" true after every entry point.
  for (let i = 0; i < NROI; i++) drawn[i] = codeOf(i);
  const runs = planeRuns[state.plane];
  for (let i = 0; i < runs.length; i++) {
    const r = runs[i], roi = roiOf[r];
    if (drawn[roi] !== HIDDEN) paintRun(r, roi, drawn[roi]);
  }
}
function diffPaint() {
  for (let roi = 0; roi < NROI; roi++) {
    const c = codeOf(roi);
    if (c === drawn[roi]) continue;
    drawn[roi] = c;
    paintROI(roi, c);
  }
}

let pending = 0;            // 0 none, 1 diff, 2 full
function schedule(full) {
  pending = Math.max(pending, full ? 2 : 1);
  if (pending && !schedule.queued) {
    schedule.queued = true;
    requestAnimationFrame(() => {
      schedule.queued = false;
      const p = pending; pending = 0;
      if (p === 2) fullPaint(); else diffPaint();
      octx.putImageData(img, 0, 0);
      updateCounts();
    });
  }
}
function updateCounts() {
  let n = 0;
  for (let i = 0; i < NROI; i++) n += visible[i];
  const pct = NROI ? (100 * n / NROI).toFixed(0) : 0;
  $("counts").innerHTML = `<b>${n.toLocaleString()}</b> / ${NROI.toLocaleString()} ROIs pass · ${pct}%`;
  $("hud").textContent = state.hover >= 0
    ? `ROI ${state.hover}   ·   plane ${state.plane} / ${NZ - 1}`
    : `plane ${state.plane} / ${NZ - 1}   ·   ${NY}×${NX}`;
}

/* ---------- background planes ---------- */
function planeURL(z, which) { return `viewer/planes/z${String(z).padStart(2, "0")}_${which}.png`; }
function setBackground() { bg.src = planeURL(state.plane, state.bg); }
function prefetchPlanes() {
  // neighbours only: a decoded SS004 plane is ~2.7 MB of RGBA, and there are 22.
  for (const dz of [-1, 1]) {
    const z = state.plane + dz;
    if (z >= 0 && z < NZ) new Image().src = planeURL(z, state.bg);
  }
}
function setPlane() {
  $("plane").value = state.plane;
  $("hudp").textContent = state.plane;
  setBackground(); prefetchPlanes(); prefetchMovie(); showFrame(); schedule(true);
}

/* ---------- movie snippets ----------
 * One <img> per frame, swapped by src. Never drawn into a canvas: a file:// image
 * taints it and getImageData throws. Frames for the current plane are prefetched
 * into the browser's image cache so playback does not stutter on the first pass. */
const MOVIES = META.movies || {};
const movImg = $("mov");
const prefetched = new Set();

function frameURL(key, z, t) {
  const m = MOVIES[key];
  return `viewer/movies/${key}/z${String(z).padStart(2, "0")}_t${String(t).padStart(4, "0")}.${m.ext}`;
}
// A snippet may cover only some planes (`planes=` in the spec). Requesting a frame
// that was never written just leaves a blank <img> on file://, with nothing to
// catch -- so check first and say so.
function hasPlane(key, z) {
  const p = MOVIES[key].planes;
  return !p || p.indexOf(z) >= 0;
}
function prefetchMovie() {
  if (!state.movie || !hasPlane(state.movie, state.plane)) return;
  const key = `${state.movie}:${state.plane}`;
  if (prefetched.has(key)) return;
  prefetched.add(key);
  for (let t = 0; t < MOVIES[state.movie].n_frames; t++) new Image().src = frameURL(state.movie, state.plane, t);
}
function showFrame() {
  if (!state.movie) { movImg.hidden = true; return; }
  if (!hasPlane(state.movie, state.plane)) {
    movImg.hidden = true;
    $("movhint").textContent = `no movie frames for plane ${state.plane} (snippet covers ${MOVIES[state.movie].planes.join(", ")})`;
    return;
  }
  const m = MOVIES[state.movie];
  state.frame = Math.max(0, Math.min(m.n_frames - 1, state.frame));
  movImg.src = frameURL(state.movie, state.plane, state.frame);
  movImg.hidden = false;
  movHint();                              // clears any "no frames for this plane"
  $("frame").value = state.frame;
  $("frameout").textContent = state.frame;
  drawCursor();
}
function movHint() {
  const m = MOVIES[state.movie];
  const ds = m.downsample > 1 ? `, ${m.downsample}× downsampled` : "";
  const zs = m.planes && m.planes.length < NZ ? `, planes ${m.planes.join("/")}` : "";
  $("movhint").textContent = `${m.label} · ${m.n_frames} frames from t=${m.t0}${ds}${zs}`;
}
function traceIndexOfFrame() {          // movie frame -> sample on the trace time axis
  if (!state.movie) return -1;
  const m = MOVIES[state.movie];
  return m.t0 + state.frame * m.stride;
}
function setMovie(key) {
  state.movie = key || null;
  if (state.movie) {
    const m = MOVIES[state.movie];
    $("frame").max = m.n_frames - 1;
    state.frame = Math.min(state.frame, m.n_frames - 1);
    movHint();
    prefetchMovie();
  } else {
    stopPlay();
    $("movhint").textContent = "";
  }
  $("movsel").value = state.movie || "";
  showFrame();
}
let playTimer = null;
function stopPlay() { state.playing = false; clearInterval(playTimer); playTimer = null; $("play").textContent = "▶ Play"; }
function togglePlay() {
  if (state.playing) return stopPlay();
  if (!state.movie) return;
  state.playing = true; $("play").textContent = "❚❚ Pause";
  playTimer = setInterval(() => {
    state.frame = (state.frame + 1) % MOVIES[state.movie].n_frames;
    showFrame();
  }, 1000 / +$("fps").value);
}

/* ---------- zoom / pan ---------- */
let scale = 1, tx = 0, ty = 0, minScale = 0.05;
function fit() {
  const st = $("stage");
  scale = Math.min(st.clientWidth / NX, st.clientHeight / NY) * 0.96;
  minScale = scale * 0.5;
  tx = (st.clientWidth - NX * scale) / 2; ty = (st.clientHeight - NY * scale) / 2;
  applyTransform();
}
function applyTransform() { layers.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`; }
$("stage").addEventListener("wheel", (e) => {
  e.preventDefault();
  const st = $("stage").getBoundingClientRect();
  const r = layers.getBoundingClientRect();
  const px = (e.clientX - r.left) / scale, py = (e.clientY - r.top) / scale;
  scale = Math.max(minScale, Math.min(60, scale * Math.exp(-e.deltaY * 0.0015)));
  tx = e.clientX - st.left - px * scale;      // keep the pixel under the cursor fixed
  ty = e.clientY - st.top - py * scale;
  applyTransform();
}, { passive: false });
let drag = null;
$("stage").addEventListener("pointerdown", (e) => { drag = { x: e.clientX - tx, y: e.clientY - ty, moved: false }; });
window.addEventListener("pointermove", (e) => {
  if (!drag) return;
  tx = e.clientX - drag.x; ty = e.clientY - drag.y;
  if (!drag.moved && Math.abs(e.movementX) + Math.abs(e.movementY) > 0) drag.moved = true;
  applyTransform();
});
window.addEventListener("pointerup", () => { drag = null; });

/* ---------- picking ---------- */
function roiAt(e) {
  const r = ov.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) / (r.width / NX));
  const y = Math.floor((e.clientY - r.top) / (r.height / NY));
  if (x < 0 || y < 0 || x >= NX || y >= NY) return -1;
  const roi = labels[state.plane][y * NX + x];
  return roi >= 0 && visible[roi] ? roi : -1;
}
ov.addEventListener("pointermove", (e) => {
  if (drag) return;
  const roi = roiAt(e);
  if (roi === state.hover) return;
  state.hover = roi;                        // repaints exactly two ROIs
  ov.style.cursor = roi >= 0 ? "pointer" : "grab";
  schedule(false);
});
ov.addEventListener("pointerleave", () => { if (state.hover >= 0) { state.hover = -1; schedule(false); } });
ov.addEventListener("click", (e) => {
  if (drag && drag.moved) return;
  const roi = roiAt(e);
  if (roi < 0) { select(-1); return; }
  if (e.altKey) { toggleManual(roi); return; }
  select(roi);
});
function toggleManual(roi) {
  state.manual.set(roi, visible[roi] ? 0 : 1);
  applyCuration(); schedule(false); persist();
  if (roi === state.selected) $("incl").textContent = visible[roi] ? "cell" : "not cell";
}
function select(roi) {
  state.selected = roi;
  schedule(state.onlySelected);            // "only selected" changes every ROI
  if (roi < 0) { $("tracebar").classList.remove("open"); shown = null; return; }
  showRoi(roi);
}

/* ---------- traces: lazy JSONP chunks ---------- */
const chunks = new Map();     // id -> {F, Fneu, spks, scale, offset} decoded
const pending_ = new Map();
S3D.trace = function (id, payload) {
  const nroi = payload.n, nt = payload.nt;
  const out = {};
  for (const k of ["F", "Fneu", "spks"]) {
    if (!payload[k]) continue;
    const q = META.trace_dtype === "int16" ? dec(payload[k], Int16Array) : dec(payload[k], Float32Array);
    out[k] = { q, scale: dec(payload[k + "_scale"], Float32Array), offset: dec(payload[k + "_offset"], Float32Array) };
  }
  out.nt = nt; out.nroi = nroi;
  chunks.set(id, out);
  const waiters = pending_.get(id) || [];
  pending_.delete(id);
  waiters.forEach((fn) => fn(out));
};
function loadChunk(id, cb) {
  if (chunks.has(id)) return cb(chunks.get(id));
  if (pending_.has(id)) { pending_.get(id).push(cb); return; }
  pending_.set(id, [cb]);
  const s = document.createElement("script");
  s.src = `viewer/traces/chunk_${String(id).padStart(4, "0")}.js`;
  s.onerror = () => { pending_.delete(id); $("tracemsg").textContent = `trace chunk ${id} missing`; };
  document.head.appendChild(s);
}
function dequant(ch, key, row) {
  const nt = ch.nt, q = ch[key].q, s = ch[key].scale[row], o = ch[key].offset[row];
  const out = new Float32Array(nt);
  const base = row * nt;
  for (let t = 0; t < nt; t++) out[t] = q[base + t] * s + o;
  return out;
}

function showRoi(roi) {
  $("tracebar").classList.add("open");
  const sw = $("sw"); const c = roiRGB(roi);
  sw.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
  $("roiid").textContent = roi;
  $("npix").textContent = STAT.npix ? STAT.npix[roi] : "–";
  $("zspan").textContent = STAT.zspan ? STAT.zspan[roi] : "–";
  $("pv").textContent = STAT.peak_val ? STAT.peak_val[roi].toFixed(3) : "–";
  $("incl").textContent = visible[roi] ? "cell" : "not cell";

  if (!META.has_traces) { $("tracemsg").textContent = "no F.npy in this run — traces unavailable"; return; }
  $("tracemsg").textContent = "loading…";
  const id = Math.floor(roi / META.chunk_rois), row = roi % META.chunk_rois;
  loadChunk(id, (ch) => {
    if (state.selected !== roi) return;    // user moved on while the chunk loaded
    $("tracemsg").textContent = ""; shown = { ch, row }; drawTraces(ch, row);
  });
}

// Redrawing the whole trace canvas per movie frame is cheap (a few thousand
// points) and keeps the cursor in the same coordinate system as the traces.
let shown = null;
function drawCursor() { if (shown) drawTraces(shown.ch, shown.row); }

function drawTraces(ch, row) {
  const cv = $("traces"), W = cv.clientWidth, H = 190;
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  cv.style.height = H + "px";
  const g = cv.getContext("2d"); g.scale(devicePixelRatio, devicePixelRatio);
  g.clearRect(0, 0, W, H);
  const keys = ["F", "Fneu", "spks"].filter((k) => ch[k]);
  const cols = { F: "#63d8a4", Fneu: "#8b93a1", spks: "#f2a65a" };
  const h = H / keys.length, nt = ch.nt, fs = META.fs_vol;
  keys.forEach((k, i) => {
    const v = dequant(ch, k, row);
    let lo = Infinity, hi = -Infinity;
    for (let t = 0; t < nt; t++) { if (v[t] < lo) lo = v[t]; if (v[t] > hi) hi = v[t]; }
    if (hi === lo) hi = lo + 1;
    const y0 = i * h + 6, y1 = (i + 1) * h - 12;
    g.strokeStyle = cols[k]; g.lineWidth = 1; g.beginPath();
    if (k === "spks") {                      // sparse, non-negative -> stems
      for (let t = 0; t < nt; t++) {
        if (v[t] <= 0) continue;
        const x = (t / (nt - 1)) * (W - 54) + 48;
        const yy = y1 - ((v[t] - lo) / (hi - lo)) * (y1 - y0);
        g.moveTo(x, y1); g.lineTo(x, yy);
      }
    } else {
      for (let t = 0; t < nt; t++) {
        const x = (t / (nt - 1)) * (W - 54) + 48;
        const yy = y1 - ((v[t] - lo) / (hi - lo)) * (y1 - y0);
        t ? g.lineTo(x, yy) : g.moveTo(x, yy);
      }
    }
    g.stroke();
    g.fillStyle = cols[k]; g.font = "11px system-ui";
    g.fillText(k, 6, y0 + 10);
    g.fillStyle = "#9aa2ae"; g.font = "10px system-ui";
    g.fillText(hi.toFixed(0), 6, y0 + 22);
  });

  // where the movie currently sits on the trace time axis
  const ti = traceIndexOfFrame();
  if (ti >= 0 && ti < nt) {
    const x = (ti / (nt - 1)) * (W - 54) + 48;
    g.strokeStyle = "#4c9ffe"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H - 12); g.stroke();
  }

  g.fillStyle = "#9aa2ae"; g.font = "10px system-ui";
  g.fillText("0 s", 48, H - 1);
  g.fillText(((nt - 1) / fs).toFixed(1) + " s", W - 46, H - 1);
}

/* ---------- iscell.npy, built in the browser ---------- */
function iscellNpy() {
  const n = NROI;
  const hdrObj = `{'descr': '|b1', 'fortran_order': False, 'shape': (${n},), }`;
  let hdr = hdrObj;
  const pre = 10;                                     // magic(6)+ver(2)+len(2)
  while ((pre + hdr.length + 1) % 64 !== 0) hdr += " ";
  hdr += "\n";
  const buf = new Uint8Array(pre + hdr.length + n);
  buf.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  buf[8] = hdr.length & 0xff; buf[9] = (hdr.length >> 8) & 0xff;
  for (let i = 0; i < hdr.length; i++) buf[pre + i] = hdr.charCodeAt(i);
  for (let i = 0; i < n; i++) buf[pre + hdr.length + i] = visible[i];
  return buf;
}
function download(name, bytes, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes], { type: mime || "application/octet-stream" }));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}

/* ---------- persistence (localStorage, keyed by job) ---------- */
const KEY = `s3d-curation:${META.job_id}:${NROI}`;
let persistTimer = null;
function persist() {                       // debounced: a thumb drag fires ~60x/s
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    localStorage.setItem(KEY, JSON.stringify({
      schema: 1, filters: state.filters, manual: Object.fromEntries(state.manual),
    }));
  }, 250);
}
function restore() {
  try {
    const d = JSON.parse(localStorage.getItem(KEY) || "null");
    if (!d) return false;
    // drop stats this run does not have, so an old session cannot delete every ROI
    state.filters = {};
    for (const k in d.filters || {}) if (STAT[k]) state.filters[k] = d.filters[k];
    state.manual = new Map(Object.entries(d.manual || {}).map(([k, v]) => [+k, +v]));
    return true;
  } catch (e) { return false; }
}

/* ---------- filters: dual-thumb sliders over a histogram ----------
 * Two <input type=range> cannot express "a range", and npix spans 3 decades, so
 * a linear thumb spends 90% of its travel on 1% of the ROIs. Each filter draws
 * its own distribution, highlights the selected band, and maps position -> value
 * on a per-stat scale. `state.filters` still holds plain [lo, hi] in stat units,
 * so the Python predicate is untouched. */
const FILTERS = [
  ["npix", "voxels", "log"], ["zspan", "z-planes", "linear"],
  ["peak_val", "peak val", "linear"], ["vox_snr", "vox SNR", "linear"],
];
const NBINS = 64, PAD = 7, HH = 34;         // thumb radius margin, histogram height

function mkScale(kind, lo0, hi0) {
  const log = kind === "log" && lo0 >= 0;
  const a = log ? Math.log1p(lo0) : lo0;
  const b = log ? Math.log1p(hi0) : hi0;
  const span = (b - a) || 1;
  return {
    lo0, hi0, int: Number.isInteger(lo0) && Number.isInteger(hi0),
    toPos: (v) => Math.max(0, Math.min(1, ((log ? Math.log1p(Math.max(v, 0)) : v) - a) / span)),
    toVal: (p) => { const x = a + Math.max(0, Math.min(1, p)) * span; return log ? Math.expm1(x) : x; },
  };
}

function buildFilters() {
  const host = $("filters");
  host.innerHTML = "";
  for (const [key, label, kind] of FILTERS) {
    const v = STAT[key];
    if (!v) continue;
    let lo0 = Infinity, hi0 = -Infinity, nfin = 0;
    for (let i = 0; i < NROI; i++) { const x = v[i]; if (!Number.isFinite(x)) continue; nfin++; if (x < lo0) lo0 = x; if (x > hi0) hi0 = x; }
    if (!nfin || lo0 === hi0) continue;

    const sc = mkScale(kind, lo0, hi0);
    if (!(key in state.filters)) state.filters[key] = [lo0, hi0];
    const f = state.filters[key];
    f[0] = Math.max(lo0, Math.min(f[0], hi0));      // clamp a restored range
    f[1] = Math.max(lo0, Math.min(f[1], hi0));

    const hist = new Float32Array(NBINS);
    for (let i = 0; i < NROI; i++) {
      const x = v[i];
      if (!Number.isFinite(x)) continue;
      hist[Math.min(NBINS - 1, (sc.toPos(x) * NBINS) | 0)]++;
    }
    let hmax = 0;
    for (let b = 0; b < NBINS; b++) hmax = Math.max(hmax, hist[b]);

    const el = document.createElement("div");
    el.className = "filt";
    el.innerHTML = `<div class="fhdr"><span>${label}${kind === "log" ? " <i>log</i>" : ""}</span>` +
                   `<span class="fval"></span></div><canvas class="fcv"></canvas>`;
    host.appendChild(el);
    const cv = el.querySelector("canvas"), out = el.querySelector(".fval");
    const fmt = (x) => (sc.int ? Math.round(x).toString() : (hi0 - lo0 > 20 ? x.toFixed(1) : x.toFixed(3)));

    const draw = () => {
      const W = cv.clientWidth || 220, H = HH + 2 * PAD;
      cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
      cv.style.height = H + "px";
      const g = cv.getContext("2d");
      g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      g.clearRect(0, 0, W, H);
      const x0 = PAD, w = W - 2 * PAD, base = PAD + HH;
      const pLo = sc.toPos(f[0]), pHi = sc.toPos(f[1]);
      const bw = w / NBINS;
      for (let b = 0; b < NBINS; b++) {
        const c = (b + 0.5) / NBINS;
        // sqrt so a bin with 20 ROIs is visible next to one with 20,000
        const h = hmax ? Math.sqrt(hist[b] / hmax) * HH : 0;
        g.fillStyle = (c >= pLo && c <= pHi) ? "#4c9ffe" : "#39404b";
        g.fillRect(x0 + b * bw, base - h, Math.max(1, bw - 0.6), h);
      }
      g.strokeStyle = "#2a2f38"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0, base + 0.5); g.lineTo(x0 + w, base + 0.5); g.stroke();
      for (const p of [pLo, pHi]) {
        const cx = x0 + p * w;
        g.beginPath(); g.arc(cx, base, 5, 0, 6.2832);
        g.fillStyle = "#e6e8ec"; g.fill();
        g.strokeStyle = "#101216"; g.lineWidth = 1.5; g.stroke();
      }
      out.textContent = `${fmt(f[0])} – ${fmt(f[1])}`;
    };

    const posAt = (e) => {
      const r = cv.getBoundingClientRect();
      return Math.max(0, Math.min(1, (e.clientX - r.left - PAD) / (r.width - 2 * PAD)));
    };
    const setThumb = (which, p) => {
      let val = sc.toVal(p);
      if (sc.int) val = Math.round(val);
      f[which] = which === 0 ? Math.min(val, f[1]) : Math.max(val, f[0]);
      draw();
      applyCuration(); schedule(false); persist();
    };
    let grabbed = -1;
    cv.addEventListener("pointerdown", (e) => {
      const p = posAt(e);
      grabbed = Math.abs(p - sc.toPos(f[0])) <= Math.abs(p - sc.toPos(f[1])) ? 0 : 1;
      cv.setPointerCapture(e.pointerId);
      setThumb(grabbed, p);
      e.stopPropagation();
    });
    cv.addEventListener("pointermove", (e) => { if (grabbed >= 0) setThumb(grabbed, posAt(e)); });
    cv.addEventListener("pointerup", (e) => { grabbed = -1; cv.releasePointerCapture(e.pointerId); });
    cv.addEventListener("dblclick", () => { f[0] = lo0; f[1] = hi0; draw(); applyCuration(); schedule(false); persist(); });

    draw();
    el._draw = draw;
  }
}
function redrawFilters() { for (const el of $("filters").children) el._draw && el._draw(); }

/* ---------- boot ---------- */
function boot() {
  $("title").textContent = META.job_id;
  $("sub").textContent = `${NROI.toLocaleString()} ROIs · ${NZ}×${NY}×${NX} · ${META.n_frames} frames @ ${META.fs_vol.toFixed(3)} Hz`;
  if (META.fs_warning) { $("fswarn").textContent = META.fs_warning; }
  restore();
  buildFilters();
  applyCuration();

  const pl = $("plane");
  pl.max = NZ - 1; pl.value = state.plane;
  pl.addEventListener("input", () => { state.plane = +pl.value; setPlane(); });
  $("bgsel").addEventListener("change", (e) => { state.bg = e.target.value; setBackground(); prefetchPlanes(); });
  $("only").addEventListener("change", (e) => { state.onlySelected = e.target.checked; schedule(true); });
  $("fit").addEventListener("click", fit);
  $("reset").addEventListener("click", () => {
    state.filters = {}; state.manual.clear();
    buildFilters(); applyCuration(); schedule(false); persist();
  });
  $("dlnpy").addEventListener("click", () => download("iscell.npy", iscellNpy()));
  $("dljson").addEventListener("click", () => download("curation.json",
    new TextEncoder().encode(JSON.stringify({ schema: 1, job_id: META.job_id, n_rois: NROI,
      filters: state.filters, manual: Object.fromEntries(state.manual) }, null, 1)), "application/json"));

  const keys = Object.keys(MOVIES);
  if (keys.length) {
    $("moviebox").hidden = false;
    const sel = $("movsel");
    sel.innerHTML = `<option value="">— none —</option>` +
      keys.map((k) => `<option value="${k}">${MOVIES[k].label}</option>`).join("");
    sel.addEventListener("change", (e) => setMovie(e.target.value));
    $("frame").addEventListener("input", (e) => { stopPlay(); state.frame = +e.target.value; showFrame(); });
    $("fps").addEventListener("input", (e) => {
      $("fpsout").textContent = e.target.value;
      if (state.playing) { stopPlay(); togglePlay(); }
    });
    $("play").addEventListener("click", togglePlay);
    $("movoff").addEventListener("click", () => setMovie(""));
  }

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      state.plane = Math.max(0, Math.min(NZ - 1, state.plane + (e.key === "ArrowUp" ? 1 : -1)));
      setPlane(); e.preventDefault();
    }
    if (e.key === " " && state.movie) { togglePlay(); e.preventDefault(); }
    if (e.key === "Escape") select(-1);
    if (e.key === "f") fit();
  });

  bg.width = NX; bg.height = NY;
  movImg.style.width = NX + "px"; movImg.style.height = NY + "px";
  ov.style.cursor = "grab";
  $("hudp").textContent = state.plane;
  bg.addEventListener("load", () => { fit(); }, { once: true });   // fit ONCE, not on every plane
  setBackground();
  prefetchPlanes();
  schedule(true);
  window.addEventListener("resize", () => { fit(); redrawFilters(); });
}
document.addEventListener("DOMContentLoaded", boot);
