/* Live jigsaw demo of NIMBUS, run in the browser.
 *
 * A generated picture is cut into N = n x n square tiles. The spins x[t, p] = 1 place tile t in slot p; the
 * constraint is a permutation (every tile in exactly one slot, every slot holding one tile), as in graph
 * matching. The energy is quadratic in the spins: the sum, over neighbouring slots, of the edge mismatch of
 * the two tiles placed there (prediction-based dissimilarity of the boundary pixels).
 *
 * Every move is a fixed permutation of the slots, so it maps every complete arrangement to another one, and
 * its effect on every product of spins is set by one product of spins (one switch per connection), as for the
 * row exchanges of graph matching (nimbus.transitions.builders.column_permutation, block_exchanges):
 *   exchange the tiles of two slots;
 *   exchange two non-overlapping blocks of h x w slots (h, w <= n/2);
 *   shift a block of slots by k slots along its rows or columns, the displaced columns or rows wrapping to the
 *   other side of the block (every k, so the inverse of each shift is in the library).
 * Proposal: an exchange of two slots with probability 0.5, otherwise a block move (a fixed backbone share).
 * Annealing as in the package: accept iff Delta E <= -tau ln u; Fowler-Nordheim schedule with eta = 0.1;
 * acceptance-rule calibration with minimum ratio kappa = 64 (see calibrate).
 */
(function () {
  "use strict";

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffled(arr, rng) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ---------- picture ----------
  function paint(canvas, seed) {
    const rng = mulberry32(seed);
    const W = canvas.width, H = canvas.height, ctx = canvas.getContext("2d");
    const hz = H * (0.56 + 0.08 * rng());                       // horizon
    const hue = 200 + 80 * rng();                                // sky hue
    const sky = ctx.createLinearGradient(0, 0, 0, hz);
    sky.addColorStop(0, `hsl(${hue + 10}, 60%, 26%)`);
    sky.addColorStop(0.45, `hsl(${hue + 60}, 55%, 48%)`);
    sky.addColorStop(0.8, `hsl(${(hue + 140) % 360}, 85%, 64%)`);
    sky.addColorStop(1, `hsl(${(hue + 170) % 360}, 95%, 74%)`);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, hz);
    const flip = rng() < 0.5;                                     // colour shift from left to right
    const side = ctx.createLinearGradient(flip ? W : 0, 0, flip ? 0 : W, 0);
    side.addColorStop(0, "rgba(10,20,80,0.45)");
    side.addColorStop(0.5, "rgba(10,20,80,0)");
    side.addColorStop(1, "rgba(255,120,60,0.3)");
    ctx.fillStyle = side;
    ctx.fillRect(0, 0, W, hz);
    ctx.save();                                                   // aurora ribbons
    ctx.globalCompositeOperation = "lighter";
    for (let k = 0; k < 3; k++) {
      const base = hz * (0.12 + 0.12 * k + 0.05 * rng()), amp = H * (0.03 + 0.04 * rng());
      const f1 = (2 + 3 * rng()) * Math.PI / W, f2 = (5 + 6 * rng()) * Math.PI / W, ph = rng() * 6.28;
      const hueA = [140, 170, 300][k];
      for (let x = 0; x < W; x += 1) {
        const y = base + amp * Math.sin(f1 * x + ph) + amp * 0.4 * Math.sin(f2 * x + 2 * ph);
        const len = H * (0.05 + 0.04 * Math.sin(f2 * x * 0.7 + ph) + 0.03);
        const s = 0.5 + 0.5 * Math.sin(f2 * 1.7 * x + k);
        const g = ctx.createLinearGradient(0, y, 0, y - len);
        g.addColorStop(0, `hsla(${hueA}, 90%, 60%, ${0.10 + 0.12 * s})`);
        g.addColorStop(1, `hsla(${hueA}, 90%, 60%, 0)`);
        ctx.fillStyle = g;
        ctx.fillRect(x, y - len, 1, len);
      }
    }
    ctx.restore();
    for (let k = 0; k < 220; k++) {                               // stars
      const y = rng() * hz * 0.55;
      ctx.fillStyle = `rgba(255,255,255,${(0.25 + 0.75 * rng()) * (1 - y / (hz * 0.55))})`;
      const r = rng() < 0.08 ? 1.6 : 0.8;
      ctx.beginPath(); ctx.arc(rng() * W, y, r, 0, 2 * Math.PI); ctx.fill();
    }
    const sx = W * (0.25 + 0.5 * rng()), sy = hz - H * (0.08 + 0.06 * rng()), sr = W * 0.07;
    const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr * 4);
    glow.addColorStop(0, "rgba(255,245,220,1)");
    glow.addColorStop(0.22, "rgba(255,210,150,0.85)");
    glow.addColorStop(1, "rgba(255,150,90,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, W, hz);
    for (let k = 0; k < 4; k++) {                                 // thin clouds
      const cy = hz * (0.25 + 0.5 * rng()), cx = rng() * W, cw = W * (0.3 + 0.4 * rng());
      const g = ctx.createLinearGradient(cx - cw / 2, 0, cx + cw / 2, 0);
      g.addColorStop(0, "rgba(255,190,200,0)"); g.addColorStop(0.5, "rgba(255,190,200,0.35)");
      g.addColorStop(1, "rgba(255,190,200,0)");
      ctx.fillStyle = g;
      ctx.fillRect(cx - cw / 2, cy, cw, 2 + 4 * rng());
    }
    const layers = 4;
    for (let l = 0; l < layers; l++) {                            // mountain ridges, far to near
      const f = l / (layers - 1);
      const base = hz - H * (0.22 - 0.05 * l) ;
      const amps = [0.05, 0.025, 0.012, 0.006].map((a) => a * H * (0.7 + 0.6 * rng()));
      const freqs = [1.5, 3.7, 8.3, 19].map((w) => (w * (0.8 + 0.4 * rng()) * 2 * Math.PI) / W);
      const phs = amps.map(() => rng() * 2 * Math.PI);
      ctx.beginPath();
      ctx.moveTo(0, hz);
      for (let x = 0; x <= W; x += 2) {
        let y = base + H * 0.03 * l;
        for (let k = 0; k < 4; k++) y += amps[k] * Math.sin(freqs[k] * x + phs[k]);
        ctx.lineTo(x, Math.min(y, hz));
      }
      ctx.lineTo(W, hz);
      ctx.closePath();
      const g = ctx.createLinearGradient(0, base - H * 0.08, 0, hz);
      g.addColorStop(0, `hsl(${hue + 70 - 50 * f}, ${30 + 25 * f}%, ${66 - 40 * f}%)`);
      g.addColorStop(1, `hsl(${hue + 60 - 50 * f}, ${35 + 25 * f}%, ${50 - 34 * f}%)`);
      ctx.fillStyle = g;
      ctx.fill();
    }
    for (let k = 0; k < 2; k++) {                                 // pine trees on the near shore
      const side = k === 0 ? 0 : W;
      const nT = 6 + Math.floor(5 * rng());
      for (let t = 0; t < nT; t++) {
        const x = Math.abs(side - rng() * W * 0.3), h = H * (0.08 + 0.12 * rng()), w = h * 0.35;
        ctx.fillStyle = `hsl(${hue + 10}, 35%, ${6 + 6 * rng()}%)`;
        for (let s = 0; s < 3; s++) {
          ctx.beginPath();
          const top = hz - h + s * h * 0.25;
          ctx.moveTo(x, top);
          ctx.lineTo(x - w * (0.6 + 0.2 * s), top + h * 0.45);
          ctx.lineTo(x + w * (0.6 + 0.2 * s), top + h * 0.45);
          ctx.closePath(); ctx.fill();
        }
      }
    }
    // lake: reflection of the scene above the horizon, darker, with ripples
    const above = ctx.getImageData(0, 0, W, Math.floor(hz));
    const img = ctx.getImageData(0, 0, W, H);
    const d = img.data, a = above.data, hzi = Math.floor(hz);
    for (let y = hzi; y < H; y++) {
      const depth = (y - hzi) / (H - hzi);
      const src = Math.max(0, hzi - 1 - Math.floor((y - hzi) * 1.15));
      const shift = Math.round(3 * Math.sin(y * 0.9) + 2 * Math.sin(y * 0.23 + 1));
      for (let x = 0; x < W; x++) {
        const xs = Math.min(W - 1, Math.max(0, x + shift));
        const i = (y * W + x) * 4, j = (src * W + xs) * 4;
        const dark = 0.8 - 0.35 * depth;
        d[i] = a[j] * dark + 8; d[i + 1] = a[j + 1] * dark + 14; d[i + 2] = a[j + 2] * dark + 30; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // near shore along the bottom: a dark wavy bank with grass blades and a few rocks
    const shoreY = (x) => H * 0.9 + H * 0.025 * Math.sin(x * 0.013 + seed) + H * 0.012 * Math.sin(x * 0.051 + 2 * seed);
    ctx.fillStyle = `hsl(${hue + 20}, 30%, 9%)`;
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let x = 0; x <= W; x += 2) ctx.lineTo(x, shoreY(x));
    ctx.lineTo(W, H);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = `hsl(${hue + 20}, 30%, 9%)`;
    ctx.lineWidth = 1.2;
    for (let k = 0; k < 260; k++) {
      const x = rng() * W, y = shoreY(x) + 2, h = H * (0.015 + 0.05 * rng() * rng()), lean = (rng() - 0.5) * h * 0.8;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.quadraticCurveTo(x + lean * 0.3, y - h * 0.6, x + lean, y - h); ctx.stroke();
    }
    for (let k = 0; k < 5; k++) {
      const x = rng() * W, r = H * (0.012 + 0.02 * rng());
      ctx.fillStyle = `hsl(${hue + 20}, 20%, ${12 + 8 * rng()}%)`;
      ctx.beginPath(); ctx.ellipse(x, shoreY(x) + r * 0.3, r * 1.6, r, 0, Math.PI, 2 * Math.PI); ctx.fill();
    }
    const img2 = ctx.getImageData(0, 0, W, H), d2 = img2.data;   // fine grain
    for (let i = 0; i < d2.length; i += 4) {
      const g = (rng() - 0.5) * 6;
      d2[i] += g; d2[i + 1] += g; d2[i + 2] += g;
    }
    ctx.putImageData(img2, 0, 0);
  }

  // ---------- puzzle model ----------
  function buildPuzzle(canvas, n) {
    const W = canvas.width, ts = Math.floor(W / n), N = n * n;
    const px = canvas.getContext("2d").getImageData(0, 0, ts * n, ts * n);
    const P = (x, y, c) => px.data[(y * px.width + x) * 4 + c];
    const edges = (t) => {
      const ox = (t % n) * ts, oy = Math.floor(t / n) * ts;
      const col = (x) => { const v = new Float32Array(ts * 3); for (let y = 0; y < ts; y++) for (let c = 0; c < 3; c++) v[y * 3 + c] = P(ox + x, oy + y, c); return v; };
      const row = (y) => { const v = new Float32Array(ts * 3); for (let x = 0; x < ts; x++) for (let c = 0; c < 3; c++) v[x * 3 + c] = P(ox + x, oy + y, c); return v; };
      return { L1: col(0), L2: col(1), R1: col(ts - 1), R2: col(ts - 2), T1: row(0), T2: row(1), B1: row(ts - 1), B2: row(ts - 2) };
    };
    const E = [...Array(N).keys()].map(edges);
    const pred = (a1, a2, b1, b2) => {      // a1 last line of the first tile, a2 the one before; b1, b2 of the second
      let s = 0;
      for (let k = 0; k < a1.length; k++) {
        const u = 2 * a1[k] - a2[k] - b1[k], v = 2 * b1[k] - b2[k] - a1[k];
        s += u * u + v * v;
      }
      return Math.sqrt(s / a1.length);
    };
    const DH = [...Array(N)].map(() => new Float64Array(N)), DV = [...Array(N)].map(() => new Float64Array(N));
    const all = [];
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) if (a !== b) {
      DH[a][b] = pred(E[a].R1, E[a].R2, E[b].L1, E[b].L2);
      DV[a][b] = pred(E[a].B1, E[a].B2, E[b].T1, E[b].T2);
      all.push(DH[a][b], DV[a][b]);
    }
    all.sort((x, y) => x - y);
    const scale = all[Math.floor(all.length / 2)] || 1;
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) { DH[a][b] /= scale; DV[a][b] /= scale; }
    // slot adjacency: edges (p, q, horizontal?) and the edges of each slot
    const links = [], slotLinks = [...Array(N)].map(() => []);
    for (let p = 0; p < N; p++) {
      const r = Math.floor(p / n), c = p % n;
      if (c < n - 1) { slotLinks[p].push(links.length); slotLinks[p + 1].push(links.length); links.push([p, p + 1, 0]); }
      if (r < n - 1) { slotLinks[p].push(links.length); slotLinks[p + n].push(links.length); links.push([p, p + n, 1]); }
    }
    const color = [...Array(N).keys()].map((t) => {
      const ox = (t % n) * ts, oy = Math.floor(t / n) * ts;
      let r = 0, g = 0, b = 0, m = 0;
      for (let y = 0; y < ts; y += 3) for (let x = 0; x < ts; x += 3) { r += P(ox + x, oy + y, 0); g += P(ox + x, oy + y, 1); b += P(ox + x, oy + y, 2); m++; }
      return `rgb(${r / m | 0},${g / m | 0},${b / m | 0})`;
    });
    // Moves, each a fixed permutation of the slots: the tile in slot src[k] moves to slot dst[k].
    //   block exchange: exchange two non-overlapping blocks of h x w slots (1 x 1: exchange of two slots)
    //   block shift:    shift a block of slots by k slots along its rows or columns, the displaced columns or
    //                   rows wrapping to the other side of the block (both directions are in the library)
    const moves = [];
    const slot = (r, c) => r * n + c;
    const maxB = Math.max(1, Math.floor(n / 2));
    for (let h = 1; h <= maxB; h++) for (let w = 1; w <= maxB; w++) {
      const pos = [];
      for (let r = 0; r + h <= n; r++) for (let c = 0; c + w <= n; c++) pos.push([r, c]);
      for (let i = 0; i < pos.length; i++) for (let j = i + 1; j < pos.length; j++) {
        const [ra, ca] = pos[i], [rb, cb] = pos[j];
        if (ra < rb + h && rb < ra + h && ca < cb + w && cb < ca + w) continue;   // overlapping
        const src = [], dst = [];
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          src.push(slot(ra + y, ca + x), slot(rb + y, cb + x));
          dst.push(slot(rb + y, cb + x), slot(ra + y, ca + x));
        }
        moves.push({ kind: 0, src, dst });
      }
    }
    for (let r1 = 0; r1 < n; r1++) for (let r2 = r1; r2 < n; r2++) for (let c1 = 0; c1 < n; c1++) for (let c2 = c1; c2 < n; c2++) {
      const h = r2 - r1 + 1, w = c2 - c1 + 1;
      for (let k = 1; k < w; k++) {
        if (h === 1 && w === 2) break;                                      // equals an exchange of two slots
        const src = [], dst = [];
        for (let r = r1; r <= r2; r++) for (let x = 0; x < w; x++) { src.push(slot(r, c1 + x)); dst.push(slot(r, c1 + (x + k) % w)); }
        moves.push({ kind: 1, src, dst });
      }
      for (let k = 1; k < h; k++) {
        if (w === 1 && h === 2) break;
        const src = [], dst = [];
        for (let c = c1; c <= c2; c++) for (let y = 0; y < h; y++) { src.push(slot(r1 + y, c)); dst.push(slot(r1 + (y + k) % h, c)); }
        moves.push({ kind: 1, src, dst });
      }
    }
    for (const mv of moves) mv.slots = mv.src;
    const nSwap = (N * (N - 1)) / 2;                       // the first moves are the exchanges of two slots
    return { n, N, ts, DH, DV, links, slotLinks, color, moves, nSwap };
  }

  function linkCost(M, perm, e) {
    const [p, q, v] = M.links[e];
    return v ? M.DV[perm[p]][perm[q]] : M.DH[perm[p]][perm[q]];
  }
  function energy(M, perm) {
    let s = 0;
    for (let e = 0; e < M.links.length; e++) s += linkCost(M, perm, e);
    return s;
  }
  const buf = [];
  function apply(perm, mv) {                // the tile in slot src[k] moves to slot dst[k]
    const s = mv.src, d = mv.dst;
    for (let k = 0; k < s.length; k++) buf[k] = perm[s[k]];
    for (let k = 0; k < s.length; k++) perm[d[k]] = buf[k];
  }
  function undo(perm, mv) {
    const s = mv.src, d = mv.dst;
    for (let k = 0; k < s.length; k++) buf[k] = perm[d[k]];
    for (let k = 0; k < s.length; k++) perm[s[k]] = buf[k];
  }
  const mark = { stamp: 0, seen: null };
  function deltaE(M, perm, mv) {
    if (!mark.seen || mark.seen.length < M.links.length) mark.seen = new Int32Array(M.links.length);
    const stamp = ++mark.stamp, list = [];
    for (const p of mv.slots) for (const e of M.slotLinks[p]) if (mark.seen[e] !== stamp) { mark.seen[e] = stamp; list.push(e); }
    let before = 0, after = 0;
    for (const e of list) before += linkCost(M, perm, e);
    apply(perm, mv);
    for (const e of list) after += linkCost(M, perm, e);
    undo(perm, mv);
    return after - before;
  }

  function solveTau(uphill, p) {
    if (!uphill.length) return 1;
    const acc = (tau) => uphill.reduce((s, d) => s + Math.exp(-d / tau), 0) / uphill.length;
    let lo = 1e-6, hi = 1e6;
    for (let it = 0; it < 100; it++) { const mid = Math.sqrt(lo * hi); if (acc(mid) < p) lo = mid; else hi = mid; }
    return Math.sqrt(lo * hi);
  }
  // Proposal: an exchange of two slots with probability SHARE, otherwise a block move, uniform within each class
  // (a fixed backbone share, as in nimbus.problems.matching.shortcut_library).
  const SHARE = 0.5, KAPPA = 64;
  function propose(M, rng) {
    return rng() < SHARE ? M.moves[Math.floor(rng() * M.nSwap)]
                         : M.moves[M.nSwap + Math.floor(rng() * (M.moves.length - M.nSwap))];
  }
  // Acceptance-rule calibration (nimbus.annealing.protocol): tau_start gives a mean uphill acceptance of 0.8 at
  // random states; tau_end gives 0.01 after a descent of 8192 attempts from tau_start to 0.02 tau_start, and
  // tau_end <= tau_start / kappa. kappa = 64 was selected on separate pictures (seeds 101-104), as the paper selects
  // its settings on training instances.
  function calibrate(M, rng) {
    const ids = [...Array(M.N).keys()], upStart = [], upEnd = [];
    for (let r = 0; r < 32; r++) {
      const perm = shuffled(ids, rng);
      for (let q = 0; q < 32; q++) { const d = deltaE(M, perm, propose(M, rng)); if (d > 0) upStart.push(d); }
    }
    const tauStart = solveTau(upStart, 0.8);
    for (let r = 0; r < 4; r++) {
      const perm = shuffled(ids, rng), tau = fnSchedule(tauStart, 0.02 * tauStart, 0.1, 8192);
      for (let q = 0; q < 8192; q++) {
        const mv = propose(M, rng), d = deltaE(M, perm, mv);
        if (d <= -tau(q) * Math.log(1 - rng())) apply(perm, mv);
      }
      for (let q = 0; q < 500; q++) { const d = deltaE(M, perm, propose(M, rng)); if (d > 0) upEnd.push(d); }
    }
    return { tauStart, tauEnd: Math.min(solveTau(upEnd, 0.01), tauStart / KAPPA) };
  }
  function fnSchedule(tauStart, tauEnd, eta, B) {
    const tau0 = tauEnd * Math.log1p(eta), qStart = Math.expm1(tau0 / tauStart);
    return (t) => tau0 / Math.log1p(qStart + (eta - qStart) * (B === 1 ? 0 : t / (B - 1)));
  }

  window.NimbusJigsaw = { mulberry32, shuffled, paint, buildPuzzle, energy, deltaE, apply, calibrate, fnSchedule, propose };

  // ---------- page ----------
  const root = document.getElementById("nimbus-jigsaw-demo");
  if (!root) return;
  root.innerHTML = `
    <div class="cj-stage">
      <canvas class="cj-board" width="480" height="480" aria-label="Puzzle board"></canvas>
      <div class="cj-side">
        <canvas class="cj-matrix" width="240" height="240" aria-label="Assignment spins"></canvas>
        <div class="cj-cap">spins x<sub>t,p</sub> (tile t in slot p): one per row and column</div>
        <canvas class="cj-target" width="120" height="120" aria-label="The picture"></canvas>
        <div class="cj-cap">the picture</div>
      </div>
    </div>
    <div class="cj-stats">
      <span><b class="cj-it">0</b> iterations</span>
      <span>temperature <b class="cj-tau">-</b></span>
      <span>energy <b class="cj-E">-</b></span>
      <span>tiles in place <b class="cj-ok">-</b></span>
      <span class="cj-valid">every state uses each tile exactly once</span>
    </div>
    <div class="cj-controls">
      <button class="md-button md-button--primary cj-run">Run</button>
      <button class="md-button cj-shuffle">Shuffle</button>
      <button class="md-button cj-new">New picture</button>
      <label>tiles <select class="cj-n"><option value="4">4 x 4</option><option value="6" selected>6 x 6</option><option value="8">8 x 8</option></select></label>
      <label>speed <input class="cj-speed" type="range" min="0" max="4" step="1" value="2"></label>
    </div>`;
  const q = (s) => root.querySelector(s);
  const board = q(".cj-board"), bctx = board.getContext("2d"), mctx = q(".cj-matrix").getContext("2d");
  const pic = document.createElement("canvas");
  pic.width = pic.height = 480;
  const DURATION = [40, 20, 10, 5, 2.5];                  // seconds per run, by speed setting
  let S = null, running = false, picSeed = 7;

  function newPicture() {
    paint(pic, picSeed);
    const t = q(".cj-target").getContext("2d");
    t.drawImage(pic, 0, 0, 120, 120);
    setup();
  }
  function setup() {
    running = false;
    const n = +q(".cj-n").value;
    const M = buildPuzzle(pic, n);
    S = { M, rng: mulberry32(picSeed * 1000 + n) };
    window.NimbusJigsaw.state = S;
    shuffle();
  }
  function shuffle() {
    running = false;
    const s = S, M = s.M;
    s.perm = shuffled([...Array(M.N).keys()], s.rng);
    const c = calibrate(M, s.rng);
    s.B = (M.n >= 8 ? 20000 : 5000) * M.N;
    s.tau = fnSchedule(c.tauStart, c.tauEnd, 0.1, s.B);
    s.t = 0;
    s.E = energy(M, s.perm);
    draw();
  }
  function step(k) {
    const s = S, M = s.M;
    for (let r = 0; r < k && s.t < s.B; r++, s.t++) {
      const mv = propose(M, s.rng);
      const d = deltaE(M, s.perm, mv);
      if (d <= -s.tau(s.t) * Math.log(1 - s.rng())) { apply(s.perm, mv); s.E += d; }
    }
  }
  function draw() {
    const s = S, M = s.M, n = M.n, ts = M.ts, cell = board.width / n;
    bctx.fillStyle = getComputedStyle(root).getPropertyValue("--md-default-bg-color") || "#fff";
    bctx.fillRect(0, 0, board.width, board.height);
    let ok = 0;
    for (let p = 0; p < M.N; p++) {
      const t = s.perm[p];
      if (t === p) ok++;
      bctx.drawImage(pic, (t % n) * ts, Math.floor(t / n) * ts, ts, ts, (p % n) * cell + 0.5, Math.floor(p / n) * cell + 0.5, cell - 1, cell - 1);
    }
    const mc = mctx.canvas.width / M.N;
    mctx.clearRect(0, 0, mctx.canvas.width, mctx.canvas.height);
    mctx.fillStyle = "rgba(128,128,128,0.12)";
    mctx.fillRect(0, 0, mctx.canvas.width, mctx.canvas.height);
    for (let p = 0; p < M.N; p++) {
      const t = s.perm[p];
      mctx.fillStyle = M.color[t];
      mctx.fillRect(p * mc, t * mc, Math.max(mc, 1.5), Math.max(mc, 1.5));
    }
    q(".cj-it").textContent = s.t.toLocaleString();
    q(".cj-tau").textContent = s.tau(Math.min(s.t, s.B - 1)).toPrecision(2);
    q(".cj-E").textContent = s.E.toFixed(1);
    q(".cj-ok").textContent = `${ok} / ${M.N}`;
    q(".cj-run").textContent = s.t >= s.B ? "Run again" : running ? "Pause" : "Run";
  }
  let last = 0;
  function loop(now) {
    if (!running) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    step(Math.max(1, Math.round((S.B * dt) / DURATION[+q(".cj-speed").value])));
    draw();
    if (S.t >= S.B) { running = false; draw(); return; }
    requestAnimationFrame(loop);
  }
  q(".cj-run").addEventListener("click", () => {
    if (S.t >= S.B) shuffle();
    running = !running;
    last = 0;
    draw();
    if (running) requestAnimationFrame(loop);
  });
  q(".cj-shuffle").addEventListener("click", shuffle);
  q(".cj-new").addEventListener("click", () => { picSeed++; newPicture(); });
  q(".cj-n").addEventListener("change", setup);
  newPicture();

  // Start once, the first time the board is in view (not when the viewer asks for reduced motion).
  const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!still && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        if (!running && S.t === 0) q(".cj-run").click();
      }
    }, { threshold: 0.5 });
    io.observe(board);
  }
})();
