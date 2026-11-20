/* Live graph-matching demo of NIMBUS, run in the browser.
 *
 * The same steps as the package, for one replica:
 *   instance   nimbus.problems.formats.matching_tensor.similarity_deformation_pair / point_pair_tensor
 *              (triangle angle-sine features, K = 3n nearest ordered target triangles, value exp(-d^2 / gamma))
 *   moves      complete library of row exchanges (nimbus.problems.matching, library "complete")
 *   acceptance accept iff Delta E <= -tau ln u                (nimbus.annealing.policy, metropolis_accept)
 *   schedule   Fowler-Nordheim between tau_start and tau_end, eta = 0.1   (nimbus.annealing.schedule)
 *   calibration acceptance rule: mean uphill acceptance 0.8 at random states and 0.01 after a descent,
 *              tau_start / tau_end >= 2                         (nimbus.annealing.protocol)
 * Random numbers come from a small seeded generator, so runs differ from the package's random streams.
 */
(function () {
  "use strict";

  // ---------- random numbers ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  function shuffled(arr, rng) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ---------- instance ----------
  function makePoints(n, sigma, rng) {
    const P1 = [];
    for (let i = 0; i < n; i++) P1.push([rng(), rng()]);
    const th = rng() * Math.PI / 4, sc = 0.8 + 0.4 * rng();
    const c = Math.cos(th), s = Math.sin(th);
    const P2raw = P1.map(([x, y]) => [sc * (c * x - s * y) + sigma * gauss(rng), sc * (s * x + c * y) + sigma * gauss(rng)]);
    const order = shuffled([...Array(n).keys()], rng);          // P2[k] = P2raw[order[k]]
    const P2 = order.map((k) => P2raw[k]);
    const truth = new Array(n);
    order.forEach((src, k) => { truth[src] = k; });
    return { P1, P2, truth };
  }

  function angle(u, v, w) {
    const x0 = v[0] - u[0], x1 = v[1] - u[1], y0 = w[0] - u[0], y1 = w[1] - u[1];
    let cs = (x0 * y0 + x1 * y1) / (Math.hypot(x0, x1) * Math.hypot(y0, y1) + 1e-12);
    cs = Math.max(-1, Math.min(1, cs));
    return Math.acos(cs);
  }
  function features(P, a, b, c) {
    return [Math.sin(angle(P[a], P[b], P[c])), Math.sin(angle(P[b], P[c], P[a])), Math.sin(angle(P[c], P[a], P[b]))];
  }

  // Triangle affinity tensor, stored per source triangle as a map from the target triple to its value.
  function buildTensor(P1, P2, rng) {
    const n = P1.length;
    let t1 = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) t1.push([i, j, k]);
    if (t1.length > n * n) t1 = shuffled(t1, rng).slice(0, n * n);
    const t2 = [];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) for (let c = 0; c < n; c++)
      if (a !== b && b !== c && a !== c) t2.push([a, b, c]);
    const f2 = t2.map(([a, b, c]) => features(P2, a, b, c));
    const K = Math.min(3 * n, t2.length);
    const near = [];
    let dsum = 0;
    for (const [i, j, k] of t1) {
      const f = features(P1, i, j, k);
      const d = f2.map((g, idx) => [Math.hypot(f[0] - g[0], f[1] - g[1], f[2] - g[2]), idx]);
      d.sort((p, q) => p[0] - q[0]);
      const best = d.slice(0, K);
      best.forEach(([dist]) => { dsum += dist; });
      near.push(best);
    }
    const gamma = Math.pow(dsum / (t1.length * K), 2) + 1e-12;
    const tris = t1.map((tri, m) => {
      const map = new Map();
      for (const [dist, idx] of near[m]) {
        const [a, b, c] = t2[idx];
        const key = (a * n + b) * n + c;
        map.set(key, (map.get(key) || 0) + Math.exp(-dist * dist / gamma));
      }
      return { tri, map };
    });
    const rowTris = [...Array(n)].map(() => []);
    tris.forEach((t, m) => t.tri.forEach((r) => rowTris[r].push(m)));
    return { n, tris, rowTris };
  }

  function contribution(T, m, perm) {
    const [i, j, k] = T.tris[m].tri, n = T.n;
    return T.tris[m].map.get((perm[i] * n + perm[j]) * n + perm[k]) || 0;
  }
  function objective(T, perm) {
    let F = 0;
    for (let m = 0; m < T.tris.length; m++) F += contribution(T, m, perm);
    return F;
  }
  // Change of F when rows i and j exchange their targets (only triangles that contain i or j change).
  function deltaF(T, perm, i, j) {
    const seen = new Set();
    let before = 0, after = 0;
    for (const m of T.rowTris[i].concat(T.rowTris[j])) {
      if (seen.has(m)) continue;
      seen.add(m);
      before += contribution(T, m, perm);
    }
    [perm[i], perm[j]] = [perm[j], perm[i]];
    for (const m of seen) after += contribution(T, m, perm);
    [perm[i], perm[j]] = [perm[j], perm[i]];
    return after - before;
  }

  // ---------- calibration and schedule ----------
  function solveTau(uphill, p) {
    // mean(exp(-dE / tau)) = p, by bisection on log tau
    if (!uphill.length) return 1;
    const acc = (tau) => uphill.reduce((s, d) => s + Math.exp(-d / tau), 0) / uphill.length;
    let lo = 1e-6, hi = 1e6;
    for (let it = 0; it < 100; it++) {
      const mid = Math.sqrt(lo * hi);
      if (acc(mid) < p) lo = mid; else hi = mid;
    }
    return Math.sqrt(lo * hi);
  }
  function calibrate(T, moves, rng) {
    const n = T.n, upStart = [], upEnd = [];
    for (let r = 0; r < 64; r++) {
      const perm = shuffled([...Array(n).keys()], rng);
      for (let q = 0; q < 16; q++) {
        const [i, j] = moves[Math.floor(rng() * moves.length)];
        const dE = -deltaF(T, perm, i, j);
        if (dE > 0) upStart.push(dE);
      }
    }
    for (let r = 0; r < 24; r++) {                               // descent, then the uphill moves from there
      const perm = shuffled([...Array(n).keys()], rng);
      for (let q = 0; q < 40 * moves.length; q++) {
        const [i, j] = moves[Math.floor(rng() * moves.length)];
        if (-deltaF(T, perm, i, j) <= 0) [perm[i], perm[j]] = [perm[j], perm[i]];
      }
      for (const [i, j] of moves) {
        const dE = -deltaF(T, perm, i, j);
        if (dE > 0) upEnd.push(dE);
      }
    }
    const tauStart = solveTau(upStart, 0.8);
    const tauEnd = Math.min(solveTau(upEnd, 0.01), tauStart / 2);
    return { tauStart, tauEnd };
  }
  function fnSchedule(tauStart, tauEnd, eta, B) {
    const tau0 = tauEnd * Math.log1p(eta), qStart = Math.expm1(tau0 / tauStart);
    return (t) => tau0 / Math.log1p(qStart + (eta - qStart) * (B === 1 ? 0 : t / (B - 1)));
  }

  // expose the model for tests
  window.NimbusDemo = { mulberry32, makePoints, buildTensor, objective, deltaF, calibrate, fnSchedule };

  // ---------- page ----------
  const root = document.getElementById("nimbus-matching-demo");
  if (!root) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 640, H = 300, PAD = 28, GAP = 40, PW = (W - GAP) / 2;
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  root.innerHTML = `
    <div class="cd-stage">
      <svg class="cd-points" viewBox="0 0 ${W} ${H}" role="img" aria-label="Two point sets and the current assignment"></svg>
      <div class="cd-side">
        <svg class="cd-matrix" viewBox="0 0 200 200" role="img" aria-label="Assignment spins"></svg>
        <div class="cd-matrix-label">spins s<sub>i,a</sub>: one per row and column</div>
      </div>
    </div>
    <div class="cd-stats">
      <span><b class="cd-it">0</b> iterations</span>
      <span>temperature <b class="cd-tau">-</b></span>
      <span>objective <b class="cd-F">-</b></span>
      <span>correct matches <b class="cd-acc">-</b></span>
      <span class="cd-ok">every state is a valid matching</span>
    </div>
    <div class="cd-controls">
      <button class="md-button md-button--primary cd-run">Run</button>
      <button class="md-button cd-new">New problem</button>
      <label>points <select class="cd-n"><option>6</option><option>8</option><option selected>10</option><option>12</option><option>15</option></select></label>
      <label>noise <select class="cd-sigma"><option value="0.02" selected>low</option><option value="0.05">medium</option><option value="0.1">high</option></select></label>
      <label>speed <input class="cd-speed" type="range" min="0" max="4" step="1" value="3"></label>
    </div>`;

  const svg = root.querySelector(".cd-points"), mat = root.querySelector(".cd-matrix");
  const q = (s) => root.querySelector(s);
  const SPEEDS = [5, 20, 80, 320, 1280];

  let state = null, running = false, seed = 20261113;

  function panelMap(P, x0) {
    const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
    const mx = Math.min(...xs), Mx = Math.max(...xs), my = Math.min(...ys), My = Math.max(...ys);
    const s = Math.min((PW - 2 * PAD) / (Mx - mx || 1), (H - 2 * PAD) / (My - my || 1));
    const ox = x0 + (PW - s * (Mx - mx)) / 2, oy = (H - s * (My - my)) / 2;
    return {
      to: (p) => [ox + s * (p[0] - mx), H - (oy + s * (p[1] - my))],
      from: (x, y) => [mx + (x - ox) / s, my + (H - y - oy) / s],
    };
  }

  function newProblem() {
    const n = +q(".cd-n").value, sigma = +q(".cd-sigma").value;
    const rng = mulberry32(seed++);
    const { P1, P2, truth } = makePoints(n, sigma, rng);
    const moves = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) moves.push([i, j]);
    state = { n, P1, P2, truth, moves, rng, map1: panelMap(P1, 0), map2: panelMap(P2, PW + GAP) };
    window.NimbusDemo.state = state;
    rebuild(true);
  }

  function rebuild(randomStart) {
    const s = state;
    s.T = buildTensor(s.P1, s.P2, s.rng);
    if (randomStart || !s.perm) s.perm = shuffled([...Array(s.n).keys()], s.rng);
    const { tauStart, tauEnd } = calibrate(s.T, s.moves, s.rng);
    s.B = 1000 * s.n * s.n;
    s.tau = fnSchedule(tauStart, tauEnd, 0.1, s.B);
    s.t = 0;
    s.F = objective(s.T, s.perm);
    s.last = null;
    draw();
  }

  function step(k) {
    const s = state;
    for (let r = 0; r < k && s.t < s.B; r++, s.t++) {
      const [i, j] = s.moves[Math.floor(s.rng() * s.moves.length)];
      const dE = -deltaF(s.T, s.perm, i, j);
      const u = 1 - s.rng();
      if (dE <= -s.tau(s.t) * Math.log(u)) {
        [s.perm[i], s.perm[j]] = [s.perm[j], s.perm[i]];
        s.F -= dE;
        s.last = [i, j];
      }
    }
  }

  function hue(i, n) { return `hsl(${Math.round((360 * i) / n)}, 70%, 52%)`; }

  function draw() {
    const s = state, n = s.n;
    svg.textContent = "";
    el("text", { x: PW / 2, y: 16, class: "cd-title" }, svg).textContent = "image 1";
    el("text", { x: PW + GAP + PW / 2, y: 16, class: "cd-title" }, svg).textContent = "image 2";
    el("rect", { x: 1, y: 22, width: PW - 2, height: H - 23, rx: 6, class: "cd-panel" }, svg);
    el("rect", { x: PW + GAP + 1, y: 22, width: PW - 2, height: H - 23, rx: 6, class: "cd-panel" }, svg);
    for (let i = 0; i < n; i++) {
      const [x1, y1] = s.map1.to(s.P1[i]), [x2, y2] = s.map2.to(s.P2[s.perm[i]]);
      const ok = s.perm[i] === s.truth[i];
      const hot = s.last && (s.last[0] === i || s.last[1] === i);
      el("line", { x1, y1, x2, y2, class: "cd-line" + (ok ? " ok" : "") + (hot ? " hot" : ""), style: ok ? `stroke:${hue(i, n)}` : "" }, svg);
    }
    const drawPts = (P, map, which) => P.forEach((p, k) => {
      const [x, y] = map.to(p);
      const src = which === 1 ? k : s.truth.indexOf(k);
      const c = el("circle", { cx: x, cy: y, r: 7, fill: hue(src, n), class: "cd-pt", "data-set": which, "data-k": k }, svg);
      c.addEventListener("pointerdown", startDrag);
    });
    drawPts(s.P1, s.map1, 1);
    drawPts(s.P2, s.map2, 2);

    mat.textContent = "";
    const cs = 200 / n;
    for (let i = 0; i < n; i++) for (let a = 0; a < n; a++) {
      const on = s.perm[i] === a;
      el("rect", { x: a * cs + 1, y: i * cs + 1, width: cs - 2, height: cs - 2, rx: 2,
                   class: "cd-cell" + (on ? " on" : ""), style: on ? `fill:${hue(i, n)}` : "" }, mat);
    }
    q(".cd-it").textContent = s.t.toLocaleString();
    q(".cd-tau").textContent = s.tau(Math.min(s.t, s.B - 1)).toPrecision(2);
    q(".cd-F").textContent = s.F.toFixed(2);
    q(".cd-acc").textContent = `${s.perm.filter((a, i) => a === s.truth[i]).length} / ${n}`;
    const done = s.t >= s.B;
    q(".cd-run").textContent = done ? "Run again" : running ? "Pause" : "Run";
  }

  let drag = null;
  function startDrag(ev) {
    const c = ev.target, set = +c.dataset.set, k = +c.dataset.k;
    drag = { set, k };
    svg.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  }
  svg.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const x0 = drag.set === 1 ? 0 : PW + GAP;
    const x = Math.max(x0 + 8, Math.min(x0 + PW - 8, p.x)), y = Math.max(30, Math.min(H - 8, p.y));
    const P = drag.set === 1 ? state.P1 : state.P2, map = drag.set === 1 ? state.map1 : state.map2;
    P[drag.k] = map.from(x, y);
    draw();
  });
  function endDrag() {
    if (!drag) return;
    drag = null;
    rebuild(false);                       // new tensor for the moved point; continue from the current matching
    if (!running) toggle();
  }
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  function loop() {
    if (!running) return;
    step(SPEEDS[+q(".cd-speed").value]);
    draw();
    if (state.t >= state.B) { running = false; draw(); return; }
    requestAnimationFrame(loop);
  }
  function toggle() {
    if (state.t >= state.B) rebuild(true);
    running = !running;
    draw();
    if (running) requestAnimationFrame(loop);
  }
  q(".cd-run").addEventListener("click", toggle);
  q(".cd-new").addEventListener("click", () => { running = false; newProblem(); });
  q(".cd-n").addEventListener("change", () => { running = false; newProblem(); });
  q(".cd-sigma").addEventListener("change", () => { running = false; newProblem(); });

  newProblem();
})();
