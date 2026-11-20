// Rubik's cube demo. The constraint (a valid cube) is kept by the moves: every move is a permutation of the 54
// stickers that the cube itself can make. Backbone: the 18 face turns. Shortcuts: short commutators that change only
// two to four pieces (corner and edge 3-cycles, pairs of twisted corners or flipped edges, and a swap of two corners
// and two edges), each conjugated by setup turns so that it can act on any pieces. Energy: the number of stickers whose
// colour differs from the centre of their face. Acceptance: the Metropolis rule; temperature: the Fowler-Nordheim
// schedule; an unsolved run restarts the schedule from the state it reached.
(function () {
  "use strict";
  const root = document.getElementById("nimbus-rubik-demo");
  if (!root) return;

  // ---------- the cube as a permutation group on 54 stickers ----------
  const AX = { U: [1, 1], D: [1, -1], R: [0, 1], L: [0, -1], F: [2, 1], B: [2, -1] };
  const ST = [];
  for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
    const p = [x, y, z];
    for (let a = 0; a < 3; a++) if (p[a] !== 0) { const n = [0, 0, 0]; n[a] = p[a]; ST.push({ p, n }); }
  }
  const keyOf = (p, n) => p.join(",") + "|" + n.join(",");
  const IDX = new Map(ST.map((s, i) => [keyOf(s.p, s.n), i]));
  const FACES = "UDRLFB";
  const faceOf = (n) => FACES[[1, 1, 0, 0, 2, 2].findIndex((a, k) => n[a] === (k % 2 ? -1 : 1))];
  const HOME = ST.map((s) => FACES.indexOf(faceOf(s.n)));
  function rot(v, axis, sgn) {
    const [x, y, z] = v;
    let r = axis === 0 ? [x, z, -y] : axis === 1 ? [-z, y, x] : [y, -x, z];
    if (sgn < 0) r = rot(rot(r, axis, 1), axis, 1);
    return r;
  }
  function layerPerm(axis, layer, like) {
    const perm = new Uint8Array(54).map((_, i) => i);
    ST.forEach((s, i) => { if (s.p[axis] === layer) perm[IDX.get(keyOf(rot(s.p, axis, like), rot(s.n, axis, like)))] = i; });
    return perm;
  }
  const mul = (a, b) => { const c = new Uint8Array(54); for (let j = 0; j < 54; j++) c[j] = a[b[j]]; return c; }; // apply a, then b
  const MV = {};
  for (const f in AX) { const [a, s] = AX[f]; const q = layerPerm(a, s, s); MV[f] = q; MV[f + "2"] = mul(q, q); MV[f + "'"] = mul(mul(q, q), q); }
  for (const [m, a, like] of [["M", 0, -1], ["E", 1, -1], ["S", 2, 1]]) { const q = layerPerm(a, 0, like); MV[m] = q; MV[m + "2"] = mul(q, q); MV[m + "'"] = mul(mul(q, q), q); }
  const TURNS = ["U", "U'", "U2", "D", "D'", "D2", "R", "R'", "R2", "L", "L'", "L2", "F", "F'", "F2", "B", "B'", "B2"];
  const ID = new Uint8Array(54).map((_, i) => i);
  const seqPerm = (seq) => seq.reduce((p, m) => mul(p, MV[m]), ID);
  const invMove = (m) => m[0] + (m.endsWith("'") ? "" : m.endsWith("2") ? "2" : "'");
  const inverse = (p) => { const q = new Uint8Array(54); for (let j = 0; j < 54; j++) q[p[j]] = j; return q; };

  // ---------- the move library ----------
  const BASES = {
    corner3: ["R' D' R U R' D R U'", 4], edge3: ["R U' R U R U R U' R' U' R2", 4],
    twist: ["R' D' R D R' D' R D U D' R' D R D' R' D R U'", 3], flip: ["M' U M' U M' U2 M U M U M U2", 3],
    swap: ["R U R' U' R' F R2 U' R' U' R U R' F'", 2],
  };
  const KINDS = ["turn", "corner3", "edge3", "twist", "flip", "swap"];
  let WEIGHT = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1];
  let LIB = null;
  function buildLibrary() {
    const lib = { turn: TURNS.map((t) => MV[t]) };
    // setup permutations of up to 4 turns (consecutive turns on different faces), with their inverses
    let layer = [{ p: ID, last: "" }], setups = [ID];
    const setupsByLen = [[ID]];
    for (let L = 1; L <= 4; L++) {
      const next = [];
      for (const s of layer) for (const t of TURNS) if (t[0] !== s.last) next.push({ p: mul(s.p, MV[t]), last: t[0] });
      layer = next; setupsByLen.push(next.map((s) => s.p));
    }
    for (const kind in BASES) {
      const [seq, maxL] = BASES[kind];
      const X = seqPerm(seq.split(" ")), Xi = inverse(X);
      const seen = new Set(), out = [];
      for (let L = 0; L <= maxL; L++) for (const S of setupsByLen[L]) {
        const Si = inverse(S);
        for (const Y of [X, Xi]) {
          const P = mul(mul(S, Y), Si);
          let h = 2166136261, g = 0;
          for (let j = 0; j < 54; j++) { h = Math.imul(h ^ P[j], 16777619); g = (g * 31 + P[j]) | 0; }
          const k = h + ":" + g;
          if (!seen.has(k)) { seen.add(k); out.push(P); }
        }
      }
      lib[kind] = out;
    }
    // moved positions of every permutation, for the incremental energy
    const moved = (P) => { const m = []; for (let j = 0; j < 54; j++) if (P[j] !== j) m.push(j); return Uint8Array.from(m); };
    LIB = KINDS.map((k) => lib[k].map((P) => ({ P, m: moved(P) })));
  }

  // ---------- annealing ----------
  let seed = 1;
  const rnd = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const TAU0 = 3, TAU1 = 0.25, ETA = 0.1, B = 300000;
  const tau0 = TAU1 * Math.log(1 + ETA), qs = Math.exp(tau0 / TAU0) - 1;
  const tauAt = (n) => tau0 / Math.log(1 + qs + (ETA - qs) * (n / (B - 1)));
  const S = { st: new Uint8Array(ID), E: 0, n: 0, total: 0, restarts: 0, acc: new Array(6).fill(0), solved: false, running: false };
  const wrongAt = (st, j) => (HOME[st[j]] !== HOME[j] ? 1 : 0);
  const energy = (st) => { let e = 0; for (let j = 0; j < 54; j++) e += wrongAt(st, j); return e; };
  const tmp = new Uint8Array(54);
  function stepOnce() {
    const tau = tauAt(S.n);
    let r = rnd(), k = 0;
    while (r > WEIGHT[k] && k < 5) { r -= WEIGHT[k]; k++; }
    const grp = LIB[k], mv = grp[Math.floor(rnd() * grp.length)];
    const st = S.st, P = mv.P, m = mv.m;
    let d = 0;
    for (let i = 0; i < m.length; i++) { const j = m[i]; d += (HOME[st[P[j]]] !== HOME[j] ? 1 : 0) - (HOME[st[j]] !== HOME[j] ? 1 : 0); }
    if (d <= 0 || rnd() < Math.exp(-d / tau)) {
      for (let i = 0; i < m.length; i++) tmp[m[i]] = st[P[m[i]]];
      for (let i = 0; i < m.length; i++) st[m[i]] = tmp[m[i]];
      S.E += d; S.acc[k]++;
    }
    S.n++; S.total++;
    if (S.E === 0) { S.solved = true; return; }
    if (S.n >= B) { S.n = 0; S.restarts++; }
  }
  function scramble(len) {
    S.st = new Uint8Array(ID);
    for (let i = 0; i < len; i++) S.st = mul(S.st, MV[TURNS[Math.floor(rnd() * 18)]]);
    S.E = energy(S.st); S.n = 0; S.total = 0; S.restarts = 0; S.acc.fill(0); S.solved = S.E === 0;
  }

  // ---------- drawing: a rotating 3-D cube and the unfolded net ----------
  const COLORS = { U: "#f4f4f5", D: "#ffd500", F: "#009b48", B: "#0046ad", R: "#b71234", L: "#ff5800" };
  // position of sticker j in its face, as seen from outside the face
  const cellOf = (s) => {
    const [x, y, z] = s.p, f = faceOf(s.n);
    if (f === "U") return [z + 1, x + 1];
    if (f === "D") return [1 - z, x + 1];
    if (f === "F") return [1 - y, x + 1];
    if (f === "B") return [1 - y, 1 - x];
    if (f === "R") return [1 - y, 1 - z];
    return [1 - y, z + 1];
  };
  root.innerHTML = `
    <div class="rk-stage">
      <div class="rk-3d"><div class="rk-cube">${"UDFBRL".split("").map((f) => `<div class="rk-face rk-${f}">${'<i></i>'.repeat(9)}</div>`).join("")}</div></div>
      <div class="rk-net">${"ULFRBD".split("").map((f) => `<div class="rk-nface rk-n${f}">${'<i></i>'.repeat(9)}</div>`).join("")}</div>
    </div>
    <div class="rk-stats">
      <span><b class="rk-it">0</b> iterations</span><span>temperature <b class="rk-tau">-</b></span>
      <span>wrong stickers <b class="rk-E">-</b></span><span class="rk-valid">every state is a valid cube</span>
    </div>
    <div class="rk-kinds"></div>
    <div class="rk-controls">
      <button class="md-button md-button--primary rk-run">Solve</button>
      <button class="md-button rk-scr">Scramble</button>
      <label>speed <input class="rk-speed" type="range" min="0" max="4" step="1" value="2"></label>
    </div>`;
  const $ = (q) => root.querySelector(q);
  const cubeCells = {}, netCells = {};
  for (const f of FACES) { cubeCells[f] = [...root.querySelectorAll(`.rk-${f} i`)]; netCells[f] = [...root.querySelectorAll(`.rk-n${f} i`)]; }
  const where = ST.map((s) => { const [r, c] = cellOf(s); return { f: faceOf(s.n), k: r * 3 + c }; });
  const KLAB = { turn: "face turns", corner3: "corner 3-cycles", edge3: "edge 3-cycles", twist: "corner twists", flip: "edge flips", swap: "swaps" };
  function draw() {
    for (let j = 0; j < 54; j++) {
      const col = COLORS[FACES[HOME[S.st[j]]]], w = where[j];
      cubeCells[w.f][w.k].style.background = col; netCells[w.f][w.k].style.background = col;
    }
    $(".rk-it").textContent = S.total.toLocaleString();
    $(".rk-tau").textContent = S.solved ? "-" : tauAt(S.n).toFixed(2);
    $(".rk-E").textContent = S.E;
    $(".rk-kinds").innerHTML = "accepted: " + KINDS.map((k, i) => `<span class="${i ? "rk-sc" : ""}">${KLAB[k]} <b>${S.acc[i].toLocaleString()}</b></span>`).join("") +
      (S.restarts ? ` <span>restarts <b>${S.restarts}</b></span>` : "");
    $(".rk-run").textContent = S.solved ? "Solved" : S.running ? "Pause" : "Solve";
    root.classList.toggle("rk-done", S.solved);
  }

  const SPEED = [40, 120, 250, 1200, 8000];
  let raf = 0;
  function loop() {
    if (!S.running) return;
    const n = SPEED[+$(".rk-speed").value];
    for (let i = 0; i < n && !S.solved; i++) stepOnce();
    draw();
    if (S.solved) { S.running = false; draw(); return; }
    raf = requestAnimationFrame(loop);
  }
  function start() { if (!LIB) buildLibrary(); if (S.solved) return; S.running = true; draw(); cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); }
  function pause() { S.running = false; draw(); }
  $(".rk-run").addEventListener("click", () => (S.running ? pause() : start()));
  $(".rk-scr").addEventListener("click", () => { pause(); seed = (Math.random() * 1e9) | 0; scramble(40); draw(); });
  seed = 20261117; scramble(40); draw();
  // build the moves and start when the demo comes into view
  const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!still && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting && S.total === 0 && !S.running) { io.disconnect(); setTimeout(start, 400); } }), { threshold: 0.4 });
    io.observe(root);
  }
  window.NimbusRubik = { S, stepOnce, scramble, setWeights: (w) => { WEIGHT = w; }, apply: (seq) => { S.running = false; S.st = seq.split(" ").reduce((st, m) => mul(st, MV[m]), new Uint8Array(ID)); S.E = energy(S.st); S.solved = false; draw(); }, build: () => { if (!LIB) buildLibrary(); return LIB.map((g) => g.length); }, setSeed: (s) => { seed = s; } };
})();
