// Surface-code decoding demo. A correction (x and z bits per data qubit) must reproduce the measured syndrome: the
// constraint. Moves that keep it: every stabilizer (flips the x or z bits of its qubits) and the two logical operators
// (change the logical class). Energy for depolarizing noise of rate p: omega times the number of qubits that the
// correction touches, omega = ln(3(1 - p)/p), so that sampling at temperature 1 visits each correction with its
// posterior probability. Parallel tempering (replicas from tau = 1 to 10 that exchange temperatures) lets the replica
// at tau = 1 move between logical classes; the decision is the class it visits most often.
(function () {
  "use strict";
  const root = document.getElementById("nimbus-surface-demo");
  if (!root) return;

  let seed = 7;
  const rnd = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

  // ---------- the rotated surface code ----------
  function buildCode(d) {
    const q = (r, c) => r * d + c;
    const checks = []; // {type: 'X' | 'Z', qubits, cell: [i, j]}
    for (let i = -1; i < d; i++) for (let j = -1; j < d; j++) {
      const type = (i + j) % 2 === 0 ? "X" : "Z";
      const qs = [[i, j], [i, j + 1], [i + 1, j], [i + 1, j + 1]].filter(([r, c]) => r >= 0 && r < d && c >= 0 && c < d).map(([r, c]) => q(r, c));
      const interior = i >= 0 && j >= 0 && i < d - 1 && j < d - 1;
      const topBot = (i === -1 || i === d - 1) && j >= 0 && j < d - 1, leftRight = (j === -1 || j === d - 1) && i >= 0 && i < d - 1;
      if (interior || (topBot && type === "X") || (leftRight && type === "Z")) checks.push({ type, qubits: qs, cell: [i, j] });
    }
    const N = d * d;
    const XL = Array.from({ length: d }, (_, r) => q(r, 0)); // X on a column: commutes with every Z check
    const ZL = Array.from({ length: d }, (_, c) => q(0, c)); // Z on a row: commutes with every X check
    // sanity: logicals commute with the checks of the other type
    for (const ch of checks) {
      const ov = (L) => ch.qubits.filter((k) => L.includes(k)).length % 2;
      if ((ch.type === "Z" && ov(XL)) || (ch.type === "X" && ov(ZL))) throw new Error("logical operator does not commute");
    }
    return { d, N, checks, XL, ZL };
  }

  // syndrome of an error (x, z bits): Z checks see x bits, X checks see z bits
  const syndrome = (C, x, z) => C.checks.map((ch) => ch.qubits.reduce((s, k) => s ^ (ch.type === "Z" ? x[k] : z[k]), 0));
  // logical class of an error relative to the identity: (x part anticommutes with ZL, z part with XL)
  const classOf = (C, x, z) => (C.ZL.reduce((s, k) => s ^ x[k], 0) ? 1 : 0) + (C.XL.reduce((s, k) => s ^ z[k], 0) ? 2 : 0);
  const CLASS_NAME = ["I", "X", "Z", "Y"];

  // any correction with the given syndrome: Gaussian elimination over GF(2), one system per error type
  function startCorrection(C, syn) {
    const x = new Uint8Array(C.N), z = new Uint8Array(C.N);
    for (const [type, bits] of [["Z", x], ["X", z]]) {
      const rows = [], rhs = [];
      C.checks.forEach((ch, i) => { if (ch.type === type) { const r = new Uint8Array(C.N); ch.qubits.forEach((k) => (r[k] = 1)); rows.push(r); rhs.push(syn[i]); } });
      const piv = []; let r0 = 0;
      for (let col = 0; col < C.N && r0 < rows.length; col++) {
        let p = -1;
        for (let r = r0; r < rows.length; r++) if (rows[r][col]) { p = r; break; }
        if (p < 0) continue;
        [rows[r0], rows[p]] = [rows[p], rows[r0]]; [rhs[r0], rhs[p]] = [rhs[p], rhs[r0]];
        for (let r = 0; r < rows.length; r++) if (r !== r0 && rows[r][col]) { for (let c = 0; c < C.N; c++) rows[r][c] ^= rows[r0][c]; rhs[r] ^= rhs[r0]; }
        piv.push(col); r0++;
      }
      piv.forEach((col, r) => (bits[col] = rhs[r]));
    }
    return { x, z };
  }

  // ---------- the sampler ----------
  const S = { C: null, p: 0.1, ex: null, ez: null, syn: null, reps: [], counts: [0, 0, 0, 0], sweeps: 0, burn: 200, budget: 4000, running: false, flash: [] };
  let MOVES = [];
  function setup(d) {
    S.C = buildCode(d);
    MOVES = S.C.checks.map((ch, i) => ({ kind: ch.type === "X" ? "x" : "z", qubits: ch.qubits, check: i }))
      .concat([{ kind: "x", qubits: S.C.XL, logical: "X" }, { kind: "z", qubits: S.C.ZL, logical: "Z" }]);
  }
  const omega = () => Math.log((3 * (1 - S.p)) / S.p);
  function newErrors(custom) {
    const N = S.C.N;
    if (!custom) {
      S.ex = new Uint8Array(N); S.ez = new Uint8Array(N);
      for (let k = 0; k < N; k++) if (rnd() < S.p) { const t = Math.floor(rnd() * 3); if (t !== 1) S.ex[k] = 1; if (t !== 0) S.ez[k] = 1; }
    }
    S.syn = syndrome(S.C, S.ex, S.ez);
    const c0 = startCorrection(S.C, S.syn);
    const R = 12;
    S.reps = Array.from({ length: R }, (_, r) => ({ x: new Uint8Array(c0.x), z: new Uint8Array(c0.z), tau: Math.pow(10, r / (R - 1)) }));
    S.reps.forEach((rp) => (rp.E = weight(rp)));
    S.counts = [0, 0, 0, 0]; S.sweeps = 0;
  }
  const weight = (rp) => { let w = 0; for (let k = 0; k < S.C.N; k++) w += rp.x[k] | rp.z[k]; return w; };
  function sweep() {
    const om = omega();
    for (const rp of S.reps) {
      for (let mi = 0; mi < MOVES.length; mi++) {
        const mv = MOVES[Math.floor(rnd() * MOVES.length)], bits = mv.kind === "x" ? rp.x : rp.z, other = mv.kind === "x" ? rp.z : rp.x;
        let dw = 0;
        for (const k of mv.qubits) { const before = bits[k] | other[k], after = (bits[k] ^ 1) | other[k]; dw += after - before; }
        if (dw <= 0 || rnd() < Math.exp((-om * dw) / rp.tau)) {
          for (const k of mv.qubits) bits[k] ^= 1;
          rp.E += dw;
          if (rp === S.reps[0] && S.flash.length < 6) S.flash.push({ mv, t: performance.now() });
        }
      }
    }
    // replica exchange between neighbouring temperatures
    for (let r = S.sweeps % 2; r + 1 < S.reps.length; r += 2) {
      const a = S.reps[r], b = S.reps[r + 1];
      const arg = om * (a.E - b.E) * (1 / a.tau - 1 / b.tau);
      if (arg >= 0 || rnd() < Math.exp(arg)) {
        // swap the states, keep the temperatures in place
        [a.x, b.x] = [b.x, a.x]; [a.z, b.z] = [b.z, a.z]; [a.E, b.E] = [b.E, a.E];
      }
    }
    S.sweeps++;
    if (S.sweeps > S.burn) S.counts[classOf(S.C, S.reps[0].x, S.reps[0].z)]++;
  }
  // the class of the true error, relative to the start correction's class frame
  const trueClass = () => classOf(S.C, S.ex, S.ez);
  const decision = () => { let b = 0; for (let c = 1; c < 4; c++) if (S.counts[c] > S.counts[b]) b = c; return b; };

  // ---------- drawing ----------
  root.innerHTML = `
    <div class="sc-stage">
      <svg class="sc-svg" viewBox="0 0 400 400" role="img" aria-label="Surface code lattice"></svg>
      <div class="sc-side">
        <div class="sc-bars"></div>
        <div class="sc-decision"></div>
        <div class="sc-legend">
          <span><i class="sc-k-x"></i>X check</span><span><i class="sc-k-z"></i>Z check</span><span><i class="sc-k-lit"></i>violated check</span>
          <span><b class="sc-k-err">X</b>error</span><span><b class="sc-k-cor">X</b>correction (τ = 1)</span>
        </div>
      </div>
    </div>
    <div class="sc-stats"><span><b class="sc-sw">0</b> sweeps</span><span>replicas <b>12</b>, τ = 1 to 10</span><span class="sc-ok">every correction reproduces the syndrome</span></div>
    <div class="sc-controls">
      <button class="md-button md-button--primary sc-run">Decode</button>
      <button class="md-button sc-new">New errors</button>
      <label>distance <select class="sc-d"><option>3</option><option selected>5</option><option>7</option></select></label>
      <label>error rate <input class="sc-p" type="range" min="0.03" max="0.18" step="0.01" value="0.10"> <b class="sc-pv">0.10</b></label>
      <label class="sc-chk"><input type="checkbox" class="sc-show" checked> show the errors</label>
    </div>
    <p class="sc-hint">Click a qubit to cycle its error: none → X → Z → Y.</p>`;
  const $ = (q) => root.querySelector(q);
  const svg = $(".sc-svg"), NS = "http://www.w3.org/2000/svg";
  const el = (tag, a, p, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.textContent = txt; if (p) p.appendChild(e); return e; };
  let G = null;
  function buildSvg() {
    svg.innerHTML = "";
    const d = S.C.d, m = 50, s = (400 - 2 * m) / (d - 1);
    const X = (c) => m + c * s, Y = (r) => m + r * s;
    G = { checks: [], qubits: [] };
    S.C.checks.forEach((ch) => {
      const [i, j] = ch.cell;
      let shape;
      if (ch.qubits.length === 4) shape = el("rect", { x: X(j), y: Y(i), width: s, height: s }, svg);
      else {
        // half disk on the boundary
        let dpath;
        if (i === -1) dpath = `M${X(j)},${Y(0)} A${s / 2},${s / 2} 0 0 1 ${X(j + 1)},${Y(0)} Z`;
        else if (i === d - 1) dpath = `M${X(j)},${Y(d - 1)} A${s / 2},${s / 2} 0 0 0 ${X(j + 1)},${Y(d - 1)} Z`;
        else if (j === -1) dpath = `M${X(0)},${Y(i)} A${s / 2},${s / 2} 0 0 0 ${X(0)},${Y(i + 1)} Z`;
        else dpath = `M${X(d - 1)},${Y(i)} A${s / 2},${s / 2} 0 0 1 ${X(d - 1)},${Y(i + 1)} Z`;
        shape = el("path", { d: dpath }, svg);
      }
      shape.setAttribute("class", "sc-check sc-" + ch.type.toLowerCase());
      G.checks.push(shape);
    });
    for (let r = 0; r < d; r++) for (let c = 0; c < d; c++) {
      const k = r * d + c;
      const g = el("g", { class: "sc-q", transform: `translate(${X(c)},${Y(r)})` }, svg);
      el("circle", { r: Math.min(13, s * 0.22) }, g);
      const err = el("text", { class: "sc-err", x: -Math.min(13, s * 0.22) - 2, y: -Math.min(13, s * 0.22) - 1 }, g, "");
      const cor = el("text", { class: "sc-cor", y: 4 }, g, "");
      g.addEventListener("click", () => {
        // cycle none -> X -> Z -> Y -> none (code: x bit + 2 z bit, so X = 1, Z = 2, Y = 3)
        const cur = S.ex[k] + 2 * S.ez[k], next = (cur + 1) % 4;
        S.ex[k] = next & 1; S.ez[k] = next >> 1 & 1;
        const was = S.running; stop(); newErrors(true); render(); if (was) run();
      });
      G.qubits.push({ g, err, cor });
    }
  }
  const letter = (x, z) => (x && z ? "Y" : x ? "X" : z ? "Z" : "");
  function render() {
    const show = $(".sc-show").checked, now = performance.now();
    S.C.checks.forEach((ch, i) => {
      G.checks[i].classList.toggle("sc-lit", !!S.syn[i]);
      const fl = S.flash.find((f) => f.mv.check === i && now - f.t < 250);
      G.checks[i].classList.toggle("sc-flash", !!fl);
    });
    S.flash = S.flash.filter((f) => now - f.t < 250);
    const rp = S.reps[0];
    G.qubits.forEach((Q, k) => {
      Q.err.textContent = show ? letter(S.ex[k], S.ez[k]) : "";
      Q.cor.textContent = letter(rp.x[k], rp.z[k]);
      Q.g.classList.toggle("sc-has-cor", !!(rp.x[k] | rp.z[k]));
    });
    const tot = S.counts.reduce((a, b) => a + b, 0), dec = decision(), tc = trueClass();
    // classes are shown relative to the true error: "correct" is the class that undoes it
    $(".sc-bars").innerHTML = `<div class="sc-bt">visits of the logical classes by the replica at τ = 1</div>` + [0, 1, 2, 3].map((c) => {
      const f = tot ? S.counts[c] / tot : 0;
      return `<div class="sc-bar ${c === tc && show ? "sc-true" : ""} ${tot && c === dec ? "sc-dec" : ""}"><span>${CLASS_NAME[c]}</span><i style="width:${(100 * f).toFixed(1)}%"></i><b>${(100 * f).toFixed(0)}%</b></div>`;
    }).join("");
    $(".sc-sw").textContent = S.sweeps.toLocaleString();
    if (!tot) $(".sc-decision").innerHTML = `<span class="sc-dim">The decision appears after the first ${S.burn} sweeps.</span>`;
    else {
      const ok = dec === tc, conf = S.counts[dec] / tot;
      $(".sc-decision").innerHTML = `Decision: class <b>${CLASS_NAME[dec]}</b>, confidence ${(100 * conf).toFixed(0)}%` +
        (show ? ` · <span class="${ok ? "sc-good" : "sc-bad"}">${ok ? "✓ corrects the error" : "✗ logical error"}</span>` : "");
    }
    $(".sc-run").textContent = S.running ? "Pause" : S.sweeps >= S.budget ? "Decode again" : "Decode";
  }

  const SWEEPS_PER_FRAME = 10;
  let raf = 0;
  function loop() {
    if (!S.running) return;
    for (let i = 0; i < SWEEPS_PER_FRAME && S.sweeps < S.budget; i++) sweep();
    render();
    if (S.sweeps >= S.budget) { S.running = false; render(); return; }
    raf = requestAnimationFrame(loop);
  }
  function run() { if (S.sweeps >= S.budget) newErrors(true); S.running = true; render(); cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); }
  function stop() { S.running = false; cancelAnimationFrame(raf); }
  $(".sc-run").addEventListener("click", () => (S.running ? (stop(), render()) : run()));
  $(".sc-new").addEventListener("click", () => { stop(); seed = (Math.random() * 1e9) | 0; newErrors(false); render(); run(); });
  $(".sc-d").addEventListener("change", () => { stop(); setup(+$(".sc-d").value); newErrors(false); buildSvg(); render(); });
  $(".sc-p").addEventListener("input", () => { $(".sc-pv").textContent = (+$(".sc-p").value).toFixed(2); S.p = +$(".sc-p").value; });
  $(".sc-p").addEventListener("change", () => { stop(); newErrors(true); render(); });
  $(".sc-show").addEventListener("change", render);
  setup(5); S.p = 0.1; seed = 424242; newErrors(false); buildSvg(); render();
  const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!still && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting && S.sweeps === 0 && !S.running) { io.disconnect(); setTimeout(run, 400); } }), { threshold: 0.4 });
    io.observe(root);
  }
  window.NimbusSurface = { S, sweep, setup, newErrors, trueClass, decision, setSeed: (s) => { seed = s; }, classOf, syndrome, buildCode };
})();
