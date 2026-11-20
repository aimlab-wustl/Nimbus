// "How NIMBUS works": the derivation of the architecture on a small example, three beads red, green and blue in
// positions p1, p2, p3, with
//   E = -(red in p1) - 2 (blue in p3) - (green in p1)(blue in p2) - 2 (red in p3).
// Spin s_{3(x-1)+a} = +1 if colour x is in position p_a. The compiler below: for every
// stored clause and every move, the switch is a constant or (1 + p T_j)/2 for one stored clause T_j; a product that
// is not stored is appended as an auxiliary clause (weight 0), until no new clause is needed.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const COL = [
    { name: "red", hex: "#d55e00" },
    { name: "green", hex: "#009e73" },
    { name: "blue", hex: "#0072b2" },
  ];
  const SUB = "₀₁₂₃₄₅₆₇₈₉";
  const sub = (n) => String(n).split("").map((d) => SUB[+d]).join("");

  // ---------------- model ----------------
  const PAPER = ["RGB", "RBG", "GRB", "BRG", "GBR", "BGR"].map((w) => w.split("").map((c) => "RGB".indexOf(c)));
  const key = (arr) => arr.join("");
  const STATE_ID = new Map(PAPER.map((a, i) => [key(a), i + 1]));
  const spins = (arr) => { const s = Array(9).fill(-1); arr.forEach((x, a) => { s[3 * x + a] = 1; }); return s; };
  const S = PAPER.map(spins);
  const prod = (s, sup) => sup.reduce((p, i) => p * s[i], 1);
  const spinLabel = (i) => ({ x: Math.floor(i / 3), a: i % 3 });
  const OBJ = [
    { sup: [0], J: 1 }, { sup: [8], J: 2 }, { sup: [3, 7], J: 1 }, { sup: [2], J: 2 },
  ];
  const MOVES = [
    { id: "G1", a: 0, b: 1, label: "red ↔ green" },
    { id: "G2", a: 1, b: 2, label: "green ↔ blue" },
    { id: "G3", a: 0, b: 2, label: "red ↔ blue", shortcut: true },
  ];
  const applyMove = (arr, m) => arr.map((x) => (x === m.a ? m.b : x === m.b ? m.a : x));
  const energyArr = (arr) => { const s = spins(arr); return -OBJ.reduce((e, c) => e + c.J * prod(s, c.sup), 0); };
  const E_OF = PAPER.map(energyArr); // -2, 4, 2, 4, 0, 0

  function compile(moves) {
    const cl = OBJ.map((c) => ({ sup: c.sup.slice(), J: c.J, aux: false }));
    const sw = []; // sw[k][r] = {c: 0|1} or {j, p}
    const fitStored = (b) => {
      for (let j = 0; j < cl.length; j++) for (const p of [-1, 1]) {
        if (S.every((s, n) => b[n] === (1 + p * prod(s, cl[j].sup)) / 2)) return { j, p };
      }
      return null;
    };
    const fitNew = (b) => {
      const subsets = [];
      for (let i = 0; i < 9; i++) subsets.push([i]);
      for (let i = 0; i < 9; i++) for (let j = i + 1; j < 9; j++) subsets.push([i, j]);
      for (const sup of subsets) for (const p of [-1, 1]) {
        if (S.every((s, n) => b[n] === (1 + p * prod(s, sup)) / 2)) return { sup, p };
      }
      return null;
    };
    for (let k = 0; k < cl.length; k++) {
      sw[k] = [];
      moves.forEach((m, r) => {
        const b = PAPER.map((arr, n) => (prod(S[n], cl[k].sup) !== prod(spins(applyMove(arr, m)), cl[k].sup) ? 1 : 0));
        if (b.every((x) => x === b[0])) { sw[k][r] = { c: b[0] }; return; }
        let f = fitStored(b);
        if (!f) { const g = fitNew(b); cl.push({ sup: g.sup, J: 0, aux: true }); f = { j: cl.length - 1, p: g.p }; }
        sw[k][r] = f;
      });
    }
    return { cl, sw, moves };
  }
  const alpha = (net, k, r, T) => { const w = net.sw[k][r]; return "c" in w ? w.c : (1 + w.p * T[w.j]) / 2; };
  const inputOf = (net, r, T) => net.cl.reduce((v, c, k) => v + c.J * alpha(net, k, r, T) * T[k], 0);
  const clauseOutputs = (net, arr) => { const s = spins(arr); return net.cl.map((c) => prod(s, c.sup)); };
  const BACKBONE = compile(MOVES.slice(0, 2));
  const WITH_SHORTCUT = compile(MOVES);

  // ---------------- small drawing helpers ----------------
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const a in attrs) e.setAttribute(a, attrs[a]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const fmt = (x) => (Math.abs(x) < 1e-9 ? "0" : (x < 0 ? "−" : "") + String(+Math.abs(x).toFixed(2)));
  const sgn = (x) => (x > 0 ? "+1" : "−1");
  const dot = (x) => `<i class="nd-dot" style="background:${COL[x].hex}"></i>`;
  const clauseText = (sup) => sup.map((i) => { const { x, a } = spinLabel(i); return `(${dot(x)} in p${sub(a + 1)})`; }).join("");
  const clauseSpins = (sup) => sup.map((i) => "s" + sub(i + 1)).join("");
  function beads(svg, arr, x, y, r, gap, cls) {
    const g = el("g", { class: "nd-beads " + (cls || ""), transform: `translate(${x},${y})` }, svg);
    el("line", { x1: 0, y1: 0, x2: 2 * gap, y2: 0, class: "nd-wire" }, g);
    arr.forEach((c, a) => el("circle", { cx: a * gap, cy: 0, r, fill: COL[c].hex, class: "nd-bead" }, g));
    return g;
  }

  // ---------------- 1. the problem ----------------
  function problemWidget(root) {
    let arr = [1, 2, 0]; // start: green, blue, red (state 5)
    let s = spins(arr), picked = null;
    root.innerHTML = `
      <div class="nd-row">
        <div class="nd-col">
          <div class="nd-cap">Click two beads to swap them</div>
          <svg class="nd-svg nd-beadsvg" viewBox="0 0 300 120"></svg>
          <div class="nd-energy"></div>
        </div>
        <div class="nd-col">
          <div class="nd-cap">Spins: one per colour and position. Click one to flip it alone</div>
          <div class="nd-grid"></div>
          <div class="nd-feas"></div>
          <button class="md-button nd-small nd-fix">Back to a valid arrangement</button>
        </div>
      </div>`;
    const svg = root.querySelector(".nd-beadsvg"), grid = root.querySelector(".nd-grid");
    function draw() {
      svg.innerHTML = "";
      ["p₁", "p₂", "p₃"].forEach((p, a) => el("text", { x: 60 + a * 90, y: 22, class: "nd-ptext" }, svg, p));
      const valid = isValid(s);
      el("line", { x1: 60, y1: 70, x2: 240, y2: 70, class: "nd-wire" }, svg);
      for (let a = 0; a < 3; a++) {
        const here = [0, 1, 2].filter((x) => s[3 * x + a] > 0);
        if (here.length === 1) {
          const c = el("circle", { cx: 60 + a * 90, cy: 70, r: 26, fill: COL[here[0]].hex, class: "nd-bead nd-click" + (picked === a ? " nd-picked" : "") }, svg);
          c.addEventListener("click", () => pick(a));
        } else {
          el("circle", { cx: 60 + a * 90, cy: 70, r: 26, class: "nd-bead-bad" }, svg);
          el("text", { x: 60 + a * 90, y: 76, class: "nd-badtext" }, svg, here.length ? here.map((x) => COL[x].name[0].toUpperCase()).join("+") : "?");
        }
      }
      // energy with every bracket evaluated
      const terms = OBJ.map((c, k) => ({ k, J: c.J, T: prod(s, c.sup) }));
      const E = -terms.reduce((e, t) => e + t.J * t.T, 0);
      root.querySelector(".nd-energy").innerHTML =
        `<div class="nd-eq">E = ${terms.map((t, i) => `${i ? " − " : "−"}${t.J > 1 ? t.J + " " : ""}<span class="nd-term" title="T${sub(t.k + 1)}">${clauseText(OBJ[t.k].sup)}</span>`).join("")}</div>` +
        `<div class="nd-eq nd-eqnum">= ${terms.map((t, i) => `${i ? " − " : "−"}${t.J > 1 ? t.J + "·" : ""}<b class="${t.T > 0 ? "nd-pos" : "nd-neg"}">(${sgn(t.T)})</b>`).join("")} = <b class="nd-E">${fmt(E)}</b></div>`;
      // spin grid
      grid.innerHTML = `<div></div>${["p₁", "p₂", "p₃"].map((p) => `<div class="nd-gh">${p}</div>`).join("")}` +
        [0, 1, 2].map((x) => `<div class="nd-gh">${dot(x)} ${COL[x].name}</div>` + [0, 1, 2].map((a) => {
          const i = 3 * x + a;
          return `<button class="nd-spin ${s[i] > 0 ? "nd-up" : ""}" data-i="${i}"><span>s${sub(i + 1)}</span><b>${s[i] > 0 ? "+1" : "−1"}</b></button>`;
        }).join("")).join("");
      grid.querySelectorAll(".nd-spin").forEach((b) => b.addEventListener("click", () => { s[+b.dataset.i] *= -1; picked = null; draw(); }));
      const badRows = [0, 1, 2].filter((x) => [0, 1, 2].filter((a) => s[3 * x + a] > 0).length !== 1);
      const badCols = [0, 1, 2].filter((a) => [0, 1, 2].filter((x) => s[3 * x + a] > 0).length !== 1);
      root.querySelector(".nd-feas").innerHTML = valid
        ? `<span class="nd-ok">✓ every colour in exactly one position and every position holds one colour</span>`
        : `<span class="nd-bad">✗ infeasible: ${badRows.map((x) => COL[x].name).concat(badCols.map((a) => "p" + sub(a + 1))).join(", ")} violate${badRows.length + badCols.length > 1 ? "" : "s"} the constraint. A single spin flip always leaves the feasible set.</span>`;
      root.querySelector(".nd-fix").style.visibility = valid ? "hidden" : "visible";
    }
    const isValid = (s) => [0, 1, 2].every((x) => [0, 1, 2].filter((a) => s[3 * x + a] > 0).length === 1) &&
      [0, 1, 2].every((a) => [0, 1, 2].filter((x) => s[3 * x + a] > 0).length === 1);
    function pick(a) {
      if (picked === null) { picked = a; draw(); return; }
      if (picked !== a) { const t = arr[a]; arr[a] = arr[picked]; arr[picked] = t; s = spins(arr); }
      picked = null; draw();
    }
    root.querySelector(".nd-fix").addEventListener("click", () => { s = spins(arr); draw(); });
    draw();
  }

  // ---------------- 2-4. the graph of feasible states ----------------
  const HEX = { 1: [120, 1], 2: [60, 1], 5: [0, 1], 6: [-60, 1], 4: [-120, 1], 3: [180, 1] };
  const nodeXY = (id, cx, cy, R) => { const t = (HEX[id][0] * Math.PI) / 180; return [cx + R * 1.25 * Math.cos(t), cy - R * Math.sin(t)]; };
  const changed = (i, j) => S[i - 1].reduce((n, v, k) => n + (v !== S[j - 1][k] ? 1 : 0), 0);
  const MOVE_EDGES = MOVES.map((m) => PAPER.map((arr, n) => [n + 1, STATE_ID.get(key(applyMove(arr, m)))]).filter(([a, b]) => a < b));

  function graphWidget(root, mode) {
    const W = 560, H = 330, cx = 280, cy = 165, R = 120;
    let sel = [], tree = 2, cur = 6, shortcut = false, path = [6];
    const bar = mode === "trees" ? `<div class="nd-tabs"><button data-t="1">Tree 1</button><button data-t="2" class="on">Tree 2</button></div>`
      : mode === "moves" ? `<div class="nd-tabs"><button data-m="0" class="nd-g1">apply G₁: red ↔ green</button><button data-m="1" class="nd-g2">apply G₂: green ↔ blue</button>` +
        `<button data-m="2" class="nd-g3" disabled>apply G₃: red ↔ blue</button><label class="nd-chk"><input type="checkbox" class="nd-sc"> add the shortcut G₃</label>` +
        `<button class="nd-reset">start again at state 6</button></div>` : "";
    root.innerHTML = `${bar}<div class="nd-row"><div class="nd-col nd-wide"><svg class="nd-svg" viewBox="0 0 ${W} ${H}"></svg></div>` +
      `<div class="nd-col nd-side"></div></div>`;
    const svg = root.querySelector("svg"), side = root.querySelector(".nd-side");
    function edgeSet() {
      if (mode === "pairs") { const all = []; for (let i = 1; i <= 6; i++) for (let j = i + 1; j <= 6; j++) all.push([i, j, "nd-e-all"]); return all; }
      if (mode === "trees") return (tree === 1 ? [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]] : [[1, 2], [2, 5], [5, 6], [6, 4], [4, 3]]).map(([a, b]) => [a, b, "nd-e-tree"]);
      const es = MOVE_EDGES[0].map(([a, b]) => [a, b, "nd-e-g1"]).concat(MOVE_EDGES[1].map(([a, b]) => [a, b, "nd-e-g2"]));
      return shortcut ? es.concat(MOVE_EDGES[2].map(([a, b]) => [a, b, "nd-e-g3"])) : es;
    }
    function draw() {
      svg.innerHTML = "";
      for (const [a, b, cls] of edgeSet()) {
        const [x1, y1] = nodeXY(a, cx, cy, R), [x2, y2] = nodeXY(b, cx, cy, R);
        const on = (sel.length === 2 && sel.includes(a) && sel.includes(b)) ||
          (mode === "moves" && path.length > 1 && path.some((p, n) => n && ((path[n - 1] === a && p === b) || (path[n - 1] === b && p === a))));
        el("line", { x1, y1, x2, y2, class: `nd-edge ${cls}${on ? " nd-edge-on" : ""}` }, svg);
        if (mode === "trees" || (mode === "pairs" && on)) {
          const n = changed(a, b);
          el("rect", { x: (x1 + x2) / 2 - 10, y: (y1 + y2) / 2 - 10, width: 20, height: 20, rx: 10, class: "nd-elab-bg" }, svg);
          el("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 + 4, class: "nd-elab" }, svg, n);
        }
      }
      for (let id = 1; id <= 6; id++) {
        const [x, y] = nodeXY(id, cx, cy, R);
        const g = el("g", { class: "nd-node" + (sel.includes(id) || (mode === "moves" && id === cur) ? " nd-node-on" : "") + (id === 1 ? " nd-opt" : ""), transform: `translate(${x},${y})` }, svg);
        el("rect", { x: -44, y: -20, width: 88, height: 40, rx: 20, class: "nd-node-bg" }, g);
        PAPER[id - 1].forEach((c, a) => el("circle", { cx: -24 + a * 24, cy: 0, r: 9, fill: COL[c].hex }, g));
        el("text", { x: 0, y: -27, class: "nd-node-id" }, g, `${id}`);
        el("text", { x: 0, y: 36, class: "nd-node-E" }, g, `E = ${fmt(E_OF[id - 1])}`);
        if (mode === "pairs") g.addEventListener("click", () => { sel = sel.length === 2 ? [id] : sel.includes(id) ? sel : sel.concat(id); draw(); });
      }
      side.innerHTML = sideText();
    }
    function sideText() {
      if (mode === "pairs") {
        if (sel.length < 2) return `<p class="nd-hint">Click a state, then another one, to inspect the spin transition between them.</p>`;
        const [i, j] = sel, mu = S[i - 1], nu = S[j - 1];
        const ch = mu.map((v, k) => (v !== nu[k] ? k : -1)).filter((k) => k >= 0);
        const rows = OBJ.map((c, k) => {
          const odd = c.sup.filter((q) => ch.includes(q)).length % 2;
          return `<tr><td>T${sub(k + 1)} = ${clauseSpins(c.sup)}</td><td>${c.sup.filter((q) => ch.includes(q)).length}</td><td><b>${odd}</b></td><td>${sgn(prod(mu, c.sup))}</td></tr>`;
        }).join("");
        const sum = OBJ.reduce((v, c) => v + c.J * (c.sup.filter((q) => ch.includes(q)).length % 2) * prod(mu, c.sup), 0);
        return `<p><b>${i} → ${j}</b>: ${ch.length} spins change (${ch.map((k) => "s" + sub(k + 1)).join(", ")}).</p>
          <table class="nd-table"><tr><th>clause</th><th>changed spins in it</th><th>α</th><th>T(μ)</th></tr>${rows}</table>
          <p>A clause toggles exactly when an odd number of its spins change: α = 1. Then</p>
          <p class="nd-math">ΔE = 2 Σ J<sub>k</sub> α<sub>k</sub> T<sub>k</sub>(μ) = <b>${fmt(2 * sum)}</b> = E(${j}) − E(${i}) = ${fmt(E_OF[j - 1])} − (${fmt(E_OF[i - 1])})</p>`;
      }
      if (mode === "trees") return tree === 1
        ? `<p><b>Tree 1</b> connects all six states with five transitions, two of which change six spins.</p>`
        : `<p><b>Tree 2</b> connects all six states with five transitions that each change only four spins, the smallest change between two feasible states. Every longer transition is a sequence of these.</p>`;
      const trace = path.map((p) => `${p}<sub>${fmt(E_OF[p - 1])}</sub>`).join(" → ");
      return `<p>Current state <b>${cur}</b>, E = ${fmt(E_OF[cur - 1])}.</p><p class="nd-trace">${trace}</p>` +
        (cur === 1 ? `<p class="nd-ok">✓ The optimum, E = −2, reached in ${path.length - 1} transition${path.length === 2 ? "" : "s"}.</p>` : "") +
        `<p class="nd-hint">${shortcut ? "With the shortcut, state 6 reaches the optimum in one transition without climbing to E = 4." :
          "From state 6, the backbone reaches the optimum only through state 4 or 2, both at E = 4: an energy barrier."}</p>`;
    }
    root.querySelectorAll("[data-t]").forEach((b) => b.addEventListener("click", () => {
      tree = +b.dataset.t; root.querySelectorAll("[data-t]").forEach((x) => x.classList.toggle("on", x === b)); draw();
    }));
    root.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => {
      const m = MOVES[+b.dataset.m]; cur = STATE_ID.get(key(applyMove(PAPER[cur - 1], m))); path.push(cur); draw();
    }));
    const sc = root.querySelector(".nd-sc");
    if (sc) sc.addEventListener("change", () => { shortcut = sc.checked; root.querySelector(".nd-g3").disabled = !shortcut; draw(); });
    const rs = root.querySelector(".nd-reset");
    if (rs) rs.addEventListener("click", () => { cur = 6; path = [6]; draw(); });
    if (mode === "pairs") sel = [4, 3];
    draw();
  }

  // ---------------- 5. switches ----------------
  function switchWidget(root) {
    let r = 0;
    root.innerHTML = `<div class="nd-tabs"><button data-r="0" class="on nd-g1">G₁: red ↔ green</button><button data-r="1" class="nd-g2">G₂: green ↔ blue</button></div><div class="nd-sw"></div>`;
    const box = root.querySelector(".nd-sw");
    function draw() {
      const m = MOVES[r];
      const rows = PAPER.map((arr, n) => [n + 1, arr]).filter(([id, arr]) => id < STATE_ID.get(key(applyMove(arr, m)))).map(([id, arr]) => {
        const to = STATE_ID.get(key(applyMove(arr, m))), mu = S[id - 1], nu = S[to - 1];
        const a = OBJ.map((c) => (prod(mu, c.sup) !== prod(nu, c.sup) ? 1 : 0));
        const third = [0, 1, 2].find((x) => x !== m.a && x !== m.b);
        const pos = arr.indexOf(third);
        return `<tr><td class="nd-trn">${beadHTML(arr)} ↔ ${beadHTML(applyMove(arr, m))}<br><small>${id} ↔ ${to}</small></td>` +
          a.map((v) => `<td class="${v ? "nd-a1" : "nd-a0"}">${v}</td>`).join("") +
          `<td>${dot(third)} in p${sub(pos + 1)}</td></tr>`;
      }).join("");
      const third = [0, 1, 2].find((x) => x !== m.a && x !== m.b);
      box.innerHTML = `<table class="nd-table nd-swtable"><tr><th>transition of ${m.id}</th><th>α₁</th><th>α₂</th><th>α₃</th><th>α₄</th><th>${COL[third].name} stays in</th></tr>${rows}</table>
        <p class="nd-hint">${r === 0
          ? "G₁ moves red and green but never blue. Whether T₁ = (red in p₁) toggles depends on whether blue occupies p₁, that is, on the spin s₇: α₁ = (1 − s₇)/2. Likewise α₃ = (1 − s₇)/2 and α₄ = (1 − s₉)/2."
          : "G₂ moves green and blue but never red. T₂ and T₃ toggle unless red occupies p₃, so α₂ = α₃ = (1 − s₃)/2; T₁ and T₄ never toggle."}</p>`;
    }
    root.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => {
      r = +b.dataset.r; root.querySelectorAll("[data-r]").forEach((x) => x.classList.toggle("on", x === b)); draw();
    }));
    draw();
  }
  const beadHTML = (arr) => `<span class="nd-bh">${arr.map((c) => dot(c)).join("")}</span>`;

  // ---------------- 6. auxiliary clauses: rewrite the switches in clause outputs ----------------
  function auxWidget(root) {
    // the switches of the backbone in terms of spins, and the clause that supplies each spin
    const rows = [
      ["(1 − s₇)/2", "0"], ["0", "(1 − s₃)/2"], ["(1 − s₇)/2", "(1 − s₃)/2"], ["(1 − s₉)/2", "0"],
    ];
    const SUPPLY = { "s₉": "T₂", "s₃": "T₄", "s₇": "T₅" };
    let stage = 0;
    root.innerHTML = `<div class="nd-auxwrap"><table class="nd-table nd-auxtable"></table><div class="nd-auxside"><p class="nd-auxmsg"></p>
      <button class="md-button md-button--primary nd-small nd-auxgo"></button></div></div>`;
    const tbl = root.querySelector("table"), msg = root.querySelector(".nd-auxmsg"), go = root.querySelector(".nd-auxgo");
    const cell = (txt) => txt.replace(/s([₃₇₉])/g, (m) => {
      if (stage === 0) return `<span class="nd-hid">${m}</span>`;
      if (m === "s₇" && stage === 1) return `<span class="nd-miss">${m}</span>`;
      return `<span class="nd-sup ${m === "s₇" ? "nd-sup-aux" : ""}">${SUPPLY[m]}</span>`;
    });
    function draw() {
      const body = rows.map((r, k) => `<tr><td>T${sub(k + 1)}</td><td>${cell(r[0])}</td><td>${cell(r[1])}</td></tr>`).join("") +
        (stage >= 2 ? `<tr class="nd-auxrow"><td>T₅</td><td>0</td><td><span class="nd-sup">(1 − T₁)/2</span></td></tr>` : "");
      tbl.innerHTML = `<tr><th>clause</th><th>switch to G₁</th><th>switch to G₂</th></tr>${body}`;
      msg.innerHTML = [
        "The switches depend on three spins, s₃, s₇ and s₉. The machine stores clause outputs, not spins, so each of these must be the output of a stored clause.",
        "s₉ is the output of T₂ (blue in p₃) and s₃ is the output of T₄ (red in p₃). No stored clause has the output s₇ (blue in p₁).",
        "The compiler appends T₅ = s₇ with weight 0: an auxiliary clause. Its own switch to G₂ is set by T₁, which is stored, so no further clause is needed. Every switch now reads one clause output.",
      ][stage];
      go.textContent = ["Express the switches in clause outputs", "Append the missing clause", "Start again"][stage];
    }
    go.addEventListener("click", () => { stage = (stage + 1) % 3; draw(); });
    draw();
  }

  // ---------------- 8. the machine ----------------
  // One column of clause neurons, one of move neurons. Every connection is drawn with its switch (a lever near the
  // move neuron) and a thin red line from the clause output that sets the switch. The encoder reads the clause outputs
  // along the connections whose switch is on; the decoder sends the firing neuron's spikes back along the same
  // connections; the toggled clauses then re-set the switches they control.
  function machineWidget(root) {
    const autoplay = root.dataset.autoplay === "true";
    let net = BACKBONE, routes = [], arr = PAPER[5].slice(), T, iter = 0, phase = 0, cur = null, hist = [], timer = null, busy = false, hitAt = null;
    let lastEdge = null;
    root.innerHTML = `
      <button class="nd-full" title="Full screen" aria-label="Full screen">⤢</button>
      <div class="nd-mgrid">
        <div class="nd-mmain"><svg class="nd-svg nd-msvg" viewBox="0 0 640 440" role="img" aria-label="Clause neurons connected to move neurons through switches"></svg></div>
        <aside class="nd-mside">
          <div class="nd-mhead"><svg class="nd-svg nd-mbeads" viewBox="0 0 190 56"></svg><div class="nd-mE"></div></div>
          <svg class="nd-svg nd-mmap" viewBox="0 0 300 150"></svg>
          <div class="nd-steps"><span data-p="1">encoder</span><span data-p="2">threshold</span><span data-p="3">decoder</span><span data-p="4">switches</span></div>
          <div class="nd-mtext" aria-live="polite"></div>
          <div class="nd-controls">
            <button class="md-button md-button--primary nd-next">Next step</button>
            <button class="md-button nd-play">Play</button>
            <button class="md-button nd-reset" title="Back to state 6">Reset</button>
          </div>
          <div class="nd-controls">
            <label>temperature <input class="nd-tau" type="range" min="0.2" max="4" step="0.1" value="1.2"> <b class="nd-tauv">1.2</b></label>
            <label class="nd-chk"><input type="checkbox" class="nd-sc"> shortcut G₃ (red ↔ blue)</label>
          </div>
          <svg class="nd-svg nd-mtrace" viewBox="0 0 300 64"></svg>
        </aside>
      </div>`;
    const svg = root.querySelector(".nd-msvg");
    const $ = (q) => root.querySelector(q);
    const CX = 168, MXc = 548;
    const ty = (k) => { const n = net.cl.length, top = 50, bot = 408; return n === 1 ? (top + bot) / 2 : top + ((bot - top) * k) / (n - 1); };
    const gy = (r) => { const n = net.moves.length; return 210 + (r - (n - 1) / 2) * (n > 2 ? 120 : 150); };
    const mvName = (m) => m.id.replace(/\d/, (d) => sub(d));
    const say = (t) => { $(".nd-mtext").innerHTML = t; };
    const isOnR = (rt) => alpha(net, rt.k, rt.r, T) === 1;

    // ---- beads: persistent, so that a swap slides ----
    const sb = $(".nd-mbeads");
    el("line", { x1: 72, y1: 24, x2: 168, y2: 24, class: "nd-wire" }, sb);
    const beadG = COL.map((c) => { const g = el("g", { class: "nd-slide" }, sb); el("circle", { cx: 0, cy: 24, r: 15, fill: c.hex, class: "nd-bead" }, g); return g; });
    const idText = el("text", { x: 26, y: 31, class: "nd-node-id nd-big" }, sb, "");
    ["p₁", "p₂", "p₃"].forEach((p, a) => el("text", { x: 72 + a * 48, y: 52, class: "nd-ptext nd-ptext-s" }, sb, p));
    function placeBeads() { arr.forEach((c, a) => { beadG[c].style.transform = `translate(${72 + a * 48}px, 0px)`; }); }

    // ---- the state map: one node per arrangement ----
    const mp = $(".nd-mmap");
    const MAPC = [150, 75], MAPR = 54;
    const mapXY = (id) => nodeXY(id, MAPC[0], MAPC[1], MAPR);
    let mapEdges = [];
    function buildMap() {
      mp.innerHTML = "";
      const es = MOVE_EDGES[0].map((e) => e.concat("nd-e-g1")).concat(MOVE_EDGES[1].map((e) => e.concat("nd-e-g2")))
        .concat(net === WITH_SHORTCUT ? MOVE_EDGES[2].map((e) => e.concat("nd-e-g3")) : []);
      mapEdges = es.map(([a, b, c]) => {
        const [x1, y1] = mapXY(a), [x2, y2] = mapXY(b);
        return { a, b, line: el("line", { x1, y1, x2, y2, class: "nd-edge " + c }, mp) };
      });
      for (let id = 1; id <= 6; id++) {
        const [x, y] = mapXY(id);
        const g = el("g", { class: "nd-mnode2" + (id === 1 ? " nd-opt" : ""), transform: `translate(${x},${y})`, "data-id": id }, mp);
        el("rect", { x: -27, y: -11, width: 54, height: 22, rx: 11, class: "nd-node-bg" }, g);
        PAPER[id - 1].forEach((c, a) => el("circle", { cx: -15 + a * 15, cy: 0, r: 5.6, fill: COL[c].hex }, g));
        el("text", { x: id === 3 ? -33 : id === 5 ? 33 : 0, y: id === 3 || id === 5 ? 4 : id <= 2 ? -15 : 22, class: "nd-mnode-t" + (id === 3 ? " nd-r" : id === 5 ? " nd-l" : "") }, g, id);
      }
    }
    function updateState() {
      placeBeads();
      const id = STATE_ID.get(key(arr));
      idText.textContent = id;
      $(".nd-mE").innerHTML = `E = <b>${fmt(energyArr(arr))}</b>${id === 1 ? '<br><span class="nd-ok">optimum</span>' : ""}`;
      mp.querySelectorAll(".nd-mnode2").forEach((g) => g.classList.toggle("nd-here", +g.dataset.id === id));
      mapEdges.forEach((e) => e.line.classList.toggle("nd-edge-on", !!lastEdge && ((e.a === lastEdge[0] && e.b === lastEdge[1]) || (e.a === lastEdge[1] && e.b === lastEdge[0]))));
      const tr = $(".nd-mtrace"); tr.innerHTML = "";
      const pts = hist.slice(-40), X = (n) => 26 + (n / Math.max(39, pts.length - 1)) * 268, Y = (e) => 6 + ((4 - e) / 6) * 46;
      [-2, 4].forEach((e) => { el("line", { x1: 22, x2: 296, y1: Y(e), y2: Y(e), class: "nd-gridline" }, tr); el("text", { x: 18, y: Y(e) + 3.5, class: "nd-axis nd-r" }, tr, fmt(e)); });
      if (pts.length) {
        el("polyline", { points: pts.map((e, n) => `${X(n)},${Y(e)}`).join(" "), class: "nd-traceline" }, tr);
        pts.forEach((e, n) => el("circle", { cx: X(n), cy: Y(e), r: 2, class: "nd-tracept" + (e === -2 ? " nd-opt" : "") }, tr));
      }
      el("text", { x: 296, y: 62, class: "nd-axis nd-r" }, tr, `energy over ${iter} iterations`);
    }

    // ---- the network ----
    let nodes = {};
    function buildNet() {
      svg.innerHTML = "";
      el("defs", {}, svg).innerHTML = `<marker id="nd-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="nd-arrhead"/></marker><marker id="nd-arr-gate" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 z" class="nd-arrhead-gate"/></marker>`;
      routes = [];
      net.moves.forEach((m, r) => net.cl.forEach((c, k) => {
        const w = net.sw[k][r];
        if ("c" in w && w.c === 0) return;
        routes.push({ k, r, w: c.J, ctrl: "c" in w ? -1 : w.j, pol: "c" in w ? 0 : w.p });
      }));
      const n = net.cl.length, rad = Math.min(19, (408 - 50) / Math.max(1, n - 1) / 2 - 2);
      el("text", { x: CX, y: 14, class: "nd-coltitle" }, svg, "clause neurons");
      el("text", { x: MXc, y: 14, class: "nd-coltitle" }, svg, "move neurons");
      const gCtrl = el("g", {}, svg), gConn = el("g", {}, svg), gSw = el("g", {}, svg);
      routes.forEach((rt) => {
        const x1 = CX + rad, y1 = ty(rt.k), x2 = MXc - 22, y2 = gy(rt.r);
        // a zero-weight (gate) connection is used only by the decoder: one-way, from the move neuron to the clause
        rt.path = el("path", Object.assign({ d: `M${x1 + (rt.w ? 0 : 4)},${y1} C${x1 + 150},${y1} ${x2 - 150},${y2} ${x2},${y2}`, class: "nd-conn" + (rt.w ? "" : " nd-conn-u") }, rt.w ? {} : { "marker-start": "url(#nd-arr-gate)" }), gConn);
        const L = rt.path.getTotalLength();
        rt.L = L;
        const tt = net.moves.length > 2 ? 0.3 + 0.14 * rt.r : 0.34 + 0.16 * rt.r; // one bank of switches per move neuron
        const p = rt.path.getPointAtLength(L * tt), q = rt.path.getPointAtLength(L * tt + 6);
        rt.sx = p.x; rt.sy = p.y;
        const ang = (Math.atan2(q.y - p.y, q.x - p.x) * 180) / Math.PI;
        if (rt.ctrl >= 0) {
          // the switch: a lever on the connection; closed (on) or open (off)
          rt.sw = el("g", { class: "nd-sw2", transform: `translate(${p.x},${p.y}) rotate(${ang})` }, gSw);
          el("rect", { x: -11, y: -7, width: 22, height: 14, rx: 4, class: "nd-sw2-bg" }, rt.sw);
          el("circle", { cx: -7, cy: 0, r: 2.1, class: "nd-sw2-pin" }, rt.sw);
          el("circle", { cx: 7, cy: 0, r: 2.1, class: "nd-sw2-pin" }, rt.sw);
          rt.lever = el("line", { x1: -7, y1: 0, x2: 7, y2: 0, class: "nd-sw2-lever" }, rt.sw);
          const cy = ty(rt.ctrl);
          rt.cpath = el("path", { d: `M${CX + rad - 2},${cy + 6} C${CX + rad + 40},${cy + 30} ${p.x - 30},${p.y + 40} ${p.x},${p.y + 6}`, class: "nd-ctrl2" }, gCtrl);
          rt.cL = rt.cpath.getTotalLength();
        }
      });
      nodes = { cl: [], mv: [] };
      net.cl.forEach((c, k) => {
        const g = el("g", { class: "nd-cl" + (c.aux ? " nd-aux" : ""), transform: `translate(${CX},${ty(k)})` }, svg);
        el("circle", { r: rad, class: "nd-cl-c" }, g);
        el("text", { y: -1, class: "nd-cl-t" }, g, `T${sub(k + 1)}`);
        const v = el("text", { y: rad > 15 ? 11 : 9, class: "nd-cl-v" }, g, "");
        const lab = el("foreignObject", { x: 0, y: ty(k) - 11, width: CX - rad - 8, height: 22 }, svg);
        lab.innerHTML = `<div xmlns="http://www.w3.org/1999/xhtml" class="nd-fo ${c.aux ? "nd-fo-aux" : ""}">${clauseText(c.sup)} <span class="nd-fo-J">${c.J ? "J=" + c.J : "J=0"}</span></div>`;
        nodes.cl.push({ g, v });
      });
      net.moves.forEach((m, r) => {
        const g = el("g", { class: "nd-mv" + (m.shortcut ? " nd-mv-sc" : ""), transform: `translate(${MXc},${gy(r)})` }, svg);
        el("circle", { r: 21, class: "nd-mv-c" }, g);
        el("text", { y: 5, class: "nd-mv-t" }, g, mvName(m));
        const v = el("text", { y: -28, class: "nd-mv-v" }, g, "");
        el("text", { y: 36, class: "nd-mv-l" }, g, m.label);
        nodes.mv.push({ g, v });
      });
      const ay = 420;
      el("rect", { x: MXc - 44, y: ay - 12, width: 88, height: 20, rx: 3, class: "nd-annealer" }, svg);
      el("text", { x: MXc, y: ay + 2, class: "nd-annealer-t" }, svg, "FN annealer");
      nodes.xi = el("text", { x: MXc - 50, y: ay + 2, class: "nd-xi-t nd-r" }, svg, "");
    }
    // classes and values from the current state; the switch levers follow sw (default: the current clause outputs)
    function paint(swState) {
      const sel = cur ? cur.r : -1;
      // zero-weight connections (auxiliary clauses) carry nothing to the encoder: they appear when the decoder uses
      // them and while their switches are re-set; the switch-control lines appear only in the switch stage
      const decoding = phase === 3 || phase === 4;
      routes.forEach((rt, i) => {
        const on = swState ? swState[i] : isOnR(rt);
        // outside the decoder and switch stages a gate connection is drawn like an inactive one: it feeds nothing to the encoder
        rt.path.classList.toggle("nd-conn-on", on && (rt.w || decoding));
        rt.path.classList.toggle("nd-conn-sel", on && rt.r === sel);
        if (rt.sw) {
          rt.sw.classList.toggle("nd-sw2-on", on);
          rt.lever.setAttribute("transform", on ? "" : "rotate(-38 -7 0)");
          rt.cpath.classList.toggle("nd-hide", phase !== 4);
        }
      });
      net.cl.forEach((c, k) => {
        nodes.cl[k].v.textContent = sgn(T[k]);
        nodes.cl[k].v.setAttribute("class", "nd-cl-v " + (T[k] > 0 ? "nd-pos" : "nd-neg"));
        nodes.cl[k].g.classList.toggle("nd-cl-active", routes.some((rt, i) => rt.k === k && (rt.w || decoding) && (swState ? swState[i] : isOnR(rt))));
      });
      net.moves.forEach((m, r) => {
        const on = routes.some((rt, i) => rt.r === r && (swState ? swState[i] : isOnR(rt)));
        // input from the switches as they stand (old switches and new clause outputs between the two kinds of events)
        const v = routes.reduce((s, rt, i) => s + (rt.r === r && rt.w && (swState ? swState[i] : isOnR(rt)) ? rt.w * T[rt.k] : 0), 0);
        nodes.mv[r].v.textContent = on ? `v = ${fmt(v)}` : "no switch on";
        nodes.mv[r].g.classList.toggle("nd-mv-sel", r === sel && phase >= 1);
        nodes.mv[r].g.classList.toggle("nd-mv-fire", r === sel && cur && cur.accept && phase >= 3);
        nodes.mv[r].g.classList.toggle("nd-mv-off", !on);
      });
      nodes.xi.textContent = cur && phase >= 2 ? `ξ = ${fmt(cur.xi)}` : "";
      root.querySelectorAll(".nd-steps span").forEach((s) => s.classList.toggle("on", +s.dataset.p === phase));
    }

    function travel(items, cls, ms, reverse) {
      return new Promise((done) => {
        if (!items.length) return done();
        const dots = items.map(() => el("circle", { r: 4.5, class: "nd-spike " + cls }, svg));
        const t0 = performance.now();
        (function frame(t) {
          const q = Math.min(1, (t - t0) / ms), e = q * q * (3 - 2 * q);
          items.forEach((it, i) => { const pt = it.p.getPointAtLength((reverse ? 1 - e : e) * it.L); dots[i].setAttribute("cx", pt.x); dots[i].setAttribute("cy", pt.y); });
          if (q < 1) requestAnimationFrame(frame); else { dots.forEach((d) => d.remove()); done(); }
        })(t0);
      });
    }

    function setNet(nw) { net = nw; T = clauseOutputs(net, arr); cur = null; phase = 0; buildNet(); buildMap(); paint(); updateState(); }
    const tau = () => parseFloat($(".nd-tau").value);
    function reset() {
      stop(); arr = PAPER[5].slice(); iter = 0; hist = [energyArr(arr)]; hitAt = null; lastEdge = null;
      setNet($(".nd-sc").checked ? WITH_SHORTCUT : BACKBONE);
      say("State 6. Bright connections have their switch on in this state; faint ones are off. Each move neuron holds its input v, half the energy change of its move. Press <b>Next step</b> or <b>Play</b>.");
    }

    async function step() {
      if (busy) return;
      busy = true;
      try {
        if (phase === 0 || phase === 4 || (phase === 2 && !cur.accept)) {
          // 1. encoder
          const r = Math.floor(Math.random() * net.moves.length);
          cur = { r, v: inputOf(net, r, T) };
          phase = 1; paint();
          const act = routes.filter((rt) => rt.r === r && rt.w && isOnR(rt));
          say(`<b>Encoder.</b> ${mvName(net.moves[r])} (${net.moves[r].label}) is proposed. It reads the weighted clause outputs on its ${act.length} active connection${act.length === 1 ? "" : "s"}: v = <b>${fmt(cur.v)}</b>.`);
          await travel(act.map((rt) => ({ p: rt.path, L: rt.L })), "nd-spike-enc", 520, false);
        } else if (phase === 1) {
          // 2. threshold
          const u = Math.random() * 0.98 + 0.01;
          cur.xi = -(tau() / 2) * Math.log(u); cur.accept = cur.v <= cur.xi; phase = 2; paint();
          say(`<b>Threshold.</b> The annealer draws ξ = ${fmt(cur.xi)}. ` + (cur.accept ? "v ≤ ξ: the neuron <b>fires</b>." : "v > ξ: no event; another move is proposed next."));
          if (!cur.accept) { iter++; hist.push(energyArr(arr)); updateState(); }
        } else if (phase === 2) {
          // 3. decoder: spikes back along the same active connections; the clauses they reach toggle
          const back = routes.filter((rt) => rt.r === cur.r && isOnR(rt));
          cur.swBefore = routes.map((rt) => isOnR(rt));
          phase = 3; paint();
          say(`<b>Decoder.</b> The firing neuron sends spikes back along the same ${back.length} active connection${back.length === 1 ? "" : "s"}, without weights, including those of zero-weight auxiliary clauses that the encoder does not use. The clauses they reach toggle${back.length ? "" : " (none: a self-loop)"}.`);
          await travel(back.map((rt) => ({ p: rt.path, L: rt.L })), "nd-spike-dec", 520, true);
          cur.toggled = back.map((rt) => rt.k);
          const from = STATE_ID.get(key(arr));
          cur.toggled.forEach((k) => { T[k] = -T[k]; nodes.cl[k].g.classList.add("nd-flash"); });
          setTimeout(() => nodes.cl.forEach((n) => n.g.classList.remove("nd-flash")), 700);
          arr = applyMove(arr, net.moves[cur.r]);
          if (clauseOutputs(net, arr).some((t, k) => t !== T[k])) console.error("clause outputs out of step with the spins");
          const to = STATE_ID.get(key(arr));
          lastEdge = from !== to ? [from, to] : null;
          iter++; hist.push(energyArr(arr));
          if (to === 1 && hitAt === null) hitAt = iter;
          paint(cur.swBefore); // the switches still hold their old setting until the clause events arrive
          updateState();
          say(`<b>Decoder.</b> ${cur.toggled.map((k) => "T" + sub(k + 1)).join(", ") || "Nothing"} toggled: state ${from} → ${to}, E = ${fmt(energyArr(arr))}.` + (hitAt ? ` Optimum first reached at iteration ${hitAt}.` : ""));
        } else if (phase === 3) {
          // 4. the toggled clauses re-set the switches they control
          const tog = new Set(cur.toggled);
          const hits = routes.filter((rt) => rt.ctrl >= 0 && tog.has(rt.ctrl));
          phase = 4; paint(cur.swBefore);
          hits.forEach((rt) => rt.cpath.classList.add("nd-ctrl2-ev"));
          await travel(hits.map((rt) => ({ p: rt.cpath, L: rt.cL })), "nd-spike-ctl", 560, false);
          hits.forEach((rt) => rt.cpath.classList.remove("nd-ctrl2-ev"));
          paint();
          const changed = hits.filter((rt) => cur.swBefore[routes.indexOf(rt)] !== isOnR(rt)).length;
          say(`<b>Switches.</b> The red lines lead from each clause neuron to the switches it sets. Each toggled clause signals its switches: ${changed} switch${changed === 1 ? "" : "es"} flipped. The bright connections are now those of the new state, and the inputs v are up to date.`);
        }
      } finally { busy = false; }
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; } $(".nd-play").textContent = "Play"; }
    function play(ms) { if (timer) return; $(".nd-play").textContent = "Pause"; step(); timer = setInterval(step, ms || 1250); }
    $(".nd-next").addEventListener("click", () => { stop(); step(); });
    $(".nd-play").addEventListener("click", () => (timer ? stop() : play()));
    $(".nd-reset").addEventListener("click", reset);
    $(".nd-tau").addEventListener("input", () => { $(".nd-tauv").textContent = tau().toFixed(1); });
    $(".nd-sc").addEventListener("change", () => {
      setNet($(".nd-sc").checked ? WITH_SHORTCUT : BACKBONE);
      say($(".nd-sc").checked
        ? `Shortcut G₃ added. Its switches need clause outputs that were not stored, so the compiler appends ${WITH_SHORTCUT.cl.length - BACKBONE.cl.length} auxiliary clauses (purple): a shortcut costs connections, switches and clauses.`
        : "Backbone only: G₁ and G₂, with one auxiliary clause, T₅ = (blue in p₁).");
    });
    $(".nd-full").addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (root.requestFullscreen) root.requestFullscreen();
    });
    reset();
    if (autoplay && "IntersectionObserver" in window && !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
      let user = false;
      root.querySelectorAll("button, input").forEach((b) => b.addEventListener("pointerdown", () => { user = true; }));
      new IntersectionObserver((es) => es.forEach((e) => {
        if (user) return;
        if (e.isIntersecting) play(1100); else stop();
      }), { threshold: 0.5 }).observe(root);
    }
  }

  // ---------------- progress bar of the tutorial ----------------
  function stepper(root) {
    const heads = [...document.querySelectorAll(".md-content h2[id]")].filter((h) => /^\d/.test(h.textContent.trim()));
    if (!heads.length) return;
    root.innerHTML = heads.map((h) => {
      const t = h.textContent.replace(/¶$/, "").trim(), m = t.match(/^(\d+)\.\s*(.*)$/);
      return `<a href="#${h.id}" data-id="${h.id}"><b>${m ? m[1] : ""}</b><span>${root.dataset[`s${m ? m[1] : ""}`] || (m ? m[2] : t)}</span></a>`;
    }).join("");
    const links = [...root.querySelectorAll("a")];
    const mark = () => {
      const y = window.scrollY + 160;
      let cur = 0;
      heads.forEach((h, i) => { if (h.getBoundingClientRect().top + window.scrollY <= y) cur = i; });
      links.forEach((a, i) => { a.classList.toggle("on", i === cur); a.classList.toggle("done", i < cur); });
    };
    window.addEventListener("scroll", mark, { passive: true });
    mark();
  }

  // ---------------- mount ----------------
  const mount = (id, f, arg) => { const r = document.getElementById(id); if (r) f(r, arg); };
  mount("nd-problem", problemWidget);
  mount("nd-pairs", graphWidget, "pairs");
  mount("nd-trees", graphWidget, "trees");
  mount("nd-moves", graphWidget, "moves");
  mount("nd-switches", switchWidget);
  mount("nd-aux", auxWidget);
  mount("nd-machine", machineWidget);
  mount("nd-stepper", stepper);
  window.NimbusDerive = { BACKBONE, WITH_SHORTCUT, E_OF };
})();
