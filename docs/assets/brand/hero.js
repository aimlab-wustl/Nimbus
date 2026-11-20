// Home-page hero: the update cycle of the NIMBUS architecture, drawn as neural tissue.
// Move neurons are the large cell bodies; clause neurons are the small cells around them; the curved fibres are the
// connections, bright when their switch is on. One transaction:
//   1. A move neuron is selected at random and compares its input (the energy change of its move) with a random
//      threshold at the current temperature.
//   2. If it fires, spikes travel along its active connections (switch on) to the clause neurons.
//   3. Those clause neurons toggle their sign.
//   4. Every toggled clause sends spikes to the move neurons connected to it (their inputs change) and to the
//      switches it controls (connections turn on or off).
//   5. The next move neuron is selected at random.
(function () {
  "use strict";
  const canvas = document.getElementById("nb-hero-canvas");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SIGN = { 1: [139, 123, 255], [-1]: [34, 211, 238] }; // violet, cyan: the two clause signs
  const AMB = [251, 191, 36], WHITE = [238, 242, 255], SOMA = [34, 211, 238];
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  let W = 0, H = 0, wide = true, clauses = [], moves = [], conns = [], dust = [];
  let ev = null, nextAt = 0, running = false, raf = 0, step = 0, now = 0;
  let seed = 5;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => (t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t));
  const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

  // ---------- the machine ----------
  const isOn = (cn) => clauses[cn.ctrl].T === cn.pol; // switch: on iff its controlling clause has the sign pol
  function input(m) {
    let v = 0;
    for (const cn of m.conns) if (isOn(cn)) v += cn.J * clauses[cn.k].T;
    return v;
  }

  // ---------- layout: cell bodies, their arbors of clause cells, shared cells between neighbours ----------
  function inBlob(u, v) {
    const a = Math.atan2(v - 0.5, u - 0.5);
    const r = 1 + 0.08 * Math.sin(3 * a + 0.7) + 0.06 * Math.sin(5 * a + 2.1);
    return ((u - 0.5) / 0.47) ** 2 + ((v - 0.5) / 0.45) ** 2 < r * r;
  }

  function layout() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = rect.width; H = rect.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed = 5; wide = W > 900;
    const x0 = wide ? W * 0.55 : 0, w = wide ? W * 0.43 : W, y0 = wide ? H * 0.04 : 0, h = wide ? H * 0.84 : H;
    const toXY = (u, v) => ({ x: x0 + u * w, y: y0 + v * h });
    clauses = []; moves = []; conns = [];

    // cell bodies: farthest-point sampling inside the blob
    const nM = wide ? 10 : 8;
    const spots = [];
    for (let i = 0; i < 1500; i++) {
      const u = 0.12 + rnd() * 0.76, v = 0.12 + rnd() * 0.76;
      if (inBlob((u - 0.5) / 0.78 + 0.5, (v - 0.5) / 0.78 + 0.5)) spots.push(toXY(u, v));
    }
    const c0 = toXY(0.5, 0.5);
    moves.push(spots.reduce((b, p) => (d2(p, c0) < d2(b, c0) ? p : b), spots[0]));
    while (moves.length < nM) {
      let best = null, bd = -1;
      for (const p of spots) { const d = Math.min(...moves.map((m) => d2(m, p))); if (d > bd) { bd = d; best = p; } }
      moves.push(best);
    }
    moves = moves.map((p) => ({ x: p.x, y: p.y, bx: p.x, by: p.y, ph: rnd() * 6.28, conns: [], glow: 0, sel: 0, rej: 0,
      v: 0, vShow: 0, stubs: [0, 1, 2].map(() => ({ a: rnd() * 6.28, l: 10 + rnd() * 10, b: rnd() - 0.5 })) }));
    const spacing = Math.sqrt((w * h * 0.6) / nM);

    const addClause = (p) => {
      if (p.x < 4 || p.x > W - 4 || p.y < 4 || p.y > H - 4) return -1;
      if (clauses.some((c) => d2(c, p) < 17 * 17) || moves.some((m) => d2(m, p) < 22 * 22)) return -1;
      clauses.push({ x: p.x, y: p.y, bx: p.x, by: p.y, ph: rnd() * 6.28, T: rnd() < 0.5 ? 1 : -1, flash: 0 });
      return clauses.length - 1;
    };
    const connect = (m, k) => {
      if (k < 0 || m.conns.some((cn) => cn.k === k)) return;
      m.conns.push({ m, k, ctrl: -1, pol: rnd() < 0.5 ? 1 : -1, J: rnd() * 2 - 1, on: 0, flash: 0,
        bend: (rnd() - 0.5) * 0.9, bend2: (rnd() - 0.5) * 0.7 });
    };
    // each cell body grows an arbor of its own clause cells
    for (const m of moves) {
      const n = 4 + Math.floor(rnd() * 2), base = rnd() * 6.28;
      for (let i = 0; i < n; i++) {
        for (let tries = 0; tries < 6; tries++) {
          const a = base + (i * 6.283) / n + (rnd() - 0.5) * 0.7, r = spacing * (0.3 + rnd() * 0.25);
          const k = addClause({ x: m.x + r * Math.cos(a), y: m.y + r * Math.sin(a) });
          if (k >= 0) { connect(m, k); break; }
        }
      }
    }
    // neighbouring cell bodies share a clause cell between them, and sometimes read one of each other's cells
    for (const m of moves) {
      const nb = moves.filter((q) => q !== m).sort((a, b) => d2(a, m) - d2(b, m)).slice(0, 2);
      for (const q of nb) {
        if (d2(q, m) > (1.7 * spacing) ** 2) continue;
        const dx = q.x - m.x, dy = q.y - m.y, j = (rnd() - 0.5) * 0.5;
        const k = addClause({ x: m.x + dx * 0.5 - dy * j, y: m.y + dy * 0.5 + dx * j });
        connect(m, k); connect(q, k);
        if (rnd() < 0.35 && q.conns.length) connect(m, q.conns[Math.floor(rnd() * q.conns.length)].k);
      }
    }
    // a clause cell that ended up with no fibre joins its nearest cell body
    clauses.forEach((c, k) => {
      if (!moves.some((m) => m.conns.some((cn) => cn.k === k))) connect(moves.reduce((b, m) => (d2(m, c) < d2(b, c) ? m : b)), k);
    });
    for (const m of moves) for (const cn of m.conns) {
      // the switch of a connection is controlled by one clause neuron of the same neighbourhood
      const near = clauses.map((c, i) => [d2(c, m), i]).sort((a, b) => a[0] - b[0]);
      cn.ctrl = near[Math.floor(rnd() * Math.min(8, near.length))][1];
      conns.push(cn);
    }
    for (const cn of conns) cn.on = isOn(cn) ? 1 : 0;
    for (const m of moves) { m.v = input(m); m.vShow = m.v; }
    dust = Array.from({ length: wide ? 46 : 28 }, () => ({ x: rnd() * W, y: rnd() * H, r: 0.6 + rnd() * 1.8, a: 0.04 + rnd() * 0.12, s: 0.05 + rnd() * 0.12 }));
  }

  // ---------- one transaction ----------
  const tauAt = (n) => { const p = (n % 90) / 90; return 2.2 * Math.pow(0.12 / 2.2, p); };
  const T_DECIDE = 380, T_FIRE_END = 950, T_TOGGLE = 950, T_EVENTS_END = 1600, T_DONE = 1850, T_REJ = 760;

  function propose(t) {
    const m = moves[Math.floor(rnd() * moves.length)];
    const accept = 2 * m.v <= -tauAt(step++) * Math.log(rnd() + 1e-12);
    ev = { m, t0: t, accept, phase: 0, open: m.conns.filter(isOn), toggled: [], spikesIn: [], ctrlHits: [] };
    m.sel = 1;
  }

  function advance(t) {
    if (!ev) return;
    const dt = t - ev.t0, m = ev.m;
    if (!ev.accept) {
      if (dt > T_DECIDE && ev.phase === 0) { ev.phase = 1; m.rej = 1; }
      if (dt > T_REJ) { ev = null; nextAt = t + 120; }
      return;
    }
    if (dt > T_DECIDE && ev.phase === 0) { ev.phase = 1; m.glow = 1; }
    if (dt > T_TOGGLE && ev.phase === 1) {
      ev.phase = 2;
      // the active connections were latched before anything changed; toggle those clauses together
      for (const cn of ev.open) { const c = clauses[cn.k]; c.T = -c.T; c.flash = 1; ev.toggled.push(cn.k); }
      const tog = new Set(ev.toggled);
      for (const cn of conns) {
        if (tog.has(cn.k)) ev.spikesIn.push(cn);   // inputs that read a toggled clause
        if (tog.has(cn.ctrl)) ev.ctrlHits.push(cn); // switches a toggled clause controls
      }
    }
    if (dt > T_EVENTS_END && ev.phase === 2) {
      ev.phase = 3;
      for (const cn of ev.ctrlHits) if ((isOn(cn) ? 1 : 0) !== Math.round(cn.on)) cn.flash = 1;
      for (const q of moves) q.v = input(q);
    }
    if (dt > T_DONE) { ev = null; nextAt = t + 150; }
  }

  // ---------- drawing ----------
  function drift(n, t) {
    n.x = n.bx + 2.4 * Math.sin(t * 0.00055 + n.ph);
    n.y = n.by + 2.4 * Math.cos(t * 0.00047 + n.ph * 1.3);
  }
  // the fibre of a connection: a cubic curve from the cell body to the clause cell
  function curve(cn) {
    const m = cn.m, c = clauses[cn.k];
    const dx = c.x - m.x, dy = c.y - m.y, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
    const p1 = { x: m.x + L * 0.35 * (ux * Math.cos(cn.bend) - uy * Math.sin(cn.bend)), y: m.y + L * 0.35 * (uy * Math.cos(cn.bend) + ux * Math.sin(cn.bend)) };
    const p2 = { x: c.x - L * 0.3 * ux - uy * L * 0.2 * cn.bend2, y: c.y - L * 0.3 * uy + ux * L * 0.2 * cn.bend2 };
    return [m, p1, p2, c];
  }
  function at(P, t) {
    const s = 1 - t, a = s * s * s, b = 3 * s * s * t, c = 3 * s * t * t, d = t * t * t;
    return { x: a * P[0].x + b * P[1].x + c * P[2].x + d * P[3].x, y: a * P[0].y + b * P[1].y + c * P[2].y + d * P[3].y };
  }
  function fibre(P, w0, w1, fill) {
    const N = 12, L = [], R = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N, p = at(P, t), q = at(P, Math.min(1, t + 0.02)), o = at(P, Math.max(0, t - 0.02));
      let tx = q.x - o.x, ty = q.y - o.y; const n = Math.hypot(tx, ty) || 1; tx /= n; ty /= n;
      const w = lerp(w0, w1, t) / 2;
      L.push([p.x - ty * w, p.y + tx * w]); R.push([p.x + ty * w, p.y - tx * w]);
    }
    ctx.fillStyle = fill; ctx.beginPath(); ctx.moveTo(L[0][0], L[0][1]);
    for (const p of L) ctx.lineTo(p[0], p[1]);
    for (let i = R.length - 1; i >= 0; i--) ctx.lineTo(R[i][0], R[i][1]);
    ctx.closePath(); ctx.fill();
  }
  function comet(P, t, col, r, rev) {
    for (let i = 6; i >= 0; i--) {
      const s = Math.max(0, Math.min(1, t - i * 0.035)), p = at(P, rev ? 1 - s : s);
      ctx.fillStyle = rgba(col, (1 - i / 7) * (i ? 0.45 : 1));
      ctx.beginPath(); ctx.arc(p.x, p.y, r * (1 - i * 0.09), 0, 6.2832); ctx.fill();
    }
  }
  function glowDot(x, y, r, col, a) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, rgba(col, a)); g.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    const dt = ev ? t - ev.t0 : 0;
    for (const p of dust) {
      p.y -= p.s; if (p.y < -4) { p.y = H + 4; p.x = Math.random() * W; }
      ctx.fillStyle = `rgba(199,210,254,${p.a})`; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
    }
    for (const n of moves) drift(n, t);
    for (const n of clauses) drift(n, t);
    const curves = new Map(conns.map((cn) => [cn, curve(cn)]));

    // fibres: bright and thick when the switch is on, faint and thin when off; a changing switch flashes amber
    for (const cn of conns) {
      const target = isOn(cn) ? 1 : 0;
      if (!(ev && ev.phase === 2 && ev.ctrlHits.includes(cn))) cn.on += (target - cn.on) * 0.1;
      const c = clauses[cn.k], P = curves.get(cn);
      const g = ctx.createLinearGradient(cn.m.x, cn.m.y, c.x, c.y);
      g.addColorStop(0, rgba(SOMA, lerp(0.1, 0.55, cn.on))); g.addColorStop(1, rgba(SIGN[c.T], lerp(0.08, 0.5, cn.on)));
      fibre(P, lerp(1.4, 3.6, cn.on), lerp(0.5, 1.1, cn.on), g);
      if (cn.flash > 0) { fibre(P, 4.2, 1.6, rgba(AMB, 0.75 * cn.flash)); cn.flash = Math.max(0, cn.flash - 0.022); }
    }
    if (ev && ev.accept) {
      // the move neuron fires: spikes along its active fibres to the clause cells
      if (dt > T_DECIDE && dt < T_FIRE_END + 160) {
        const p = ease((dt - T_DECIDE) / (T_FIRE_END - T_DECIDE));
        for (const cn of ev.open) comet(curves.get(cn), p, WHITE, 3.2, false);
      }
      // toggled clauses send spikes to the cell bodies that read them and to the switches they control
      if (ev.phase >= 2 && dt < T_EVENTS_END + 80) {
        const p = ease((dt - T_TOGGLE) / (T_EVENTS_END - T_TOGGLE));
        for (const cn of ev.spikesIn) comet(curves.get(cn), p, AMB, 2.4, true);
        for (const cn of ev.ctrlHits) {
          const c = clauses[cn.ctrl], s = at(curves.get(cn), 0.5);
          if (p < 1) glowDot(lerp(c.x, s.x, p), lerp(c.y, s.y, p), 6, AMB, 0.9);
        }
      }
    }
    // clause cells
    for (const c of clauses) {
      glowDot(c.x, c.y, 13, SIGN[c.T], 0.28);
      if (c.flash > 0) {
        ctx.strokeStyle = rgba(WHITE, c.flash); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(c.x, c.y, 5 + (1 - c.flash) * 13, 0, 6.2832); ctx.stroke();
        c.flash = Math.max(0, c.flash - 0.02);
      }
      ctx.fillStyle = rgba(SIGN[c.T], 1); ctx.beginPath(); ctx.arc(c.x, c.y, 4.4, 0, 6.2832); ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.beginPath(); ctx.arc(c.x - 1.3, c.y - 1.3, 1.3, 0, 6.2832); ctx.fill();
    }
    // move neurons: cell bodies with short dendrites; brighter when their input favours the move
    for (const m of moves) {
      m.vShow += (m.v - m.vShow) * 0.08;
      const fav = Math.max(0, Math.min(1, 0.5 - m.vShow / 2.5));
      for (const s of m.stubs) {
        const a = s.a + 0.15 * Math.sin(t * 0.0007 + m.ph), ex = m.x + Math.cos(a) * (12 + s.l), ey = m.y + Math.sin(a) * (12 + s.l);
        fibre([m, { x: m.x + Math.cos(a + s.b) * 10, y: m.y + Math.sin(a + s.b) * 10 }, { x: ex - Math.cos(a) * 5, y: ey - Math.sin(a) * 5 }, { x: ex, y: ey }],
          3, 0.4, rgba(SOMA, 0.35 + 0.3 * fav));
      }
      glowDot(m.x, m.y, 34 + 18 * m.glow, SOMA, 0.18 + 0.25 * fav + 0.5 * m.glow);
      const g = ctx.createRadialGradient(m.x - 3, m.y - 3, 1, m.x, m.y, 11);
      g.addColorStop(0, rgba(WHITE, 0.95)); g.addColorStop(0.35, rgba(SOMA, lerp(0.55, 1, fav))); g.addColorStop(1, rgba([14, 116, 144], 1));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(m.x, m.y, 10.5, 0, 6.2832); ctx.fill();
      m.glow = Math.max(0, m.glow - 0.016);
      if (m.sel > 0) { // selected: a ring closes while the neuron compares its input with the threshold
        const prog = ev && ev.m === m ? Math.min(1, dt / T_DECIDE) : 1;
        ctx.strokeStyle = m.rej > 0 ? `rgba(148,163,184,${m.sel})` : rgba(WHITE, m.sel); ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.arc(m.x, m.y, 17, -1.5708, -1.5708 + 6.2832 * prog); ctx.stroke();
        if (!ev || ev.m !== m) m.sel = Math.max(0, m.sel - 0.05);
      }
      if (m.rej > 0) {
        ctx.fillStyle = `rgba(15,16,38,${0.55 * m.rej})`; ctx.beginPath(); ctx.arc(m.x, m.y, 10.5, 0, 6.2832); ctx.fill();
        m.rej = Math.max(0, m.rej - 0.03);
      }
    }
  }

  function loop(t) {
    if (!running) return;
    now = t;
    if (!ev && t > nextAt) propose(t);
    advance(t);
    draw(t);
    raf = requestAnimationFrame(loop);
  }
  function start() { if (!running && !reduce) { running = true; raf = requestAnimationFrame(loop); } }
  function stop() { running = false; cancelAnimationFrame(raf); }

  layout(); draw(0);
  window.addEventListener("resize", () => { layout(); draw(now); });
  if (reduce) return;
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => es.forEach((e) => (e.isIntersecting ? start() : stop()))).observe(canvas);
  } else start();
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
})();
