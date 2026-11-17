"use strict";
/* Model X-ray: follows one next-character prediction through every stage of
   the network (embeddings, 6 layers of attention + feed-forward, output). */

const XRAY_SAMPLE = "ROMEO:\nBut soft, what light through yonder window br";
const term = (key, text) => `<span class="term" data-term="${key}">${text}</span>`;
const vis = (c) => (c === "\n" ? "↵" : c === " " ? "␣" : c);
const q = (c) => `“${vis(c)}”`;
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const qh = (c) => escapeHtml(q(c));

const xr = { req: null, data: null, tip: null };

function openXray(opts = {}) {
  let root = $("xray");
  if (!root) root = buildXray();
  root.hidden = false;
  document.body.classList.add("modal-open");
  const s = state.settings;
  xr.req = {
    pinned: opts.pinned ?? "",
    body: opts.body ?? XRAY_SAMPLE,
    target: opts.target ?? "",
    temperature: opts.temperature ?? +s.temperature,
    top_k: opts.top_k ?? +s.topK,
    top_p: opts.top_p ?? +s.topP,
  };
  xr.fromReply = !!opts.fromReply;
  $("xrayInput").value = xr.req.pinned + xr.req.body;
  $("xraySource").textContent = opts.fromReply
    ? `From your conversation: character ${opts.index + 1} of the reply.`
    : "Type any text and see how the model predicts what comes next.";
  runXray();
}

function closeXray() {
  $("xray").hidden = true;
  xr.tip.hidden = true;
  if (document.getElementById("glossary")?.hidden !== false) document.body.classList.remove("modal-open");
}

function buildXray() {
  const root = el("div", "modal xray");
  root.id = "xray";
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-label", "Model X-ray");
  root.innerHTML = `
    <div class="xray-panel">
      <header class="xray-head">
        <div>
          <h2>Model X-ray <span class="muted">· how one character gets predicted</span></h2>
          <p class="muted small" id="xraySource"></p>
        </div>
        <div class="xray-head-actions">
          <button class="ghost-btn" id="xrayGlossary">
            <svg viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5M9 7h6"/></svg>Glossary
          </button>
          <button class="icon-btn" id="xrayClose" aria-label="Close X-ray"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
        </div>
      </header>
      <form class="xray-form" id="xrayForm">
        <label for="xrayInput" class="small muted">Text the model sees (it predicts the character that comes after the last one)</label>
        <div class="xray-form-row">
          <textarea id="xrayInput" rows="2" spellcheck="false"></textarea>
          <button type="submit" class="primary-btn">Predict next</button>
        </div>
      </form>
      <nav class="xray-flow" id="xrayFlow"></nav>
      <div class="xray-body" id="xrayBody"></div>
    </div>`;
  document.body.append(root);
  xr.tip = el("div", "tooltip xray-tip");
  xr.tip.hidden = true;
  document.body.append(xr.tip);

  root.querySelector("#xrayClose").addEventListener("click", closeXray);
  root.querySelector("#xrayGlossary").addEventListener("click", () => openGlossary());
  root.addEventListener("click", (e) => { if (e.target === root) closeXray(); });
  root.querySelector("#xrayForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("xrayInput").value;
    if (!text) return;
    if (text === xr.req.pinned + xr.req.body) { runXray(); return; }
    xr.req = { ...xr.req, pinned: "", body: text, target: "" };
    $("xraySource").textContent = "Your own text.";
    runXray();
  });
  $("xrayInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); root.querySelector("#xrayForm").requestSubmit(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !root.hidden && document.getElementById("glossary")?.hidden !== false) closeXray();
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => !root.hidden && xr.data && renderXray(xr.data), 150); });
  return root;
}

async function runXray() {
  const body = $("xrayBody");
  body.innerHTML = `<div class="xray-loading"><span class="thinking"><i></i><i></i><i></i></span> Tracing the prediction…</div>`;
  $("xrayFlow").replaceChildren();
  try {
    const res = await fetch("/api/xray", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(xr.req),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error ${res.status}`);
    xr.data = await res.json();
    renderXray(xr.data);
    $("xray").querySelector(".xray-panel").scrollTop = 0;
  } catch (err) {
    body.innerHTML = "";
    body.append(el("p", "err-text", err.message));
  }
}

/* ------------------------------------------------------------- render */
function renderXray(d) {
  const body = $("xrayBody");
  body.replaceChildren();
  const target = d.target;
  const chosenKnown = !!xr.req.target;

  // Overview chips
  const flow = $("xrayFlow");
  flow.replaceChildren();
  const steps = [["x-text", "Text"], ["x-embed", "Embedding"],
    ...d.layers.map((_, i) => [`x-l${i + 1}`, `Layer ${i + 1}`]), ["x-out", "Probabilities"], ["x-pick", "Pick"]];
  steps.forEach(([id, label], i) => {
    if (i) flow.append(el("span", "flow-arrow", "→"));
    const b = el("button", "flow-chip" + (label.startsWith("Layer") ? " layer" : ""), label);
    b.addEventListener("click", () => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }));
    flow.append(b);
  });

  // Hero: answer + guess-by-layer chart
  const hero = el("section", "x-hero");
  const finalP = d.final.target_raw;
  hero.innerHTML = `
    <div class="x-hero-text">
      <p class="x-kicker">After reading <b>${escapeHtml(tail(d.window, 28))}</b></p>
      <h3>The model ${chosenKnown ? "wrote" : "predicts"} <span class="x-big">${qh(target)}</span></h3>
      <p class="muted">${chosenKnown ? "It gave this character" : "Its top choice, with"} <b>${pct(finalP)}</b> probability.
        Below, follow how that number was built up, ${term("layer", "layer")} by layer.</p>
    </div>`;
  const chartCard = el("div", "x-chart-card");
  chartCard.innerHTML = `<div class="x-chart-title">Probability of ${qh(target)} after each stage <span class="muted">(the model's ${term("logit-lens", "best guess so far")})</span></div>`;
  const chart = el("div", "x-chart");
  chartCard.append(chart);
  hero.append(chartCard);
  body.append(hero);
  drawLensChart(chart, d);

  // 1. Text in
  const T = d.window.length;
  const showN = Math.min(T, window.innerWidth < 700 ? 14 : 30);
  const sec1 = stage("x-text", 1, "Text in", `The model reads the text as ${term("token", "tokens")}: one per character.
    Each is swapped for its ${term("token-id", "token ID")} from a ${term("vocabulary", "vocabulary")} of ${d.dims.vocab}.
    It can see the last ${d.dims.block} characters (its ${term("context", "context window")}); this text uses ${T}.`);
  const chips = el("div", "tok-row");
  if (T > showN) chips.append(el("span", "tok-more", `+${T - showN} earlier`));
  for (let i = T - showN; i < T; i++) {
    const c = d.window[i];
    const chip = el("span", "tok" + (i < d.pinned_len ? " pinned" : "") + (i === T - 1 ? " last" : ""));
    chip.append(el("b", null, vis(c)), el("small", null, String(tokId(c))));
    chip.title = i < d.pinned_len ? "Part of the system prompt" : `Position ${i + 1}`;
    chips.append(chip);
  }
  chips.append(el("span", "tok next", "?"));
  sec1.body.append(chips);
  sec1.body.append(note(`The last character, ${qh(d.last)}, is where the prediction happens. Everything before it is context it can look back at.`));
  body.append(sec1.node);

  // 2. Embedding
  const sec2 = stage("x-embed", 2, "Embedding", `Each token ID is turned into a ${term("vector", "vector")}: a list of ${d.dims.n_embd} numbers the model can calculate with.
    Two lists are added together: what the character is, and where it sits.`);
  const legend = el("div", "div-legend");
  legend.innerHTML = `<span>negative</span><i></i><span>0</span><i class="pos"></i><span>positive</span>`;
  sec2.body.append(legend);
  sec2.body.append(barcodeRow(`${term("token-embedding", "Token embedding")} of ${qh(d.last)}`, d.vectors.tok_emb));
  sec2.body.append(el("div", "op", "+"));
  sec2.body.append(barcodeRow(`${term("position-embedding", "Position embedding")} for position ${d.position + 1}`, d.vectors.pos_emb));
  sec2.body.append(el("div", "op", "="));
  sec2.body.append(barcodeRow(`Starting vector that enters layer 1`, d.vectors.x0));
  sec2.body.append(note(`Already at this point the model has a guess: ${guessSentence(d.stages[0], target)} That comes only from knowing the last character was ${qh(d.last)}.`));
  body.append(sec2.node);

  // 3. Layers
  const cols = attentionColumns();
  d.layers.forEach((L, li) => {
    const before = d.stages[li], after = d.stages[li + 1];
    const s = stage(`x-l${li + 1}`, li + 3, `Layer ${li + 1} of ${d.layers.length}`,
      li === 0
        ? `Each ${term("layer", "layer")} has two parts. First ${term("attention", "attention")}: ${d.dims.n_head} ${term("head", "heads")} each look back over earlier characters and pull in what's useful.
           Then the ${term("feed-forward", "feed-forward network")}: ${d.dims.ffn.toLocaleString()} ${term("neuron", "neurons")} process the result. Both add their findings to the ${term("residual", "running vector")}.`
        : layerBlurb(li));
    s.node.classList.add("x-layer");
    const grid = el("div", "layer-grid");

    const left = el("div", "attn-wrap");
    left.append(el("div", "sub-title", `Where each head looked`));
    left.append(attentionGrid(d, L, li, cols));
    const key = el("div", "attn-key");
    key.innerHTML = `<span>0%</span><i></i><span>100% of a head's attention</span>`;
    left.append(key);
    left.append(el("p", "hint", "Each row is one head; each column a recent character. The outlined column is the character being read; \"older\" adds up everything further back."));
    grid.append(left);

    const right = el("div", "layer-side");
    right.append(el("div", "sub-title", "Best guess after this layer"));
    right.append(guessBars(after, target, before));
    const stats = el("div", "layer-stats");
    const maxNorm = Math.max(...d.layers.flatMap((x) => [x.attn_norm, x.ffn_norm]));
    stats.innerHTML = `
      <div class="sub-title">How much each part changed the vector</div>
      ${normBar("Attention", L.attn_norm, maxNorm, "attention")}
      ${normBar("Feed-forward", L.ffn_norm, maxNorm, "feed-forward")}
      <div class="stat-line"><span>${term("neuron", "Neurons")} firing</span><b>${(L.active * 100).toFixed(1)}%</b><span class="muted">of ${d.dims.ffn.toLocaleString()}</span></div>`;
    right.append(stats);
    grid.append(right);
    s.body.append(grid);
    s.body.append(note(layerStory(d, L, li, target)));
    s.body.append(barcodeRow(`Vector after layer ${li + 1}`, L.resid, true));
    body.append(s.node);
  });

  // 9. Output
  const out = stage("x-out", d.layers.length + 3, "Scores → probabilities",
    `A final ${term("layernorm", "layer norm")}, then the vector is compared with all ${d.dims.vocab} characters to give each a ${term("logits", "score (logit)")}.
     ${term("softmax", "Softmax")} turns the scores into a ${term("probability", "probability distribution")} that adds up to 100%.`);
  const table = el("div", "out-table");
  table.innerHTML = `<div class="out-row head"><span>Char</span><span>${term("logits", "Score")}</span><span>${term("raw", "Raw probability")}</span><span>${term("raw", "After settings")}</span></div>`;
  const maxP = d.final.top[0][1];
  d.final.top.forEach(([c, p, adj], i) => {
    const row = el("div", "out-row" + (c === target ? " chosen" : ""));
    row.innerHTML = `<span class="ch">${escapeHtml(vis(c))}</span><span class="mono">${d.final.logits_top[i].toFixed(2)}</span>
      <span class="out-bar"><i style="width:${(p / maxP) * 100}%"></i><b>${pct(p)}</b></span><span class="mono">${pct(adj)}</span>`;
    table.append(row);
  });
  out.body.append(table);
  out.body.append(note(`Uncertainty (${term("entropy", "entropy")}): <b>${d.final.entropy.toFixed(2)} bits</b>. ${
    d.final.entropy < 1 ? "The model is quite sure." : d.final.entropy < 2.5 ? "A few real options." : "Many plausible options."}`));
  body.append(out.node);

  // 10. Pick
  const st = d.settings;
  const pick = stage("x-pick", d.layers.length + 4, "Pick one",
    `The settings reshape the odds (${term("temperature", "temperature")} ${st.temperature}${st.top_k ? `, ${term("top-k", "top-k")} ${st.top_k}` : ""}${st.top_p < 1 ? `, ${term("top-p", "top-p")} ${st.top_p}` : ""}),
     then ${term("sampling", "a weighted random draw")} picks one character.`);
  const rank = d.final.top.findIndex((r) => r[0] === target) + 1;
  pick.body.append(note(chosenKnown
    ? `It drew <b>${qh(target)}</b>, which had a <b>${pct(d.final.target_adjusted)}</b> chance after the settings${rank ? `; ${term("rank", "rank")} #${rank}` : ""}.
       ${rank > 1 ? "A less likely character won the draw. That's sampling at work, and why replies vary." : "The favourite won the draw."}`
    : `If you sent this text, ${qh(target)} would be picked about <b>${pct(d.final.target_adjusted)}</b> of the time. The rest of the time it would be another character from the table above.`));
  body.append(pick.node);
}

/* ------------------------------------------------------------ helpers */
function tail(s, n) { const t = s.length > n ? "…" + s.slice(-n) : s; return escapeHtml(t.replace(/\n/g, " ↵ ")); }
function tokId(c) { return state.info ? state.info.vocab.indexOf(c) : ""; }
function note(html) { const p = el("p", "x-note"); p.innerHTML = html; return p; }

function stage(id, n, title, blurb) {
  const node = el("section", "x-stage");
  node.id = id;
  const head = el("div", "x-stage-head");
  head.innerHTML = `<span class="x-num">${n}</span><div><h3>${title}</h3><p>${blurb}</p></div>`;
  const body = el("div", "x-stage-body");
  node.append(head, body);
  return { node, body };
}

function guessSentence(stageData, target) {
  const [c, p] = stageData.top[0];
  return c === target
    ? `${qh(c)} at ${pct(p)}.`
    : `${qh(c)} at ${pct(p)} (${qh(target)} has ${pct(stageData.target_p)}).`;
}

function layerBlurb(li) {
  return [
    "",
    "Same two parts again, working on the improved vector from the layer before.",
    "Middle layers tend to combine clues: which word is being spelled, what came just before it.",
    "Each layer only adds a correction; the earlier information is still there.",
    "Later heads often look much further back, at names, rhymes or the start of the line.",
    "The last layer makes the final adjustments before the vector is turned into probabilities.",
  ][Math.min(li, 5)];
}

function layerStory(d, L, li, target) {
  const T = d.window.length;
  const avg = new Array(T).fill(0);
  L.attn.forEach((h) => h.forEach((w, j) => (avg[j] += w / L.attn.length)));
  let j = 0;
  avg.forEach((w, k) => { if (w > avg[j]) j = k; });
  const back = T - 1 - j;
  const where = back === 0 ? `${qh(d.window[j])} itself (the current character)` : `${qh(d.window[j])}, ${back} character${back > 1 ? "s" : ""} back`;
  const before = d.stages[li], after = d.stages[li + 1];
  const b1 = before.top[0][0], a1 = after.top[0][0];
  let change;
  if (a1 !== b1) change = `It <b>changed its mind</b>: from ${qh(b1)} to ${qh(a1)} (${pct(after.top[0][1])}).`;
  else {
    const d0 = before.top[0][1], d1 = after.top[0][1];
    change = `It still favours ${qh(a1)}, ${d1 > d0 + 0.02 ? "now more sure" : d1 < d0 - 0.02 ? "now less sure" : "about as sure"} (${pct(d0)} → ${pct(d1)}).`;
  }
  const tChange = after.target_p - before.target_p;
  const tPart = a1 === target ? "" : ` ${qh(target)} ${tChange >= 0 ? "rose" : "fell"} to ${pct(after.target_p)}.`;
  return `On average the heads looked most at ${where} (${pct(avg[j])} of attention). ${change}${tPart}`;
}

function guessBars(s, target, before) {
  const box = el("div", "guess-bars");
  const max = s.top[0][1];
  const prevTop = before?.top[0][0];
  s.top.forEach(([c, p]) => {
    const row = el("div", "gbar" + (c === target ? " target" : ""));
    row.innerHTML = `<span class="ch">${escapeHtml(vis(c))}</span><span class="track"><i style="width:${Math.max(1, (p / max) * 100)}%"></i></span><span class="pv">${pct(p)}</span>`;
    box.append(row);
  });
  if (!s.top.some(([c]) => c === target)) {
    const row = el("div", "gbar target dim");
    row.innerHTML = `<span class="ch">${escapeHtml(vis(target))}</span><span class="track"><i style="width:${Math.max(1, (s.target_p / max) * 100)}%"></i></span><span class="pv">${pct(s.target_p)}</span>`;
    box.append(row);
  }
  return box;
}

function normBar(label, v, max, key) {
  return `<div class="norm-row"><span>${term(key, label)}</span><span class="norm-track"><i style="width:${(v / max) * 100}%"></i></span><b>${v.toFixed(1)}</b></div>`;
}

/* ---------------------------------------------------------- attention */
function attentionColumns() {
  const w = Math.min(window.innerWidth, 1100);
  return Math.max(8, Math.min(28, Math.floor((w * 0.55 - 120) / 20)));
}

function attentionGrid(d, L, li, cols) {
  const T = d.window.length;
  const n = Math.min(cols, T);
  const start = T - n;
  const grid = el("div", "attn-grid");
  grid.style.gridTemplateColumns = `40px ${T > n ? "34px " : ""}repeat(${n}, minmax(12px, 1fr))`;
  // header row
  grid.append(el("span", "ag-corner"));
  if (T > n) grid.append(el("span", "ag-col earlier", "older"));
  for (let j = start; j < T; j++) grid.append(el("span", "ag-col" + (j === T - 1 ? " last" : ""), vis(d.window[j])));
  L.attn.forEach((h, hi) => {
    grid.append(el("span", "ag-row", `H${hi + 1}`));
    if (T > n) {
      const older = h.slice(0, start).reduce((a, b) => a + b, 0);
      grid.append(attnCell(older, `Head ${hi + 1}: ${pct(older)} of its attention went to the ${start} older characters not shown`, "earlier"));
    }
    for (let j = start; j < T; j++) {
      const back = T - 1 - j;
      grid.append(attnCell(h[j], `Layer ${li + 1}, head ${hi + 1} → ${q(d.window[j])} ${back ? `(${back} back)` : "(itself)"}: ${pct(h[j])}`, j === T - 1 ? "last" : ""));
    }
  });
  grid.addEventListener("mousemove", (e) => {
    const c = e.target.closest(".ag-cell");
    if (!c) { xr.tip.hidden = true; return; }
    showXTip(c.dataset.tip, e.clientX, e.clientY);
  });
  grid.addEventListener("mouseleave", () => (xr.tip.hidden = true));
  return grid;
}

function attnCell(w, tip, cls) {
  const c = el("span", "ag-cell" + (cls ? " " + cls : ""));
  c.style.setProperty("--w", Math.pow(Math.min(1, w), 0.5).toFixed(3));
  c.dataset.tip = tip;
  return c;
}

function showXTip(text, x, y) {
  const t = xr.tip;
  t.textContent = text;
  t.hidden = false;
  const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = `${Math.min(x + 12, window.innerWidth - w - 8)}px`;
  t.style.top = `${y + 16 + h > window.innerHeight ? y - h - 8 : y + 16}px`;
}

/* ------------------------------------------------------------ barcode */
function barcodeRow(labelHtml, values, compact = false) {
  const row = el("div", "barcode-row" + (compact ? " compact" : ""));
  const label = el("div", "barcode-label");
  label.innerHTML = labelHtml;
  const canvas = el("canvas", "barcode");
  row.append(label, canvas);
  requestAnimationFrame(() => drawBarcode(canvas, values));
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    const i = Math.min(values.length - 1, Math.floor(((e.clientX - r.left) / r.width) * values.length));
    showXTip(`Number ${i + 1} of ${values.length}: ${values[i] >= 0 ? "+" : ""}${values[i].toFixed(3)}`, e.clientX, e.clientY);
  });
  canvas.addEventListener("mouseleave", () => (xr.tip.hidden = true));
  return row;
}

function cssColor(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = v.match(/^#([0-9a-f]{6})$/i);
  return m ? [0, 2, 4].map((k) => parseInt(m[1].slice(k, k + 2), 16)) : [128, 128, 128];
}

function drawBarcode(canvas, values) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const neg = cssColor("--div-neg"), pos = cssColor("--div-pos"), mid = cssColor("--div-mid");
  const sorted = values.map(Math.abs).sort((a, b) => a - b);
  const scale = sorted[Math.floor(sorted.length * 0.98)] || 1;
  const cw = w / values.length;
  values.forEach((v, i) => {
    const t = Math.min(1, Math.abs(v) / scale);
    const end = v < 0 ? neg : pos;
    const c = mid.map((m, k) => Math.round(m + (end[k] - m) * t));
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(i * cw, 0, Math.ceil(cw), h);
  });
}

/* ------------------------------------------------------- lens chart */
function drawLensChart(box, d) {
  const stages = d.stages;
  const W = Math.max(280, box.clientWidth || 600), H = 230;
  const pad = { l: 40, r: 18, t: 18, b: 58 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const x = (i) => pad.l + (i / (stages.length - 1)) * iw;
  const y = (p) => pad.t + (1 - p) * ih;
  const labels = stages.map((_, i) => (i === 0 ? "Embed" : `L${i}`));
  let svg = `<svg class="lens-svg" style="width:${W}px;height:${H}px" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Probability of the predicted character after each stage">`;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svg += `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y(g)}" y2="${y(g)}"/>`;
    svg += `<text class="axis" x="${pad.l - 8}" y="${y(g) + 4}" text-anchor="end">${g * 100}%</text>`;
  }
  const pts = stages.map((s, i) => [x(i), y(s.target_p)]);
  svg += `<path class="area" d="M${pts[0][0]},${y(0)} ${pts.map((p) => `L${p[0]},${p[1]}`).join(" ")} L${pts.at(-1)[0]},${y(0)} Z"/>`;
  svg += `<polyline class="line" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
  pts.forEach(([px, py], i) => {
    const s = stages[i];
    const top = s.top[0][0];
    svg += `<circle class="dot${top === d.target ? " hit" : ""}" cx="${px}" cy="${py}" r="5"/>`;
    svg += `<text class="axis" x="${px}" y="${H - pad.b + 18}" text-anchor="middle">${labels[i]}</text>`;
    svg += `<text class="guess${top === d.target ? " hit" : ""}" x="${px}" y="${H - pad.b + 40}" text-anchor="middle">${escapeHtml(vis(top))}</text>`;
  });
  const last = pts.at(-1);
  svg += `<text class="val" x="${last[0] - 8}" y="${last[1] - 10}" text-anchor="end">${pct(stages.at(-1).target_p)}</text>`;
  svg += `<text class="axis legend-txt" x="${pad.l - 8}" y="${H - pad.b + 40}" text-anchor="end">top</text>`;
  stages.forEach((_, i) => {
    const x0 = i === 0 ? pad.l - 10 : (x(i - 1) + x(i)) / 2, x1 = i === stages.length - 1 ? W : (x(i) + x(i + 1)) / 2;
    svg += `<rect class="hit-area" data-i="${i}" x="${x0}" y="0" width="${x1 - x0}" height="${H}"/>`;
  });
  svg += "</svg>";
  box.innerHTML = svg;
  box.querySelectorAll(".hit-area").forEach((r) => {
    r.addEventListener("mousemove", (e) => {
      const s = stages[+r.dataset.i];
      showXTip(`${+r.dataset.i === 0 ? "After embedding" : `After layer ${r.dataset.i}`}: ${q(d.target)} ${pct(s.target_p)}. Top guesses: ${s.top.slice(0, 3).map(([c, p]) => `${q(c)} ${pct(p)}`).join(", ")}`, e.clientX, e.clientY);
    });
    r.addEventListener("mouseleave", () => (xr.tip.hidden = true));
    r.addEventListener("click", () => document.getElementById(+r.dataset.i === 0 ? "x-embed" : `x-l${r.dataset.i}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  });
  const caption = el("p", "x-chart-caption");
  caption.innerHTML = `The row under the axis shows the model's top guess at each stage. Click a point to jump to that layer.`;
  box.append(caption);
}
