"use strict";
/* Step-by-step mode: the reply is written one character per click. Before each
   draw you see the distribution for the NEXT character, can reshape it with the
   temperature slider, and can override the draw by picking a candidate. */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let stepReq = 0;

function startStep(chat, userMsg, history) {
  const s = state.settings;
  const modelMsg = {
    role: "model", mode: userMsg.mode, text: "", tokens: [], stepped: true,
    params: { temperature: +s.temperature, top_k: +s.topK, top_p: +s.topP },
  };
  chat.messages.push(modelMsg);
  const index = chat.messages.length - 1;
  const row = renderModel(modelMsg, index);
  const thinking = el("span", "thinking");
  thinking.innerHTML = "<i></i><i></i><i></i>";
  row._view.gen.append(thinking);
  $("threadInner").append(row);
  const card = buildStepCard();
  row.after(card);

  state.step = {
    chat, userMsg, modelMsg, history, index, row, view: row._view, card,
    seed: s.seed === "" ? Math.floor(Math.random() * 2 ** 31) : Number(s.seed),
    preview: null, loading: false, animating: false, auto: false, speed: 5,
  };
  state.live = index;
  state.sel = state.hover = null;
  setStreaming(true);
  $("input").blur();
  scrollToBottom(true);
  fetchPreview();
}

async function fetchPreview() {
  const st = state.step;
  if (!st) return;
  const s = state.settings;
  const id = ++stepReq;
  st.loading = true;
  st.card.classList.add("loading");
  const body = {
    history: st.history, prompt: st.userMsg.text, mode: st.userMsg.mode,
    speaker: s.speaker || "ROMEO", system: s.system || "", narrator: s.narrator || DEFAULTS.narrator,
    temperature: +s.temperature, top_k: +s.topK, top_p: +s.topP,
    seed: st.seed, generated: st.modelMsg.text, step: st.modelMsg.tokens.length,
  };
  try {
    const res = await fetch("/api/step", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error ${res.status}`);
    const data = await res.json();
    if (id !== stepReq || state.step !== st) return; // a newer request superseded this one
    if (!st.modelMsg.prompt) applyStepStart(st, data.start);
    st.preview = data;
    st.modelMsg.params.temperature = body.temperature;
    renderStepCard();
  } catch (err) {
    if (state.step === st) { st.modelMsg.error = err.message; finishStep("error"); }
  } finally {
    if (id === stepReq && state.step === st) { st.loading = false; st.card.classList.remove("loading"); }
  }
}

function applyStepStart(st, start) {
  st.modelMsg.prompt = { pinned: start.pinned, context: start.context, cut: start.cut, seed: st.seed };
  st.userMsg.script = start.user_text;
  if (start.dropped?.length) showNotice(`Removed characters the model has never seen: ${start.dropped.map((c) => `“${c}”`).join(" ")}`);
  const prev = st.row.previousElementSibling;
  if (prev) prev.replaceWith(renderUser(st.userMsg));
  if (st.modelMsg.mode === "story") {
    st.modelMsg.speakerName = start.speaker;
    st.modelMsg.prefill = start.prefill;
    applyStoryHeader(st.view, st.modelMsg);
  }
}

/* ------------------------------------------------------------ actions */
async function stepSample({ animate = true } = {}) {
  const st = state.step;
  if (!st || !st.preview || st.loading || st.animating) return false;
  const d = st.preview.draw;
  if (animate) await spinTo(st, d.t);
  if (state.step !== st) return false;
  return commitStep({ t: d.t, p: d.p, raw: d.raw_p, rank: d.rank }, false);
}

function stepPick(i) {
  const st = state.step;
  if (!st || !st.preview || st.loading || st.animating) return;
  stopAuto();
  const [t, p, raw] = st.preview.top[i];
  commitStep({ t, p, raw, rank: i + 1 }, true);
}

function commitStep(choice, manual) {
  const st = state.step;
  const pv = st.preview;
  const m = st.modelMsg;
  const tk = [choice.t, choice.p, choice.raw, choice.rank, pv.entropy, pv.top];
  if (manual) tk.push(1);
  st.view.gen.querySelector(".thinking")?.remove();
  appendChar(st.view, choice.t, choice.p, m.tokens.length, manual);
  m.tokens.push(tk);
  m.text += choice.t;
  st.preview = null;
  refreshInspector();
  scrollToBottom();

  const ended = m.text.endsWith("\n\n") && m.text.trim();
  const minStory = state.info?.story_min_chars ?? 350;
  if (ended && (m.mode === "dialogue" || (m.mode === "story" && m.text.length >= minStory))) { finishStep("end_of_speech"); return false; }
  if (m.tokens.length >= +state.settings.maxTokens) { finishStep("length"); return false; }
  fetchPreview();
  return true;
}

async function stepMany(n) {
  const st = state.step;
  if (!st) return;
  stopAuto();
  for (let k = 0; k < n && state.step === st; k++) {
    while (state.step === st && (st.loading || !st.preview)) await sleep(10);
    if (state.step !== st || !(await stepSample({ animate: false }))) break;
  }
}

function stepUndo() {
  const st = state.step;
  if (!st || !st.modelMsg.tokens.length || st.animating) return;
  stopAuto();
  const m = st.modelMsg;
  m.tokens.pop();
  m.text = m.tokens.map((x) => x[0]).join("");
  st.preview = null; // the old draw was for the next position, not this one
  const row = renderModel(m, st.index);
  st.row.replaceWith(row);
  st.row = row;
  st.view = row._view;
  refreshInspector();
  fetchPreview();
}

function toggleAuto() {
  const st = state.step;
  if (!st) return;
  if (st.auto) { stopAuto(); return; }
  st.auto = true;
  renderStepControls();
  (async () => {
    while (state.step === st && st.auto) {
      while (state.step === st && st.auto && (st.loading || !st.preview || st.animating)) await sleep(10);
      if (state.step !== st || !st.auto) break;
      const t0 = performance.now();
      if (!(await stepSample({ animate: st.speed <= 3 }))) break;
      await sleep(Math.max(0, 1000 / st.speed - (performance.now() - t0)));
    }
    if (state.step === st) { st.auto = false; renderStepControls(); }
  })();
}

function stopAuto() {
  const st = state.step;
  if (st?.auto) { st.auto = false; renderStepControls(); }
}

function finishStep(reason) {
  const st = state.step;
  if (!st) return;
  st.auto = false;
  stepReq++;
  const m = st.modelMsg;
  m.text = m.text.replace(/\n+$/, "");
  if (!m.tokens.length && !m.error) m.error = "No characters were sampled.";
  m.stats = { reason, tokens: m.tokens.length, stepped: true, manual: m.tokens.filter((x) => x[6]).length };
  state.step = null;
  state.live = null;
  st.card.remove();
  setStreaming(false);
  saveChats();
  st.row.replaceWith(renderModel(m, st.index));
  refreshInspector();
  scrollToBottom();
  $("input").focus();
}

/* ------------------------------------------------------------- the card */
function buildStepCard() {
  const card = el("section", "step-card");
  card.setAttribute("aria-label", "Step-by-step sampling");
  card.innerHTML = `
    <div class="step-head">
      <div>
        <div class="step-title">Next character <span class="muted" id="stepIndex"></span></div>
        <div class="step-sub" id="stepSub">Loading the model's odds…</div>
      </div>
      <div class="step-temp">
        <label for="stepTemp"><span class="term" data-term="temperature">Temperature</span></label>
        <input type="range" id="stepTemp" min="0.05" max="2" step="0.05">
        <output id="stepTempOut"></output>
      </div>
    </div>
    <ol class="step-bars" id="stepBars"></ol>
    <p class="step-mask" id="stepMask" hidden>A line break is blocked here so the storyteller keeps talking until the tale reaches ${state.info?.story_min_chars ?? 350} characters.</p>
    <div class="step-controls" id="stepControls"></div>
    <p class="step-hint">Click a candidate to choose it yourself. Keys: <kbd>→</kbd> sample · <kbd>Backspace</kbd> undo · <kbd>Esc</kbd> done</p>`;
  const temp = card.querySelector("#stepTemp");
  temp.value = state.settings.temperature;
  card.querySelector("#stepTempOut").textContent = (+temp.value).toFixed(2);
  let t;
  temp.addEventListener("input", () => {
    state.settings.temperature = +temp.value;
    card.querySelector("#stepTempOut").textContent = (+temp.value).toFixed(2);
    saveSettings();
    syncSettingsUI();
    clearTimeout(t);
    t = setTimeout(fetchPreview, 90);
  });
  card.querySelector("#stepBars").addEventListener("click", (e) => {
    const b = e.target.closest(".sbar");
    if (b) stepPick(+b.dataset.i);
  });
  return card;
}

function renderStepCard() {
  const st = state.step;
  const pv = st.preview;
  const card = st.card;
  card.querySelector("#stepIndex").textContent = `#${st.modelMsg.tokens.length + 1}`;
  const surprise = pv.entropy < 0.5 ? "The model is nearly certain." : pv.entropy < 2 ? "A few real options." : "Wide open: many characters are plausible.";
  card.querySelector("#stepSub").innerHTML =
    `${surprise} <span class="muted">${term("entropy", "Entropy")} ${pv.entropy.toFixed(2)} bits · ${pv.candidates} of 65 characters possible</span>`;
  const max = Math.max(pv.top[0][1], ...pv.top.map((r) => r[2]));
  const bars = card.querySelector("#stepBars");
  bars.replaceChildren();
  pv.top.forEach(([c, p, raw], i) => {
    const li = el("li");
    const b = el("button", "sbar" + (p < 0.0005 ? " off" : ""));
    b.type = "button";
    b.dataset.i = i;
    b.dataset.t = c;
    b.title = `Choose ${vis(c)} yourself (${pct(p)} chance of being drawn)`;
    b.innerHTML = `<span class="ch">${escapeHtml(c === "\n" ? "↵" : c === " " ? "␣" : c)}</span>
      <span class="track"><i class="raw" style="width:${(raw / max) * 100}%"></i><i class="adj" style="width:${(p / max) * 100}%"></i></span>
      <span class="pv">${pct(p)}</span>`;
    li.append(b);
    bars.append(li);
  });
  card.querySelector("#stepMask").hidden = !pv.masked_newline;
  renderStepControls();
}

function renderStepControls() {
  const st = state.step;
  if (!st) return;
  const box = st.card.querySelector("#stepControls");
  const n = st.modelMsg.tokens.length;
  box.innerHTML = `
    <button class="primary-btn step-go" data-act="sample" ${st.auto ? "disabled" : ""}>
      <svg viewBox="0 0 24 24"><path d="M6 4l12 8-12 8z"/></svg>Sample
    </button>
    <button class="ghost-btn" data-act="ten" ${st.auto ? "disabled" : ""}>+10</button>
    <button class="ghost-btn${st.auto ? " on" : ""}" data-act="auto">
      ${st.auto ? '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>Pause' : '<svg viewBox="0 0 24 24"><path d="M5 5l7 7-7 7M12 5l7 7-7 7"/></svg>Auto'}
    </button>
    <select class="speed" aria-label="Auto speed" data-act="speed">
      ${[1, 2, 5, 10, 25].map((v) => `<option value="${v}" ${v === st.speed ? "selected" : ""}>${v}/s</option>`).join("")}
    </select>
    <span class="spacer"></span>
    <button class="ghost-btn" data-act="undo" ${n ? "" : "disabled"}><svg viewBox="0 0 24 24"><path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/></svg>Undo</button>
    <button class="ghost-btn" data-act="done"><svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>Done</button>`;
  box.onclick = (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "sample") stepSample();
    else if (act === "ten") stepMany(10);
    else if (act === "auto") toggleAuto();
    else if (act === "undo") stepUndo();
    else if (act === "done") finishStep("stopped");
  };
  box.querySelector(".speed").onchange = (e) => { st.speed = +e.target.value; };
}

/** Roulette highlight across the candidate bars, landing on the drawn character. */
async function spinTo(st, target) {
  const bars = [...st.card.querySelectorAll(".sbar:not(.off)")];
  const end = bars.findIndex((b) => b.dataset.t === target);
  if (bars.length < 2 || end < 0) return;
  st.animating = true;
  const ticks = 12;
  for (let k = ticks; k >= 0; k--) {
    const i = (((end - k) % bars.length) + bars.length) % bars.length;
    bars.forEach((b, j) => b.classList.toggle("spin", j === i));
    await sleep(28 + (ticks - k) * 9);
  }
  bars[end].classList.add("won");
  await sleep(160);
  st.animating = false;
}

/* --------------------------------------------------------- composer toggle */
(function initStepToggle() {
  const btn = $("stepToggle");
  const sync = () => {
    btn.classList.toggle("on", !!state.ui.stepMode);
    btn.setAttribute("aria-pressed", String(!!state.ui.stepMode));
    btn.title = state.ui.stepMode
      ? "Step-by-step is on: replies are sampled one character at a time"
      : "Step-by-step: sample the reply one character at a time";
  };
  btn.addEventListener("click", () => { state.ui.stepMode = !state.ui.stepMode; saveUI(); sync(); });
  sync();

  document.addEventListener("keydown", (e) => {
    if (!state.step || e.target.closest?.("textarea, input, select")) return;
    if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); stepSample(); }
    else if (e.key === "Backspace") { e.preventDefault(); stepUndo(); }
  });
})();
