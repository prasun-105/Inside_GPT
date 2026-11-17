"use strict";
/* NanoGPT Chat front end: conversations, streaming, probability inspector. */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const icon = (paths) =>
  `<svg viewBox="0 0 24 24">${paths}</svg>`;
const ICONS = {
  copy: icon('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>'),
  check: icon('<path d="M5 12l5 5 9-10"/>'),
  redo: icon('<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>'),
  trash: icon('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'),
};

const STORE_CHATS = "nanogpt.chats.v1";
const STORE_SETTINGS = "nanogpt.settings.v1";
const STORE_UI = "nanogpt.ui.v1";
const DEFAULTS = {
  temperature: 0.8, topK: 0, topP: 1, maxTokens: 400, speaker: "ROMEO", seed: "",
  system: "", narrator: "KING RICHARD II",
};
// Written as scenes, not instructions: the model only continues text, so a
// system prompt steers it through mood and vocabulary.
const SYSTEM_PRESETS = [
  { label: "Love", text: "ROMEO:\nO love, sweet love, thou art my heart's delight;\nI dream of thee, and Juliet is the sun." },
  { label: "War", text: "KING RICHARD III:\nTo arms, to arms! The drums of battle sound,\nAnd blood shall wash the field ere night." },
  { label: "Grief", text: "QUEEN MARGARET:\nWeep, weep, for death hath come into our house,\nAnd all our joys lie buried in the grave." },
  { label: "Mirth", text: "PETRUCHIO:\nHa, ha! a merry jest, a merry jest!\nCome, Kate, let us be merry and make sport." },
  { label: "Magic", text: "PROSPERO:\nBy my art I raised the tempest and the storm;\nThe spirits of the isle obey my charms." },
];
const SUGGESTIONS = [
  { title: "A lover's question", text: "O gentle Juliet, wilt thou walk with me?" },
  { title: "A king's command", speaker: "KING RICHARD III", text: "Bring me my horse, and summon all my lords." },
  { title: "A story from one word", mode: "story", text: "crown" },
  { title: "Continue a line", mode: "continue", text: "To be, or not to be, that is" },
];
const STORY_WORDS = ["crown", "storm", "dagger", "rose", "ghost", "winter", "sea", "king"];
const historyFor = (messages) => messages
  .filter((m) => m.mode === "dialogue" && !m.error)
  .map((m) => ({ role: m.role, text: m.role === "user" ? (m.script || m.text) : m.text }));

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
  },
};

/* ---------------------------------------------------------------- state */
const state = {
  info: null,
  chats: store.get(STORE_CHATS, []),
  currentId: null,
  settings: { ...DEFAULTS, ...store.get(STORE_SETTINGS, {}) },
  ui: { mode: "dialogue", heat: true, inspector: window.innerWidth > 1180, theme: null, ...store.get(STORE_UI, {}) },
  controller: null,
  sel: null,        // { m: messageIndex, i: tokenIndex } pinned by click
  hover: null,      // same shape, transient
  live: null,       // message index currently streaming
};
state.currentId = state.chats[0]?.id ?? null;

const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const currentChat = () => state.chats.find((c) => c.id === state.currentId) || null;

function saveChats() {
  // Token-level detail is the bulk of the data; if storage is full, drop it
  // from the oldest conversations first and keep the text.
  while (!store.set(STORE_CHATS, state.chats)) {
    const victim = [...state.chats].reverse().find((c) => c.messages.some((m) => m.tokens));
    if (!victim) break;
    victim.messages.forEach((m) => delete m.tokens);
  }
}
const saveSettings = () => store.set(STORE_SETTINGS, state.settings);
const saveUI = () => store.set(STORE_UI, state.ui);

/* --------------------------------------------------------------- helpers */
const pct = (p) => (p >= 0.995 ? "100%" : p >= 0.1 ? (p * 100).toFixed(1) + "%" : p >= 0.001 ? (p * 100).toFixed(2) + "%" : "<0.1%");
const showChar = (c) => (c === "\n" ? "↵" : c === " " ? "␣" : c);
const isWs = (c) => c === "\n" || c === " ";
const T = (tok) => ({ t: tok[0], p: tok[1], raw: tok[2], rank: tok[3], ent: tok[4], top: tok[5], manual: !!tok[6] });
const busy = () => !!(state.controller || state.step);

function seqStats(tokens) {
  if (!tokens?.length) return null;
  let logp = 0, ent = 0;
  for (const tk of tokens) { logp += Math.log(Math.max(tk[1], 1e-12)); ent += tk[4]; }
  const n = tokens.length;
  return { n, ppl: Math.exp(-logp / n), avgP: Math.exp(logp / n), ent: ent / n };
}

/* -------------------------------------------------------------- sidebar */
function renderSidebar() {
  const list = $("chatList");
  list.replaceChildren();
  if (!state.chats.length) {
    list.append(el("div", "muted small", "No conversations yet."));
    list.lastChild.style.padding = "4px 10px";
  }
  for (const chat of state.chats) {
    const b = el("button", "chat-item" + (chat.id === state.currentId ? " active" : ""));
    b.append(el("span", "title", chat.title || "New chat"));
    const del = el("span", "del");
    del.innerHTML = ICONS.trash;
    del.title = "Delete conversation";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteChat(chat.id); });
    b.append(del);
    b.addEventListener("click", () => { selectChat(chat.id); closeSidebarMobile(); });
    list.append(b);
  }
}

function selectChat(id) {
  if (busy()) stopGeneration();
  state.currentId = id;
  state.sel = state.hover = null;
  renderSidebar();
  renderThread();
  refreshInspector();
}

function newChat() {
  if (busy()) stopGeneration();
  const existing = currentChat();
  if (existing && existing.messages.length === 0) { $("input").focus(); return; }
  state.currentId = null;
  state.sel = state.hover = null;
  renderSidebar();
  renderThread();
  refreshInspector();
  $("input").focus();
}

function deleteChat(id) {
  state.chats = state.chats.filter((c) => c.id !== id);
  if (state.currentId === id) state.currentId = state.chats[0]?.id ?? null;
  saveChats();
  selectChat(state.currentId);
}

/* --------------------------------------------------------------- thread */
function renderThread() {
  const inner = $("threadInner");
  inner.replaceChildren();
  const chat = currentChat();
  if (!chat || chat.messages.length === 0) { inner.append(renderEmpty()); return; }
  chat.messages.forEach((m, i) => inner.append(m.role === "user" ? renderUser(m) : renderModel(m, i)));
  scrollToBottom(true);
}

function renderEmpty() {
  const wrap = el("div", "empty");
  wrap.append(el("h1", null, "What light through yonder window breaks?"));
  wrap.append(el("p", null,
    "Write a line of dialogue and a small transformer trained on Shakespeare will answer as another character, one sampled character at a time."));
  const grid = el("div", "suggestions");
  for (const s of SUGGESTIONS) {
    const b = el("button", "suggestion");
    b.append(el("b", null, s.title), el("span", null, s.text));
    b.addEventListener("click", () => {
      if (s.mode) setMode(s.mode);
      else setMode("dialogue");
      if (s.speaker) { state.settings.speaker = s.speaker; syncSettingsUI(); saveSettings(); }
      $("input").value = s.text;
      autosize();
      send();
    });
    grid.append(b);
  }
  wrap.append(grid);
  return wrap;
}

function renderUser(m) {
  const row = el("div", "msg user");
  const bubble = el("div", "bubble");
  if (m.mode === "story") {
    bubble.classList.add("story");
    bubble.append(el("span", "speaker-tag", "Tell me a story about"), el("span", "word", m.text));
  } else if (m.mode === "dialogue" && m.script) {
    const nl = m.script.indexOf("\n");
    bubble.append(el("span", "speaker-tag", m.script.slice(0, nl).replace(/:$/, "")));
    bubble.append(document.createTextNode(m.script.slice(nl + 1)));
  } else {
    bubble.textContent = m.text;
  }
  row.append(bubble);
  return row;
}

/** Model message. Characters are spans so each can be inspected. In dialogue
 *  mode the first line ("JULIET:") is lifted into the speaker label. */
function renderModel(m, index) {
  const row = el("div", "msg model");
  row.dataset.m = index;
  const avatar = el("div", "avatar", "N");
  const body = el("div", "body");
  const speaker = el("div", "speaker");
  const gen = el("div", "gen");
  body.append(speaker, gen);
  row.append(avatar, body);

  const view = { row, speaker, gen, headerOpen: m.mode === "dialogue", headerSpans: [], m };
  row._view = view;
  if (m.mode === "story") applyStoryHeader(view, m);
  if (m.tokens) m.tokens.forEach((tk, i) => appendChar(view, tk[0], tk[1], i, tk[6]));
  else gen.textContent = m.text || "";
  if (m.stats || m.error) finishHeader(view); // still being written: the name line may be incomplete
  // The blank line that ends a speech is the stop signal, not content.
  if (m.stats) for (let s = gen.lastElementChild; s?.classList.contains("nl"); s = s.previousElementSibling) s.hidden = true;

  if (m.error) body.append(el("div", "err-text", m.error));
  if (m.stats) body.append(renderFoot(m, index));
  return row;
}

/** Story replies: the narrator comes from the template, and the opening line
 *  (which contains the user's word) was written by the template, not sampled. */
function applyStoryHeader(view, m) {
  view.headerOpen = false;
  if (m.speakerName) {
    view.speaker.textContent = m.speakerName;
    view.speaker.title = "Storyteller";
  }
  if (m.prefill && !view.gen.querySelector(".prefill")) {
    const pre = el("span", "prefill", m.prefill);
    pre.title = "Opening line from the story template; the model writes everything after it";
    view.gen.prepend(pre);
  }
}

/** Exactly what the model saw when it started writing this reply. */
function renderPromptView(m) {
  const p = m.prompt;
  const box = el("div", "prompt-view");
  const head = el("div", "prompt-view-head");
  const used = (p.pinned + p.context).length;
  head.innerHTML = `<span><b>What the model saw</b> before writing</span><span>${used} / ${state.info?.config.block_size ?? 256} chars</span>`;
  const pre = el("pre");
  if (p.pinned) pre.append(el("span", "pv-sys", p.pinned));
  if (p.cut) pre.append(el("span", "pv-cut", `[… ${p.cut} earlier characters no longer fit …]\n`));
  let ctx = p.context;
  if (m.prefill && ctx.endsWith(m.prefill)) {
    pre.append(document.createTextNode(ctx.slice(0, -m.prefill.length)));
    pre.append(el("span", "pv-pre", m.prefill));
  } else {
    pre.append(document.createTextNode(ctx));
  }
  pre.append(el("span", "pv-here", "▍"));
  const legend = el("div", "prompt-legend");
  if (p.pinned) legend.innerHTML += `<span><i style="background:var(--accent-soft);outline:1px solid var(--accent)"></i>system prompt (always kept)</span>`;
  if (m.prefill) legend.innerHTML += `<span><i style="background:color-mix(in srgb,var(--ent) 30%,transparent)"></i>story template</span>`;
  legend.innerHTML += `<span><i style="background:var(--heat)"></i>model starts writing here</span>`;
  box.append(head, pre, legend);
  return box;
}

function appendChar(view, ch, p, i, manual = false) {
  const span = el("span", "c" + (ch === "\n" ? " nl" : "") + (manual ? " manual" : ""), ch);
  if (manual) span.title = "You chose this character";
  span.dataset.i = i;
  span.style.setProperty("--p", p.toFixed(3));
  if (view.headerOpen) {
    if (ch === "\n") {
      view.headerSpans.push(span);
      span.hidden = true;
      view.speaker.append(span);
      finishHeader(view, true);
      return;
    }
    view.headerSpans.push(span);
    view.speaker.append(span);
    if (view.headerSpans.length > 40) finishHeader(view, true);
    return;
  }
  view.gen.append(span);
}

function finishHeader(view, sawNewline = false) {
  if (!view.headerOpen) return;
  view.headerOpen = false;
  const text = view.headerSpans.map((s) => s.textContent).join("").trim();
  const looksLikeName = sawNewline && /^[A-Za-z][A-Za-z' ]{0,30}:$/.test(text);
  if (looksLikeName) {
    view.speaker.title = "The model chose who speaks next";
    view.headerSpans.findLast((s) => s.textContent === ":")?.classList.add("colon");
    return;
  }
  // Not a speaker line: move the characters back into the body.
  const frag = document.createDocumentFragment();
  view.headerSpans.forEach((s) => { s.hidden = false; frag.append(s); });
  view.gen.prepend(frag);
  view.speaker.replaceChildren();
}

function renderFoot(m, index) {
  const foot = el("div", "msg-foot");
  const s = m.stats;
  const add = (label, value, title) => {
    const n = el("span", "stat");
    n.innerHTML = `${label} <b>${value}</b>`;
    if (title) n.title = title;
    foot.append(n);
  };
  add("chars", s.tokens);
  if (s.stepped) add("", s.manual ? `step-by-step · ${s.manual} picked by you` : "step-by-step", "Sampled one character at a time");
  else add("speed", `${s.tokens_per_sec}/s`, "Characters generated per second");
  if (m.tokens?.length) {
    const q = seqStats(m.tokens);
    add("avg p", pct(q.avgP), "Geometric mean probability of the sampled characters");
    add("perplexity", q.ppl.toFixed(2), "How surprised the sampler was on average. 1.0 = fully certain.");
  }
  add("temp", m.params.temperature);
  if (s.reason === "stopped") add("", "stopped");

  if (m.prompt) {
    const pb = el("button", "prompt-btn");
    pb.innerHTML = `${icon('<path d="M4 7l5 5-5 5M12 19h8"/>')}Prompt`;
    pb.title = "Show exactly what the model saw";
    pb.addEventListener("click", () => {
      const open = foot.nextElementSibling?.classList.contains("prompt-view");
      if (open) foot.nextElementSibling.remove();
      else foot.after(renderPromptView(m));
      pb.classList.toggle("on", !open);
    });
    foot.append(pb);
  }

  const actions = el("div", "msg-actions");
  const copy = el("button");
  copy.innerHTML = ICONS.copy;
  copy.title = "Copy";
  copy.addEventListener("click", async () => {
    const text = (m.prefill || "") + m.text;
    try { await navigator.clipboard.writeText(text); copy.innerHTML = ICONS.check; setTimeout(() => (copy.innerHTML = ICONS.copy), 1200); } catch {}
  });
  actions.append(copy);
  const chat = currentChat();
  if (chat && index === chat.messages.length - 1) {
    const redo = el("button");
    redo.innerHTML = ICONS.redo;
    redo.title = "Regenerate";
    redo.addEventListener("click", regenerate);
    actions.append(redo);
  }
  foot.append(actions);
  return foot;
}

function scrollToBottom(force = false) {
  const t = $("thread");
  const near = t.scrollHeight - t.scrollTop - t.clientHeight < 140;
  if (force || near) t.scrollTop = t.scrollHeight;
}

/* ----------------------------------------------------------- generation */
function setStreaming(on) {
  $("composer").classList.toggle("streaming", on);
  updateSendBtn();
}
function updateSendBtn() {
  $("sendBtn").disabled = !busy() && !$("input").value.trim();
  $("sendBtn").setAttribute("aria-label", state.step ? "Finish step-by-step reply" : state.controller ? "Stop" : "Send");
}

function stopGeneration() {
  state.controller?.abort();
  if (state.step) finishStep("stopped");
}

async function send(textOverride) {
  if (busy()) { stopGeneration(); return; }
  const text = (textOverride ?? $("input").value).trim();
  if (!text) return;
  if (!state.info) { showNotice("The model server isn't reachable. Start it with `python server.py`."); return; }

  let chat = currentChat();
  if (!chat) {
    chat = { id: uid(), title: "", created: Date.now(), messages: [] };
    state.chats.unshift(chat);
    state.currentId = chat.id;
  }
  const mode = state.ui.mode;
  const history = mode === "dialogue" ? historyFor(chat.messages) : [];
  if (!chat.title) chat.title = (mode === "story" ? `A tale of ${text}` : text).replace(/\s+/g, " ").slice(0, 48);
  const userMsg = { role: "user", text, mode };
  chat.messages.push(userMsg);
  if (textOverride == null) { $("input").value = ""; autosize(); }
  hideNotice();
  renderSidebar();
  renderThread();
  if (state.ui.stepMode) startStep(chat, userMsg, history);
  else await generate(chat, userMsg, history);
}

async function generate(chat, userMsg, history) {
  const s = state.settings;
  const params = { temperature: +s.temperature, top_k: +s.topK, top_p: +s.topP, max_new_tokens: +s.maxTokens };
  const modelMsg = { role: "model", mode: userMsg.mode, text: "", tokens: [], params: { temperature: params.temperature, top_k: params.top_k, top_p: params.top_p } };
  chat.messages.push(modelMsg);
  const index = chat.messages.length - 1;

  const row = renderModel(modelMsg, index);
  const view = row._view;
  const thinking = el("span", "thinking");
  thinking.innerHTML = "<i></i><i></i><i></i>";
  view.gen.append(thinking);
  const cursor = el("span", "cursor");
  $("threadInner").append(row);
  scrollToBottom(true);

  state.controller = new AbortController();
  state.live = index;
  state.sel = null;
  setStreaming(true);

  const body = {
    history, prompt: userMsg.text, mode: userMsg.mode, speaker: s.speaker || "ROMEO",
    system: s.system || "", narrator: s.narrator || DEFAULTS.narrator,
    ...params, seed: s.seed === "" ? null : Number(s.seed),
  };

  let pending = [];
  let raf = 0;
  const flush = () => {
    raf = 0;
    if (!pending.length) return;
    thinking.remove();
    for (const tk of pending) {
      appendChar(view, tk[0], tk[1], modelMsg.tokens.length);
      modelMsg.tokens.push(tk);
      modelMsg.text += tk[0];
    }
    pending = [];
    (view.headerOpen ? view.speaker : view.gen).append(cursor);
    if (!state.hover && !state.sel) refreshInspector();
    else updateTimeline();
    scrollToBottom();
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: state.controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error ${res.status}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "start") {
          userMsg.script = ev.user_text;
          modelMsg.prompt = { pinned: ev.pinned, context: ev.context, cut: ev.cut, seed: ev.seed };
          if (userMsg.mode === "story") {
            modelMsg.speakerName = ev.speaker;
            modelMsg.prefill = ev.prefill;
            thinking.remove();
            applyStoryHeader(view, modelMsg);
            view.gen.append(thinking);
          }
          if (ev.dropped?.length) showNotice(`Removed characters the model has never seen: ${ev.dropped.map((c) => `“${c}”`).join(" ")}`);
          const prev = row.previousElementSibling;
          if (prev) prev.replaceWith(renderUser(userMsg));
        } else if (ev.type === "token") {
          pending.push([ev.t, ev.p, ev.raw_p, ev.rank, ev.entropy, ev.top]);
          if (!raf) raf = requestAnimationFrame(flush);
        } else if (ev.type === "done") {
          modelMsg.stats = ev;
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      modelMsg.stats = modelMsg.stats || { reason: "stopped", tokens: modelMsg.tokens.length, tokens_per_sec: "–" };
    } else {
      modelMsg.error = err.message || String(err);
    }
  } finally {
    if (raf) cancelAnimationFrame(raf);
    flush();
    state.controller = null;
    state.live = null;
    setStreaming(false);
    modelMsg.text = modelMsg.text.replace(/\n+$/, "");
    if (!modelMsg.tokens.length && !modelMsg.error) modelMsg.error = "No output.";
    saveChats();
    row.replaceWith(renderModel(modelMsg, index));
    refreshInspector();
    scrollToBottom();
  }
}

function regenerate() {
  const chat = currentChat();
  if (!chat || busy()) return;
  const last = chat.messages[chat.messages.length - 1];
  if (last?.role !== "model") return;
  chat.messages.pop();
  const userMsg = chat.messages[chat.messages.length - 1];
  const before = chat.messages.slice(0, -1);
  const history = userMsg.mode === "dialogue" ? historyFor(before) : [];
  state.sel = null;
  renderThread();
  if (state.ui.stepMode) startStep(chat, userMsg, history);
  else generate(chat, userMsg, history);
}

/* ------------------------------------------------------------ inspector */
function activeTarget() {
  const chat = currentChat();
  if (!chat) return null;
  const target = state.hover || state.sel;
  if (target) return { ...target, live: false };
  if (state.live != null) {
    const msg = chat.messages[state.live];
    if (msg?.tokens.length) return { m: state.live, i: msg.tokens.length - 1, live: true };
  }
  // Default: last character of the latest reply.
  for (let m = chat.messages.length - 1; m >= 0; m--) {
    const msg = chat.messages[m];
    if (msg.role === "model" && msg.tokens?.length) return { m, i: msg.tokens.length - 1, live: false };
  }
  return null;
}

function refreshInspector() {
  const target = activeTarget();
  const chat = currentChat();
  $("inspEmpty").hidden = !!target;
  $("inspContent").hidden = !target;
  document.querySelectorAll(".gen .c.sel, .speaker .c.sel").forEach((n) => n.classList.remove("sel"));
  if (!target) return;
  const msg = chat.messages[target.m];
  const tk = T(msg.tokens[target.i]);
  state.inspected = { m: target.m, i: target.i };
  $("inspXray").disabled = !msg.prompt || !!state.controller;
  $("inspXray").title = !msg.prompt ? "Only available for replies made after the X-ray was added" : "";

  $("livePill").hidden = !target.live;
  $("inspTitle").textContent = target.live ? "Just sampled" : `Character ${target.i + 1} of ${msg.tokens.length}${tk.manual ? " · picked by you" : ""}`;
  $("tokGlyph").innerHTML = "";
  $("tokGlyph").append(document.createTextNode(isWs(tk.t) ? showChar(tk.t) : tk.t));
  if (isWs(tk.t)) $("tokGlyph").append(el("small", null, tk.t === "\n" ? "newline" : "space"));
  $("tokP").textContent = pct(tk.p);
  $("tokRaw").textContent = pct(tk.raw);
  $("tokRank").textContent = `#${tk.rank}`;
  $("tokEnt").textContent = `${tk.ent.toFixed(2)} bits`;

  // Context: the characters the model had just written.
  const before = msg.tokens.slice(Math.max(0, target.i - 70), target.i).map((x) => x[0]).join("");
  const peek = $("ctxPeek");
  peek.replaceChildren(document.createTextNode((target.i > 70 ? "…" : "") + before));
  peek.append(el("mark", null, showChar(tk.t)));

  // Bars: top candidates, with sampled-from (after temperature/top-k/top-p) and raw probabilities.
  const bars = $("bars");
  bars.replaceChildren();
  const maxP = Math.max(tk.top[0][1], ...tk.top.map((r) => r[2]));
  let chosenShown = false;
  for (const [c, p, raw] of tk.top) {
    if (p <= 0 && raw < 0.001) continue;
    bars.append(barRow(c, p, raw, maxP, c === tk.t));
    if (c === tk.t) chosenShown = true;
  }
  if (!chosenShown) {
    bars.append(el("li", "muted small", "…"));
    bars.append(barRow(tk.t, tk.p, tk.raw, maxP, true));
  }

  updateTimeline(target);
  if (!target.live) {
    const row = document.querySelector(`.msg.model[data-m="${target.m}"]`);
    row?.querySelector(`.c[data-i="${target.i}"]`)?.classList.add("sel");
  }
}

function barRow(c, p, raw, maxP, chosen) {
  const li = el("li", "bar" + (chosen ? " chosen" : ""));
  li.title = `${showChar(c)}: ${pct(p)} sampled-from, ${pct(raw)} raw`;
  const ch = el("span", "ch" + (isWs(c) ? " ws" : ""), isWs(c) ? (c === "\n" ? "↵ nl" : "␣ sp") : c);
  const track = el("span", "track");
  const fr = el("span", "fill-raw"); fr.style.width = `${(raw / maxP) * 100}%`;
  const f = el("span", "fill"); f.style.width = `${(p / maxP) * 100}%`;
  track.append(fr, f);
  li.append(ch, track, el("span", "pct", pct(p)));
  return li;
}

function updateTimeline(target = activeTarget()) {
  const tl = $("timeline");
  const chat = currentChat();
  if (!target || !chat) { tl.replaceChildren(); return; }
  const msg = chat.messages[target.m];
  const toks = msg.tokens;
  const n = toks.length;
  const buckets = Math.min(n, 140);
  const size = n / buckets;
  const frag = document.createDocumentFragment();
  const entPts = [];
  const maxEnt = state.info?.max_entropy_bits || 6;
  for (let b = 0; b < buckets; b++) {
    const s = Math.floor(b * size), e = Math.max(s + 1, Math.floor((b + 1) * size));
    let pSum = 0, eSum = 0;
    for (let k = s; k < e; k++) { pSum += toks[k][1]; eSum += toks[k][4]; }
    const bar = el("div", "tb");
    bar.style.height = `${Math.max(3, (pSum / (e - s)) * 100)}%`;
    bar.dataset.i = s;
    if (target.i >= s && target.i < e && !target.live) bar.classList.add("sel");
    bar.title = e - s === 1 ? `${showChar(toks[s][0])} · ${pct(toks[s][1])}` : `chars ${s + 1}–${e}`;
    frag.append(bar);
    entPts.push([(b + 0.5) / buckets * 100, 100 - Math.min(1, eSum / (e - s) / maxEnt) * 100]);
  }
  tl.replaceChildren(frag);
  tl.dataset.m = target.m;
  if (entPts.length > 1) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "ent");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    const pl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    pl.setAttribute("points", entPts.map((p) => p.join(",")).join(" "));
    pl.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(pl);
    tl.append(svg);
  }
  const q = seqStats(toks);
  $("seqStats").textContent = q ? `${q.n} chars · ppl ${q.ppl.toFixed(2)}` : "";
}

/* ------------------------------------------------------ hover & tooltip */
function charTarget(node) {
  const span = node.closest?.(".c");
  const row = span?.closest(".msg.model");
  if (!span || !row) return null;
  return { m: +row.dataset.m, i: +span.dataset.i, span };
}

function showTooltip(t, x, y) {
  const tk = T(currentChat().messages[t.m].tokens[t.i]);
  const tip = $("tooltip");
  tip.replaceChildren();
  const head = el("div", "tt-head");
  head.append(el("span", null, `“${showChar(tk.t)}” rank #${tk.rank}`), el("span", "m", pct(tk.p)));
  tip.append(head);
  const max = tk.top[0][1] || 1;
  for (const [c, p] of tk.top.slice(0, 5)) {
    const r = el("div", "tt-row" + (c === tk.t ? " chosen" : ""));
    const bar = el("span"); const i = el("i"); i.style.width = `${(p / max) * 100}%`; bar.append(i);
    r.append(el("span", null, showChar(c)), bar, el("span", null, pct(p)));
    tip.append(r);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(x + 14, window.innerWidth - w - 8)}px`;
  tip.style.top = `${y + 18 + h > window.innerHeight ? y - h - 10 : y + 18}px`;
}

function bindThreadEvents() {
  const thread = $("thread");
  thread.addEventListener("mousemove", (e) => {
    if (!state.ui.heat) return;
    const t = charTarget(e.target);
    const msg = t && currentChat()?.messages[t.m];
    if (!t || !msg?.tokens?.[t.i]) {
      if (state.hover) { state.hover = null; $("tooltip").hidden = true; refreshInspector(); }
      return;
    }
    showTooltip(t, e.clientX, e.clientY);
    if (state.hover?.m !== t.m || state.hover?.i !== t.i) { state.hover = { m: t.m, i: t.i }; refreshInspector(); }
  });
  thread.addEventListener("mouseleave", () => {
    $("tooltip").hidden = true;
    if (state.hover) { state.hover = null; refreshInspector(); }
  });
  thread.addEventListener("click", (e) => {
    if (!state.ui.heat) return;
    const t = charTarget(e.target);
    if (!t || !currentChat()?.messages[t.m]?.tokens?.[t.i]) return;
    const same = state.sel && state.sel.m === t.m && state.sel.i === t.i;
    state.sel = same ? null : { m: t.m, i: t.i };
    if (!state.ui.inspector) setInspector(true);
    refreshInspector();
  });
  $("timeline").addEventListener("click", (e) => {
    const bar = e.target.closest(".tb");
    if (!bar) return;
    state.sel = { m: +$("timeline").dataset.m, i: +bar.dataset.i };
    state.hover = null;
    refreshInspector();
    document.querySelector(".c.sel")?.scrollIntoView({ block: "center", behavior: "smooth" });
  });
}

/* ------------------------------------------------------------- settings */
const SETTING_FIELDS = [
  ["temperature", (v) => (+v).toFixed(2)],
  ["topK", (v) => (+v === 0 ? "off" : v)],
  ["topP", (v) => (+v >= 1 ? "off" : (+v).toFixed(2))],
  ["maxTokens", (v) => v],
];
function syncSettingsUI() {
  for (const [k, fmt] of SETTING_FIELDS) {
    $(k).value = state.settings[k];
    $(k + "Out").textContent = fmt(state.settings[k]);
  }
  $("speaker").value = state.settings.speaker;
  $("seed").value = state.settings.seed;
}
function bindSettings() {
  for (const [k, fmt] of SETTING_FIELDS) {
    $(k).addEventListener("input", () => {
      state.settings[k] = +$(k).value;
      $(k + "Out").textContent = fmt($(k).value);
      saveSettings();
    });
  }
  $("speaker").addEventListener("input", () => { state.settings.speaker = $("speaker").value.toUpperCase(); saveSettings(); });
  $("seed").addEventListener("input", () => {
    $("seed").value = $("seed").value.replace(/\D/g, "").slice(0, 9);
    state.settings.seed = $("seed").value; saveSettings();
  });
  $("resetSettings").addEventListener("click", () => {
    for (const k of ["temperature", "topK", "topP", "maxTokens", "seed"]) state.settings[k] = DEFAULTS[k];
    saveSettings(); syncSettingsUI();
  });
  $("settingsBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    $("settings").hidden = !$("settings").hidden;
    $("settingsBtn").classList.toggle("on", !$("settings").hidden);
  });
  document.addEventListener("click", (e) => {
    if (!$("settings").hidden && !$("settings").contains(e.target)) {
      $("settings").hidden = true; $("settingsBtn").classList.remove("on");
    }
  });
  syncSettingsUI();
}

/* ---------------------------------------------------------------- chrome */
function setMode(mode) {
  state.ui.mode = mode;
  saveUI();
  document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("input").placeholder = {
    dialogue: "Speak, and the model will answer in verse…",
    story: `Give me a word (${STORY_WORDS[Math.floor(Math.random() * STORY_WORDS.length)]}, perhaps) and I'll spin a tale…`,
    continue: "Start a passage and the model will continue it…",
  }[mode];
}

/* -------------------------------------------------------- system prompt */
function setSystem(text) {
  state.settings.system = text.slice(0, 120);
  saveSettings();
  syncSystemUI();
}
function syncSystemUI() {
  const text = state.settings.system || "";
  if ($("system").value !== text) $("system").value = text;
  $("systemCount").textContent = `${text.length} / 120`;
  const preset = SYSTEM_PRESETS.find((p) => p.text === text);
  document.querySelectorAll("#systemPresets .chip").forEach((c) => c.classList.toggle("on", c.dataset.label === preset?.label));
  $("sysPill").hidden = !text.trim();
  $("sysPillText").textContent = preset ? preset.label : text.replace(/\s+/g, " ").trim();
}
function bindSystem() {
  const chips = $("systemPresets");
  for (const p of SYSTEM_PRESETS) {
    const c = el("button", "chip", p.label);
    c.type = "button";
    c.dataset.label = p.label;
    c.title = p.text;
    c.addEventListener("click", () => setSystem(state.settings.system === p.text ? "" : p.text));
    chips.append(c);
  }
  $("system").addEventListener("input", () => setSystem($("system").value));
  $("sysPillOpen").addEventListener("click", (e) => {
    e.stopPropagation();
    $("settings").hidden = false;
    $("settingsBtn").classList.add("on");
    $("system").focus();
  });
  $("sysPillClear").addEventListener("click", () => setSystem(""));
  $("narrator").addEventListener("change", () => { state.settings.narrator = $("narrator").value; saveSettings(); });
  fillNarrators(["KING RICHARD II", "PROSPERO", "FRIAR LAURENCE"]);
  syncSystemUI();
}
function fillNarrators(names) {
  const sel = $("narrator");
  sel.replaceChildren(...names.map((n) => {
    const o = el("option", null, n.replace(/\w\S*/g, (w) => (w === "II" ? w : w[0] + w.slice(1).toLowerCase())));
    o.value = n;
    return o;
  }), Object.assign(el("option", null, "Surprise me"), { value: "random" }));
  sel.value = names.includes(state.settings.narrator) || state.settings.narrator === "random" ? state.settings.narrator : names[0];
}
function setHeat(on) {
  state.ui.heat = on; saveUI();
  $("heatToggle").checked = on;
  $("thread").classList.toggle("heat-on", on);
  if (!on) $("tooltip").hidden = true;
}
function setInspector(on) {
  state.ui.inspector = on; saveUI();
  $("app").classList.toggle("no-inspector", !on);
  $("inspectorBtn").classList.toggle("on", on);
}
function applyTheme() {
  if (state.ui.theme) document.documentElement.dataset.theme = state.ui.theme;
  else delete document.documentElement.dataset.theme;
}
function toggleTheme() {
  const dark = state.ui.theme ? state.ui.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  state.ui.theme = dark ? "light" : "dark";
  saveUI(); applyTheme();
}
function showNotice(text) { $("notice").textContent = text; $("notice").hidden = false; }
function hideNotice() { $("notice").hidden = true; }
function closeSidebarMobile() { $("app").classList.remove("sidebar-open"); }

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
  updateSendBtn();
}

async function loadInfo() {
  try {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error();
    const info = await res.json();
    state.info = info;
    $("statusDot").className = "status-dot ok";
    $("modelName").textContent = info.name;
    $("frozenBadge").hidden = !info.frozen;
    if (info.narrators) fillNarrators(info.narrators);
    const meta = info.meta || {};
    const rows = [
      ["Parameters", "parameters", (info.params / 1e6).toFixed(2) + "M"],
      ["Layers × heads", "layer", `${info.config.n_layer} × ${info.config.n_head}`],
      ["Context", "context", `${info.config.block_size} chars`],
      ["Vocabulary", "vocabulary", `${info.vocab.length} chars`],
      meta.val_loss != null && ["Val loss", "val-loss", meta.val_loss.toFixed(4)],
      meta.step != null && ["Trained steps", "step", meta.step.toLocaleString()],
      ["Running on", "gpu", info.device.startsWith("cuda") ? "GPU" : "CPU"],
    ].filter(Boolean);
    const dl = $("modelStats");
    dl.replaceChildren();
    for (const [k, key, v] of rows) {
      const dt = el("dt");
      dt.append(Object.assign(el("span", "term", k), { title: "" }));
      dt.firstChild.dataset.term = key;
      dl.append(dt, el("dd", null, v));
    }
    $("composerFoot").textContent =
      `A ${(info.params / 1e6).toFixed(1)}M-parameter character model trained only on Shakespeare. It imitates style, not meaning.`;
    hideNotice();
  } catch {
    $("statusDot").className = "status-dot err";
    $("modelName").textContent = "Server offline";
    showNotice("Can't reach the model server. Start it with chat.bat (retrying…)");
    setTimeout(loadInfo, 2000);
  }
}

function init() {
  applyTheme();
  setMode(state.ui.mode);
  setHeat(state.ui.heat);
  setInspector(state.ui.inspector);
  bindSettings();
  bindSystem();
  bindThreadEvents();

  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  $("input").addEventListener("input", autosize);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!busy()) send(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (busy()) stopGeneration();
      else if (state.sel) { state.sel = null; refreshInspector(); }
    }
  });
  $("newChat").addEventListener("click", () => { newChat(); closeSidebarMobile(); });
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("heatToggle").addEventListener("change", (e) => setHeat(e.target.checked));
  $("inspectorBtn").addEventListener("click", () => setInspector(!state.ui.inspector));
  $("closeInspector").addEventListener("click", () => setInspector(false));
  $("themeBtn").addEventListener("click", toggleTheme);
  $("glossaryBtn").addEventListener("click", () => openGlossary());
  $("xrayBtn").addEventListener("click", () => openXray());
  $("inspXray").addEventListener("click", () => {
    const t = state.inspected;
    const msg = t && currentChat()?.messages[t.m];
    if (!msg?.prompt) return;
    // Rebuild exactly the window the model had when it chose this character.
    const written = msg.tokens.slice(0, t.i).map((x) => x[0]).join("");
    const p = msg.params || {};
    openXray({
      pinned: msg.prompt.pinned, body: msg.prompt.context + written, target: msg.tokens[t.i][0],
      temperature: p.temperature, top_k: p.top_k ?? 0, top_p: p.top_p ?? 1, fromReply: true, index: t.i,
    });
  });
  $("openSidebar").addEventListener("click", () => $("app").classList.add("sidebar-open"));
  $("closeSidebar").addEventListener("click", closeSidebarMobile);
  $("scrim").addEventListener("click", closeSidebarMobile);

  renderSidebar();
  renderThread();
  refreshInspector();
  updateSendBtn();
  loadInfo();
  $("input").focus();
}

init();
