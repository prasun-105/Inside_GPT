"use strict";
/* Plain-English glossary. Any element with data-term="<key>" gets a hover
   definition and opens the full entry on click. */

const GLOSSARY_GROUPS = ["Text into numbers", "Inside the model", "Making a prediction", "How it learned"];

const GLOSSARY = {
  // ---------------------------------------------------------------- text
  token: {
    term: "Token", group: 0,
    short: "The smallest piece of text the model reads and writes. Here every token is a single character: a letter, a space, a comma or a line break.",
    like: "Like the tiles in a game of Scrabble. The model builds every reply one tile at a time.",
    here: "Every highlighted character in a reply is one token.",
  },
  vocabulary: {
    term: "Vocabulary", group: 0,
    short: "The complete list of tokens the model knows. This model knows 65 characters: upper- and lowercase letters, digits it saw, punctuation, space and newline.",
    like: "The set of tiles in the Scrabble bag. A tile that isn't in the bag (like é or an emoji) can't be played.",
    here: "That's why unusual characters you type get removed with a warning.",
  },
  "token-id": {
    term: "Token ID", group: 0,
    short: "The number a token is given so the computer can work with it. 'A' might be 13, 'e' 43. Computers can only do maths on numbers, not letters.",
    like: "A locker number: the number itself means nothing, it just tells you which locker to open.",
    here: "Shown under each character in the X-ray's first step.",
  },
  context: {
    term: "Context window", group: 0,
    short: "How much recent text the model can see at once when choosing the next token. This model sees the last 256 characters; anything older is forgotten.",
    like: "Reading through a slot that shows only the last few lines of a page.",
    here: "Long conversations scroll out of the window from the oldest end. The system prompt is pinned so it never scrolls out.",
  },
  // ---------------------------------------------------------- inside
  vector: {
    term: "Vector", group: 1,
    short: "A list of numbers that together describe something. Inside this model every character is represented by a list of 384 numbers.",
    like: "A recipe card with 384 ingredient amounts. No single number means much; the mix is what matters.",
    here: "The striped barcodes in the X-ray are vectors: blue cells are negative numbers, red are positive.",
  },
  embedding: {
    term: "Embedding", group: 1,
    short: "Turning a token into a vector (a list of numbers) that the model can compute with. Similar tokens end up with similar lists.",
    like: "Giving each word a set of GPS coordinates on a map of meaning, so related words land near each other.",
    here: "The first stage of the X-ray.",
  },
  "token-embedding": {
    term: "Token embedding", group: 1,
    short: "The learned list of 384 numbers for each character in the vocabulary. It captures what the character is: vowel or consonant, capital or not, punctuation, and so on.",
    like: "A character's ID badge describing who it is.",
    here: "The first barcode in the X-ray's embedding step.",
  },
  "position-embedding": {
    term: "Position embedding", group: 1,
    short: "A second learned list of 384 numbers for each position in the window (1st, 2nd … 256th). It's added to the token embedding so the model knows where each character sits.",
    like: "A seat number added to the ID badge: the same person in a different seat.",
    here: "The second barcode in the X-ray's embedding step.",
  },
  layer: {
    term: "Layer (transformer block)", group: 1,
    short: "One round of processing. Each layer first lets characters look at each other (attention), then lets each one think on its own (feed-forward). This model stacks 6 layers.",
    like: "A team of editors passing a draft along a line; each one improves it a little.",
    here: "Layers 1 to 6 in the X-ray.",
  },
  attention: {
    term: "Self-attention", group: 1,
    short: "The step where each character looks back at earlier characters and pulls in information from the ones that matter. It's how the model knows that 'br' after 'window' probably continues as 'breaks'.",
    like: "Glancing back over what you've written to keep a sentence consistent.",
    here: "The purple grids in each layer of the X-ray: darker cells were looked at more.",
  },
  head: {
    term: "Attention head", group: 1,
    short: "One independent attention pattern. Each layer has 6 heads working side by side, and each tends to specialise: one watches the previous letter, another the start of the word, another the speaker's name.",
    like: "Six readers each scanning the text for a different thing.",
    here: "The 6 rows of each attention grid.",
  },
  qkv: {
    term: "Query, key and value", group: 1,
    short: "The mechanism behind attention. Each character makes a query ('what am I looking for?'), offers a key ('what do I contain?') and a value ('what I'll pass on'). Queries that match keys well get more of that value.",
    like: "A library search: your question (query) is matched against book labels (keys) and you take home the matching books' contents (values).",
    here: "The grid's shading is how well the current character's query matched each earlier key.",
  },
  "causal-mask": {
    term: "Causal mask", group: 1,
    short: "A rule that stops a character from looking at anything after it. The model may only use the past to predict the future.",
    like: "Covering the rest of the page with your hand while you guess the next word.",
    here: "That's why the attention grid only ever points backwards.",
  },
  "feed-forward": {
    term: "Feed-forward network", group: 1,
    short: "The 'thinking' part of each layer. It takes each character's vector on its own and runs it through 1,536 neurons to transform it. Most of the model's stored knowledge lives here.",
    like: "After a meeting (attention), each person goes back to their desk to work through what they heard.",
    here: "'Neurons firing' and the 'feed-forward' change bar in each X-ray layer.",
  },
  neuron: {
    term: "Neuron", group: 1,
    short: "A tiny unit that computes a weighted sum of its inputs and passes it on only if it is positive (this cut-off is called ReLU). A neuron 'fires' when its output is above zero.",
    like: "A light switch with a dimmer that only turns on when pushed hard enough.",
    here: "Each layer shows what share of its 1,536 neurons fired for this prediction. Usually only a small fraction does.",
  },
  residual: {
    term: "Residual stream", group: 1,
    short: "The running vector for a character as it travels through the layers. Each layer doesn't replace it; it adds a correction on top. So information from early layers is never lost.",
    like: "A shared notebook passed down the line where each editor adds notes in the margin rather than rewriting the page.",
    here: "The barcode after each layer, and the 'change made' bars showing how much each part added.",
  },
  layernorm: {
    term: "Layer norm", group: 1,
    short: "A step that rescales a vector so its numbers have a consistent size before the next calculation. It keeps the maths stable.",
    like: "Adjusting the volume so every speaker in a meeting is heard at the same loudness.",
    here: "Happens before attention, before feed-forward, and once more before the final output.",
  },
  parameters: {
    term: "Parameters (weights)", group: 1,
    short: "All the numbers the model learned during training: the embeddings, attention and feed-forward weights. This model has 10.8 million of them; that's the 52 MB file on disk.",
    like: "The settings of millions of tiny knobs, tuned until the model got good at predicting Shakespeare.",
    here: "The 'Parameters' row in the sidebar card.",
  },
  frozen: {
    term: "Frozen weights", group: 1,
    short: "The parameters are locked: using the model never changes them. It doesn't learn from your chats.",
    like: "A printed book rather than a notebook: you can read it as often as you like, but it won't change.",
    here: "The 'frozen' badge in the sidebar.",
  },
  "logit-lens": {
    term: "Logit lens", group: 1,
    short: "A way to peek at what the model would predict if it stopped early. We take the half-finished vector after any layer and send it straight to the output step.",
    like: "Asking a student for their answer halfway through working out a problem.",
    here: "'Best guess after this layer' in the X-ray, and the chart of how the guess changes layer by layer.",
  },
  // ---------------------------------------------------------- predict
  logits: {
    term: "Logits", group: 2,
    short: "The raw scores the model gives every token in the vocabulary before they're turned into probabilities. Higher means more likely; they can be any number, even negative.",
    like: "Raw votes before they're converted into percentages.",
    here: "The 'score' column in the X-ray's output step.",
  },
  softmax: {
    term: "Softmax", group: 2,
    short: "The formula that turns logits (raw scores) into probabilities that are all positive and add up to 100%. Bigger scores get a much bigger share.",
    like: "Turning vote counts into a pie chart.",
    here: "The last step before sampling in the X-ray.",
  },
  probability: {
    term: "Probability distribution", group: 2,
    short: "The model's answer at each step: a percentage for every one of the 65 characters, adding up to 100%. The model never outputs a single letter, only these odds.",
    like: "A weather forecast: 70% sun, 20% cloud, 10% rain.",
    here: "The bars in the inspector and the coloured tint on every character.",
  },
  sampling: {
    term: "Sampling", group: 2,
    short: "Picking the next token at random, weighted by its probability. A 60% token is picked 6 times in 10. This is why the same prompt gives different replies.",
    like: "Spinning a prize wheel whose slices are sized by the probabilities.",
    here: "The 'Rank' in the inspector shows when a less likely character won the spin.",
  },
  temperature: {
    term: "Temperature", group: 2,
    short: "A dial that sharpens or flattens the probabilities before sampling. Below 1 makes the favourite even more likely (safer, more repetitive); above 1 gives long shots more chance (wilder).",
    like: "How adventurous a diner is: low orders the usual, high tries anything on the menu.",
    here: "The first slider in Settings.",
  },
  "top-k": {
    term: "Top-k", group: 2,
    short: "Only allow the k most likely tokens; the rest get zero chance. Top-k 5 means pick only among the top 5.",
    like: "Shortlisting the five best candidates before drawing a name from a hat.",
    here: "A slider in Settings. 0 means off.",
  },
  "top-p": {
    term: "Top-p (nucleus)", group: 2,
    short: "Keep the smallest set of most-likely tokens whose probabilities add up to p (say 90%), and drop the rest. It adapts: few options when the model is sure, many when it isn't.",
    like: "Inviting the most popular guests until you've covered 90% of everyone's votes.",
    here: "A slider in Settings. 1.0 means off.",
  },
  raw: {
    term: "Raw vs adjusted probability", group: 2,
    short: "Raw is what the model itself predicted. Adjusted (sampled from) is after temperature, top-k and top-p were applied, and is what the random draw actually used.",
    like: "The forecast as issued vs. the forecast after your own rules of thumb.",
    here: "Grey bars (raw) vs purple bars (sampled from) in the inspector.",
  },
  rank: {
    term: "Rank", group: 2,
    short: "Where the chosen character stood among all options. #1 means the model's favourite was picked; #4 means three likelier characters lost the draw.",
    like: "A horse race result: did the favourite win?",
    here: "Shown in the inspector for every character.",
  },
  entropy: {
    term: "Entropy", group: 2,
    short: "How uncertain the model was, in bits. Near 0 means it was sure of one character; about 6 means it had no idea (all 65 equally likely).",
    like: "How much a crowd disagrees: everyone shouting the same answer is low; everyone saying something different is high.",
    here: "The inspector's 'Entropy' and the teal line on the confidence chart.",
  },
  perplexity: {
    term: "Perplexity", group: 2,
    short: "Average uncertainty across a whole reply, as 'how many options it was effectively choosing between'. 1.0 is total certainty; 3 is like choosing among three equally likely options each time.",
    like: "A multiple-choice test where perplexity is the average number of answers you couldn't rule out.",
    here: "Under every reply.",
  },
  seed: {
    term: "Seed", group: 2,
    short: "The starting number for the random draws. The same seed with the same prompt and settings gives exactly the same reply.",
    like: "Shuffling a deck in exactly the same way every time.",
    here: "Settings. Leave it empty for a fresh random reply each time.",
  },
  "system-prompt": {
    term: "System prompt", group: 2,
    short: "Text placed in front of everything you type, which always stays in the context window. This model can't follow instructions, so it works by setting a mood with a short scene.",
    like: "Stage lighting: it doesn't tell the actors what to say, but it changes the feel of the scene.",
    here: "Settings, and the pill above the message box when one is active.",
  },
  prefill: {
    term: "Prefill (story template)", group: 2,
    short: "Text written into the reply by the app, not the model, to get it started. Story mode uses a template line containing your word.",
    like: "Giving a storyteller their first line.",
    here: "The italic first line of a story.",
  },
  inference: {
    term: "Inference", group: 2,
    short: "Using a trained model to make predictions, as opposed to training it.",
    like: "Driving a car after you've learned to drive.",
    here: "Everything this chat does.",
  },
  // ---------------------------------------------------------- training
  training: {
    term: "Training", group: 3,
    short: "Showing the model millions of snippets of Shakespeare and nudging its parameters each time so it predicts the next character a bit better. This model trained for about 30 minutes on a laptop GPU.",
    like: "Practising flashcards: guess, check the answer, adjust, repeat.",
    here: "train.py does this once; the result is saved and frozen.",
  },
  loss: {
    term: "Loss", group: 3,
    short: "A score for how wrong the model's predictions are; lower is better. It measures how surprised the model was by the real next character.",
    like: "A golf score: you want it as low as possible.",
    here: "Training printed the loss every 250 steps.",
  },
  "val-loss": {
    term: "Validation loss", group: 3,
    short: "The loss on text the model was never trained on (the last 10% of the corpus). It shows whether the model learned general patterns or just memorised.",
    like: "A test with questions that weren't in the practice set.",
    here: "'Val loss' in the sidebar: 1.475.",
  },
  overfitting: {
    term: "Overfitting", group: 3,
    short: "When a model starts memorising its training text instead of learning patterns. Training loss keeps falling but validation loss rises. Here it began after step 2,250, so that version was kept.",
    like: "Memorising last year's exam answers instead of understanding the subject.",
    here: "Why the saved model is from step 2,250, not 5,000.",
  },
  step: {
    term: "Training step", group: 3,
    short: "One round of training: the model guesses on a batch of 64 snippets, sees how wrong it was, and adjusts its parameters slightly.",
    like: "One rep at the gym.",
    here: "'Trained steps' in the sidebar.",
  },
  gpu: {
    term: "GPU", group: 3,
    short: "A graphics card. It does thousands of multiplications at once, which makes training and running neural networks far faster than on the main processor (CPU).",
    like: "A thousand calculators working at the same time instead of one very fast one.",
    here: "Your RTX 4050 trained this model and runs it.",
  },
};

/* ------------------------------------------------------------- tooltip */
(function initGlossary() {
  const tip = el("div", "term-tip");
  tip.hidden = true;
  document.body.append(tip);
  let current = null;

  document.addEventListener("mouseover", (e) => {
    const node = e.target.closest?.("[data-term]");
    if (node === current) return;
    current = node;
    const entry = node && GLOSSARY[node.dataset.term];
    if (!entry) { tip.hidden = true; return; }
    tip.replaceChildren(el("b", null, entry.term), el("p", null, entry.short), el("span", "term-more", "Click for more"));
    tip.hidden = false;
    const r = node.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - w - 8))}px`;
    tip.style.top = `${r.bottom + 8 + h > window.innerHeight ? r.top - h - 8 : r.bottom + 8}px`;
  });
  document.addEventListener("click", (e) => {
    const node = e.target.closest?.("[data-term]");
    if (!node || !GLOSSARY[node.dataset.term]) return;
    e.preventDefault();
    e.stopPropagation();
    tip.hidden = true;
    openGlossary(node.dataset.term);
  }, true);
  window.addEventListener("scroll", () => (tip.hidden = true), true);
})();

/* --------------------------------------------------------------- panel */
function openGlossary(focusKey) {
  let modal = document.getElementById("glossary");
  if (!modal) modal = buildGlossary();
  modal.hidden = false;
  document.body.classList.add("modal-open");
  const search = modal.querySelector("input");
  search.value = "";
  filterGlossary(modal, "");
  if (focusKey) {
    const card = modal.querySelector(`[data-key="${focusKey}"]`);
    card?.scrollIntoView({ block: "center" });
    card?.classList.add("flash");
    setTimeout(() => card?.classList.remove("flash"), 1400);
  } else {
    modal.querySelector(".gl-body").scrollTop = 0;
    search.focus();
  }
}

function closeGlossary() {
  const modal = document.getElementById("glossary");
  if (modal) modal.hidden = true;
  if (!document.querySelector(".xray:not([hidden])")) document.body.classList.remove("modal-open");
}

function buildGlossary() {
  const modal = el("div", "modal glossary");
  modal.id = "glossary";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-label", "Glossary");
  const panel = el("div", "modal-panel gl-panel");
  const head = el("div", "modal-head");
  const titleWrap = el("div");
  titleWrap.append(el("h2", null, "Glossary"), el("p", "muted small", "Every term in this app, in plain English."));
  const close = el("button", "icon-btn");
  close.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  close.setAttribute("aria-label", "Close glossary");
  close.addEventListener("click", closeGlossary);
  head.append(titleWrap, close);

  const search = el("input", "gl-search");
  search.type = "search";
  search.placeholder = "Search terms…";
  search.addEventListener("input", () => filterGlossary(modal, search.value));

  const body = el("div", "gl-body");
  GLOSSARY_GROUPS.forEach((g, gi) => {
    const sec = el("section", "gl-group");
    sec.append(el("h3", null, g));
    for (const [key, t] of Object.entries(GLOSSARY)) {
      if (t.group !== gi) continue;
      const card = el("article", "gl-card");
      card.dataset.key = key;
      card.dataset.text = `${t.term} ${t.short} ${t.like}`.toLowerCase();
      card.append(el("h4", null, t.term), el("p", null, t.short));
      const like = el("p", "gl-like");
      like.append(el("span", null, "Think of it like"), document.createTextNode(" " + t.like));
      card.append(like);
      if (t.here) card.append(el("p", "gl-here", t.here));
      sec.append(card);
    }
    body.append(sec);
  });
  const empty = el("p", "muted gl-empty", "No matching terms.");
  empty.hidden = true;
  body.append(empty);

  panel.append(head, search, body);
  modal.append(panel);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeGlossary(); });
  document.body.append(modal);
  return modal;
}

function filterGlossary(modal, q) {
  q = q.trim().toLowerCase();
  let shown = 0;
  modal.querySelectorAll(".gl-card").forEach((c) => {
    const ok = !q || c.dataset.text.includes(q);
    c.hidden = !ok;
    shown += ok;
  });
  modal.querySelectorAll(".gl-group").forEach((s) => (s.hidden = !s.querySelector(".gl-card:not([hidden])")));
  modal.querySelector(".gl-empty").hidden = shown > 0;
}
