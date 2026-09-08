const pptxgen = require("pptxgenjs");
const OUT = "/Users/ankursharma/Documents/robot-semantic-memory/docs/results-deck.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5
pres.author = "Ankur Sharma";
pres.title = "Robot Semantic Memory — Results";

// palette: lifted from the demo's own HUD, so deck and product match
const BG      = "141A20";
const PANEL   = "1E2731";
const PANEL2  = "27323D";
const INK     = "E9EFF4";
const SOFT    = "9EADB8";
const FAINT   = "6E7D89";
const NV      = "76B900";
const CHEAP   = "58CDB5";
const ESC     = "E8A252";
const PROJ    = "B7A6DC";

const TITLE_F = "Cambria";
const BODY_F  = "Calibri";
const DATA_F  = "Courier New";

const W = 13.33, H = 7.5;
const M = 0.7;                                // side margin
const CW = W - 2 * M;                         // 11.93 usable

function slide(dark = true) {
  const s = pres.addSlide();
  s.background = { color: dark ? BG : "FFFFFF" };
  return s;
}

// recurring motif: a small filled dot before every eyebrow label
function eyebrow(s, text, y, color = NV) {
  s.addShape(pres.ShapeType.ellipse,
    { x: M, y: y + 0.055, w: 0.11, h: 0.11, fill: { color } });
  s.addText(text, {
    isTextBox: true, x: M + 0.22, y, w: CW - 0.22, h: 0.28, margin: 0,
    fontFace: DATA_F, fontSize: 11, bold: true, color, charSpacing: 1.5,
  });
}

function title(s, text, y = 1.05, size = 40) {
  s.addText(text, {
    isTextBox: true, x: M, y, w: CW, h: size > 34 ? 1.5 : 1.0, margin: 0,
    fontFace: TITLE_F, fontSize: size, bold: true, color: INK, lineSpacing: size * 1.1,
  });
}

function card(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06,
    fill: { color: o.fill || PANEL },
  });
}

// a stat card: big value, small label beneath
function stat(s, o) {
  card(s, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill });
  if (o.kicker) {
    s.addText(o.kicker, {
      isTextBox: true, x: o.x + 0.28, y: o.y + 0.24, w: o.w - 0.56, h: 0.26, margin: 0,
      fontFace: DATA_F, fontSize: 10, bold: true, color: o.kickerColor || FAINT,
      charSpacing: 1.2,
    });
  }
  s.addText(o.value, {
    isTextBox: true, x: o.x + 0.28, y: o.y + (o.kicker ? 0.56 : 0.32),
    w: o.w - 0.56, h: 0.85, margin: 0,
    fontFace: BODY_F, fontSize: o.vsize || 40, bold: true, color: o.color || INK,
  });
  s.addText(o.label, {
    isTextBox: true, x: o.x + 0.28, y: o.y + (o.kicker ? 1.42 : 1.18),
    w: o.w - 0.56, h: o.h - (o.kicker ? 1.6 : 1.36), margin: 0,
    fontFace: BODY_F, fontSize: 13, color: SOFT, lineSpacing: 17,
  });
}

/* ─────────────────────────── 1 · title ─────────────────────────── */
{
  const s = slide();
  eyebrow(s, "NVIDIA GTC BERLIN · GOLDEN TICKET SUBMISSION", 1.5);
  s.addText("Robot Semantic Memory", {
    isTextBox: true, x: M, y: 1.95, w: CW, h: 1.0, margin: 0,
    fontFace: TITLE_F, fontSize: 54, bold: true, color: INK,
  });
  s.addText("Matryoshka embeddings as a gate on expensive perception", {
    isTextBox: true, x: M, y: 3.0, w: 9.4, h: 0.5, margin: 0,
    fontFace: BODY_F, fontSize: 21, color: NV,
  });
  s.addText(
    "A robot shouldn't wake a large vision-language model to answer questions " +
    "it already knows the answer to. This is a cheap, truncatable visual memory " +
    "plus a gate that decides when the expensive model isn't needed.", {
    isTextBox: true, x: M, y: 3.75, w: 8.6, h: 1.3, margin: 0,
    fontFace: BODY_F, fontSize: 15, color: SOFT, lineSpacing: 24,
  });
  const chips = ["jina-clip-v2 · local, CPU", "NVIDIA NIM · VLM + embeddings",
                 "NVIDIA Riva · ASR + TTS"];
  chips.forEach((c, i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: 5.35 + i * 0.52, w: 4.6, h: 0.4, rectRadius: 0.05,
      fill: { color: PANEL },
    });
    s.addShape(pres.ShapeType.ellipse,
      { x: M + 0.2, y: 5.49 + i * 0.52, w: 0.12, h: 0.12, fill: { color: NV } });
    s.addText(c, {
      isTextBox: true, x: M + 0.45, y: 5.38 + i * 0.52, w: 4.0, h: 0.34, margin: 0,
      fontFace: DATA_F, fontSize: 11, color: INK,
    });
  });
  s.addText("github.com/ankursharma435/robot-semantic-memory", {
    isTextBox: true, x: M, y: 6.85, w: CW, h: 0.3, margin: 0,
    fontFace: DATA_F, fontSize: 12, color: FAINT,
  });
  s.addNotes("Opening slide. The one-line why is on this slide deliberately — " +
             "judges asked for the problem stated in one line.");
}

/* ─────────────────────────── 2 · problem ─────────────────────────── */
{
  const s = slide();
  eyebrow(s, "THE PROBLEM", 0.6);
  title(s, "A robot sees the same\nthings, over and over.", 1.0, 36);
  s.addText(
    "A warehouse robot stares at the same aisles all day — same shelves, same " +
    "boxes. But if every question goes to a large vision-language model, it pays " +
    "full price every time, for answers it already had.", {
    isTextBox: true, x: M, y: 3.5, w: 5.6, h: 1.6, margin: 0,
    fontFace: BODY_F, fontSize: 16, color: SOFT, lineSpacing: 26,
  });
  s.addText("Slow. Power-hungry on a battery. And it occupies the one " +
            "accelerator the robot has.", {
    isTextBox: true, x: M, y: 5.2, w: 5.6, h: 0.9, margin: 0,
    fontFace: BODY_F, fontSize: 16, italic: true, color: ESC, lineSpacing: 26,
  });

  // right: the cost of asking, three stacked rows
  const rows = [
    ["one 11B VLM call", "37.4 TFLOP", ESC],
    ["encode a query locally", "9 GFLOP", INK],
    ["look it up in memory", "128 kFLOP", CHEAP],
  ];
  card(s, { x: 6.9, y: 1.0, w: CW - 6.2, h: 4.2, fill: PANEL });
  s.addText("COST OF ANSWERING ONE QUESTION", {
    isTextBox: true, x: 7.2, y: 1.3, w: 5.2, h: 0.3, margin: 0,
    fontFace: DATA_F, fontSize: 10, bold: true, color: FAINT, charSpacing: 1.2,
  });
  rows.forEach((r, i) => {
    const y = 1.85 + i * 1.05;
    s.addText(r[0], {
      isTextBox: true, x: 7.2, y, w: 5.2, h: 0.3, margin: 0,
      fontFace: BODY_F, fontSize: 14, color: SOFT,
    });
    s.addText(r[1], {
      isTextBox: true, x: 7.2, y: y + 0.32, w: 5.2, h: 0.5, margin: 0,
      fontFace: BODY_F, fontSize: 26, bold: true, color: r[2],
    });
  });
  s.addText("Estimated as 2 × params × tokens.", {
    isTextBox: true, x: 6.9, y: 5.35, w: CW - 6.2, h: 0.3, margin: 0,
    fontFace: DATA_F, fontSize: 10, color: FAINT,
  });
  s.addNotes("The three-row card sets up the whole argument: the gap between " +
             "looking something up and asking a big model is five orders of magnitude.");
}

/* ─────────────────────────── 3 · the idea ─────────────────────────── */
{
  const s = slide();
  eyebrow(s, "THE IDEA");
  title(s, "One encode. Three memory tiers.", 1.0, 36);
  s.addText(
    "Matryoshka Representation Learning trains the encoder so that any prefix of " +
    "its output is itself a valid embedding. One forward pass produces 1024 " +
    "dimensions; the first 64, 256 or 768 each still mean something alone.", {
    isTextBox: true, x: M, y: 2.15, w: 11.0, h: 1.0, margin: 0,
    fontFace: BODY_F, fontSize: 16, color: SOFT, lineSpacing: 26,
  });

  const tiers = [
    ["short", "64d", "256 B", "what I just saw\ndecays after 30s", CHEAP],
    ["medium", "256d", "1 KB", "what I've learned\nlabelled, consolidated", INK],
    ["long", "768d", "3 KB", "what I know well\ndurable", INK],
  ];
  const cw = 3.3, gap = 0.42;
  tiers.forEach((t, i) => {
    const x = M + i * (cw + gap);
    card(s, { x, y: 3.35, w: cw, h: 2.15, fill: PANEL });
    s.addText(t[0], {
      isTextBox: true, x: x + 0.26, y: 3.58, w: cw - 0.52, h: 0.32, margin: 0,
      fontFace: DATA_F, fontSize: 12, bold: true, color: t[4], charSpacing: 1.2,
    });
    s.addText(t[1], {
      isTextBox: true, x: x + 0.26, y: 3.92, w: cw - 0.52, h: 0.6, margin: 0,
      fontFace: BODY_F, fontSize: 34, bold: true, color: INK,
    });
    s.addText(t[2] + " per frame", {
      isTextBox: true, x: x + 0.26, y: 4.52, w: cw - 0.52, h: 0.3, margin: 0,
      fontFace: DATA_F, fontSize: 12, color: NV,
    });
    s.addText(t[3], {
      isTextBox: true, x: x + 0.26, y: 4.83, w: cw - 0.52, h: 0.6, margin: 0,
      fontFace: BODY_F, fontSize: 12, color: SOFT, lineSpacing: 16,
    });
  });

  card(s, { x: M, y: 5.75, w: CW, h: 1.05, fill: PANEL2 });
  s.addText([
    { text: "Truncation does not make the encoder cheaper. ", options: { bold: true, color: ESC } },
    { text: "The backbone runs once regardless of how many dimensions you keep. " +
            "What truncation buys is cheaper storage and cheaper comparison.",
      options: { color: SOFT } },
  ], {
    isTextBox: true, x: M + 0.3, y: 5.98, w: CW - 0.6, h: 0.6, margin: 0,
    fontFace: BODY_F, fontSize: 14, lineSpacing: 20,
  });
  s.addNotes("The caveat is on the slide on purpose. It is the first thing a " +
             "knowledgeable reader tests, and volunteering it buys credibility.");
}

/* ─────────────────────────── 4 · the gate ─────────────────────────── */
{
  const s = slide();
  eyebrow(s, "THE GATE");
  title(s, "Knowing when not to ask.", 1.0, 36);

  const cols = [
    ["CHEAP", CHEAP, "Answer from stored memory",
     ["match clears that tier's threshold",
      "returns a label and coordinates",
      "0.058 ms measured",
      "expensive model never invoked"]],
    ["ESCALATE", ESC, "Wake the expensive model",
     ["nothing learned is close enough",
      "or the task needs manipulation",
      "or the match came only from what's in view",
      "sends the live frame to an NVIDIA NIM VLM"]],
  ];
  cols.forEach((c, i) => {
    const x = M + i * (5.95 + 0.4);
    card(s, { x, y: 2.3, w: 5.95, h: 3.6, fill: PANEL });
    s.addShape(pres.ShapeType.roundRect, {
      x: x + 0.3, y: 2.6, w: 1.85, h: 0.46, rectRadius: 0.06,
      fill: { color: PANEL2 },
    });
    s.addText(c[0], {
      isTextBox: true, x: x + 0.3, y: 2.68, w: 1.85, h: 0.32, margin: 0,
      fontFace: DATA_F, fontSize: 15, bold: true, color: c[1], align: "center",
    });
    s.addText(c[2], {
      isTextBox: true, x: x + 0.3, y: 3.22, w: 5.35, h: 0.36, margin: 0,
      fontFace: BODY_F, fontSize: 17, bold: true, color: INK,
    });
    s.addText(c[3].map((t, j) => ({
      text: t, options: { bullet: true, breakLine: j < c[3].length - 1 },
    })), {
      isTextBox: true, x: x + 0.3, y: 3.75, w: 5.35, h: 1.9, margin: 0,
      fontFace: BODY_F, fontSize: 14, color: SOFT, paraSpaceAfter: 8,
    });
  });

  s.addText(
    "Cosine scores from different truncation widths are not comparable — a narrow " +
    "prefix systematically scores higher. So each tier carries its own threshold, " +
    "calibrated from photographs, and a match found only in the always-watching " +
    "tier never counts as something learned.", {
    isTextBox: true, x: M, y: 6.15, w: CW, h: 0.9, margin: 0,
    fontFace: BODY_F, fontSize: 13.5, color: FAINT, lineSpacing: 20,
  });
}

/* ─────────────────────────── 5 · measured, tiers ─────────────────── */
{
  const s = slide();
  eyebrow(s, "MEASURED · 11 PHOTOGRAPHS, FOUR OBJECTS", 0.6, CHEAP);
  title(s, "Every tier got every retrieval right.", 1.0, 34);

  s.addTable([
    [{ text: "Tier", options: { bold: true, color: FAINT, fontSize: 11, fontFace: DATA_F } },
     { text: "Recall@1", options: { bold: true, color: FAINT, fontSize: 11, fontFace: DATA_F } },
     { text: "Margin over best wrong match", options: { bold: true, color: FAINT, fontSize: 11, fontFace: DATA_F } },
     { text: "Calibrated threshold", options: { bold: true, color: FAINT, fontSize: 11, fontFace: DATA_F } }],
    ["long  768d", "100%", "+0.043", "0.331"],
    ["medium  256d", "100%", "+0.038", "0.361"],
    ["short  64d", "100%", "+0.013", "0.416"],
  ], {
    x: M, y: 2.25, w: CW, colW: [2.6, 2.2, 4.5, 2.63],
    rowH: 0.46, fontFace: BODY_F, fontSize: 15, color: INK,
    fill: { color: PANEL }, border: { type: "solid", color: BG, pt: 2 },
    valign: "middle",
  });

  card(s, { x: M, y: 4.5, w: CW, h: 2.1, fill: PANEL2 });
  s.addText("What the tiers actually showed", {
    isTextBox: true, x: M + 0.32, y: 4.75, w: CW - 0.64, h: 0.34, margin: 0,
    fontFace: BODY_F, fontSize: 17, bold: true, color: INK,
  });
  s.addText([
    { text: "Colour survives truncation, so two same-shape mugs separated cleanly " +
            "even at 64 dimensions. What shrinks is " },
    { text: "margin", options: { bold: true, color: CHEAP } },
    { text: " — the gap to the nearest wrong answer collapses from +0.043 to +0.013. " +
            "Three times less headroom for the same accuracy. That is the real cost of " +
            "truncation here, and it is why each tier needs its own threshold." },
  ], {
    isTextBox: true, x: M + 0.32, y: 5.18, w: CW - 0.64, h: 1.2, margin: 0,
    fontFace: BODY_F, fontSize: 14.5, color: SOFT, lineSpacing: 22,
  });
  s.addNotes("Do not claim the 64-dim tier fails. On this set it did not. " +
             "The margin narrowing is the honest finding.");
}

/* ─────────────────────────── 6 · measured, cost ──────────────────── */
{
  const s = slide();
  eyebrow(s, "MEASURED · THIS RUN", 0.6, CHEAP);
  title(s, "Remembering versus asking.", 1.0, 36);

  stat(s, { x: M, y: 2.35, w: 3.75, h: 2.5, kicker: "MEMORY LOOKUP",
            kickerColor: CHEAP, value: "0.058 ms",
            label: "Brute-force cosine over the learned tiers. No network, no GPU.",
            color: CHEAP, vsize: 36 });
  stat(s, { x: M + 4.09, y: 2.35, w: 3.75, h: 2.5, kicker: "ESCALATION",
            kickerColor: ESC, value: "1–6 s",
            label: "Round trip to an 11B vision model on an NVIDIA NIM.",
            color: ESC, vsize: 36 });
  stat(s, { x: M + 8.18, y: 2.35, w: 3.75, h: 2.5, kicker: "PER FRAME STORED",
            value: "256 B", label: "At 64 dimensions, against 8192 bytes for a " +
            "fixed-width 2048-dim baseline.", vsize: 36 });

  card(s, { x: M, y: 5.2, w: CW, h: 1.5, fill: PANEL2 });
  s.addText([
    { text: "~4,200×", options: { bold: true, fontSize: 30, color: CHEAP } },
    { text: "  fewer operations to answer from memory than to wake the " +
            "expensive model — and 32× less storage per frame.",
      options: { fontSize: 16, color: SOFT } },
  ], {
    isTextBox: true, x: M + 0.32, y: 5.62, w: CW - 0.64, h: 0.8, margin: 0,
    fontFace: BODY_F, lineSpacing: 26,
  });
}

/* ─────────────────────────── 7 · projected, storage ──────────────── */
{
  const s = slide();
  eyebrow(s, "PROJECTED · WAREHOUSE FLEET", 0.6, PROJ);
  title(s, "92 gigabytes, not three terabytes.", 1.0, 36);
  s.addText("One frame per second of continuous perception, eight-hour shifts, " +
            "250 shifts a year. Projected from directly measured bytes per frame.", {
    isTextBox: true, x: M, y: 2.15, w: 11.2, h: 0.6, margin: 0,
    fontFace: BODY_F, fontSize: 15, color: SOFT, lineSpacing: 22,
  });

  // Proportional bars drawn as shapes rather than a native chart.
  // pptxgenjs emitted an axId it never declared (2094734556) for this chart,
  // which is the fault PowerPoint responds to by discarding the chart and
  // calling the file corrupt; its valAxisLogScale option was also silently
  // ignored, leaving a linear scale across six orders of magnitude. Shapes
  // give full control of the scale and carry no chart XML at all.
  const BARMAX = 8.6;                       // widest bar, inches
  const pairs = [
    ["One robot, one 8-hour shift", 7, 236, "MB"],
    ["One robot, one year", 1.8, 59, "GB"],
    ["50-robot fleet, one year", 92, 2949, "GB"],
  ];
  pairs.forEach((p, i) => {
    const y = 2.95 + i * 1.32;
    const [lab, a, b, unit] = p;
    s.addText(lab, {
      isTextBox: true, x: M, y, w: 6.0, h: 0.28, margin: 0,
      fontFace: BODY_F, fontSize: 14, bold: true, color: INK,
    });
    // The ratio is 2048/64 = 32 exactly, the same at every scale. Deriving it
    // from the rounded display values instead gave 34x and 33x, which reads as
    // sloppy against the 32x stated elsewhere in the deck.
    s.addText("32x less", {
      isTextBox: true, x: M + 9.9, y, w: 2.0, h: 0.28, margin: 0,
      fontFace: DATA_F, fontSize: 13, bold: true, color: NV, align: "right",
    });
    // fixed-width bar (full length), tiered bar scaled against it
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: y + 0.34, w: BARMAX, h: 0.3, rectRadius: 0.03,
      fill: { color: ESC },
    });
    s.addText(`${b} ${unit}`, {
      isTextBox: true, x: M + BARMAX + 0.14, y: y + 0.33, w: 1.5, h: 0.3, margin: 0,
      fontFace: DATA_F, fontSize: 12, color: ESC, valign: "middle",
    });
    const wa = Math.max(0.06, BARMAX * (a / b));
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: y + 0.70, w: wa, h: 0.3, rectRadius: 0.03,
      fill: { color: CHEAP },
    });
    s.addText(`${a} ${unit}`, {
      isTextBox: true, x: M + wa + 0.14, y: y + 0.69, w: 1.6, h: 0.3, margin: 0,
      fontFace: DATA_F, fontSize: 12, bold: true, color: CHEAP, valign: "middle",
    });
  });

  // legend
  [["fixed-width 2048d", ESC, 0], ["tiered 64/256/768", CHEAP, 3.3]].forEach(l => {
    s.addShape(pres.ShapeType.roundRect, {
      x: M + l[2], y: 6.95, w: 0.28, h: 0.14, rectRadius: 0.02,
      fill: { color: l[1] },
    });
    s.addText(l[0], {
      isTextBox: true, x: M + l[2] + 0.4, y: 6.86, w: 3.0, h: 0.3, margin: 0,
      fontFace: BODY_F, fontSize: 12, color: SOFT,
    });
  });

  s.addNotes("Storage is the cleanest projection in the deck — it is arithmetic " +
             "from measured bytes per frame, derivable on a napkin.");
}

/* ─────────────────────────── 8 · projected, compute ──────────────── */
{
  const s = slide();
  eyebrow(s, "PROJECTED · COMPUTE AND ENERGY", 0.6, PROJ);
  title(s, "Five times less perception compute.", 1.0, 36);
  s.addText("If the gate answers eight queries in ten — realistic for a robot " +
            "working a familiar space — then across 100 queries:", {
    isTextBox: true, x: M, y: 2.15, w: 11.2, h: 0.6, margin: 0,
    fontFace: BODY_F, fontSize: 15, color: SOFT, lineSpacing: 22,
  });

  stat(s, { x: M, y: 2.85, w: 3.75, h: 2.3, kicker: "ALWAYS ESCALATE",
            kickerColor: ESC, value: "3,740 TFLOP",
            label: "Every query wakes the 11B model.", color: ESC, vsize: 30 });
  stat(s, { x: M + 4.09, y: 2.85, w: 3.75, h: 2.3, kicker: "GATED AT 80%",
            kickerColor: CHEAP, value: "749 TFLOP",
            label: "80 answered from memory, 20 escalated.", color: CHEAP, vsize: 30 });
  stat(s, { x: M + 8.18, y: 2.85, w: 3.75, h: 2.3, kicker: "REDUCTION",
            kickerColor: NV, value: "5.0×", vsize: 44,
            label: "Less perception compute — and, on fixed silicon, roughly " +
                   "5× less perception energy.", color: NV });

  card(s, { x: M, y: 5.5, w: CW, h: 1.25, fill: PANEL2 });
  s.addText([
    { text: "Energy is inferred, not measured. ", options: { bold: true, color: ESC } },
    { text: "No wattage was recorded on this laptop, and none on a Jetson. " +
            "Energy is assumed to track compute on fixed silicon. The claim that " +
            "needs no assumption: every question memory answers is a model " +
            "invocation that never happens at all.", options: { color: SOFT } },
  ], {
    isTextBox: true, x: M + 0.32, y: 5.75, w: CW - 0.64, h: 0.8, margin: 0,
    fontFace: BODY_F, fontSize: 14, lineSpacing: 21,
  });
}

/* ─────────────────────────── 9 · not claimed ─────────────────────── */
{
  const s = slide();
  eyebrow(s, "SCOPE", 0.6, ESC);
  title(s, "What this does not claim.", 1.0, 36);

  const items = [
    ["That the cheap path is faster per call.",
     "Measured false here — local encode ~1.6 s against a hosted VLM at ~1.0 s. " +
     "The expensive path runs on datacentre GPUs while the encoder runs on laptop " +
     "CPU. The claim is calls avoided, not latency won."],
    ["That truncation saves encoder compute.",
     "It does not. The backbone runs once regardless of dimensions kept."],
    ["That anything ran on a Jetson, TensorRT, or Isaac GR00T.",
     "None did. The hosted VLM is a stand-in for the escalation target."],
    ["A measured power figure.",
     "None was taken. Energy is inferred from compute and labelled as such."],
  ];
  items.forEach((it, i) => {
    const y = 2.2 + i * 1.16;
    s.addShape(pres.ShapeType.ellipse,
      { x: M, y: y + 0.09, w: 0.14, h: 0.14, fill: { color: ESC } });
    s.addText(it[0], {
      isTextBox: true, x: M + 0.36, y, w: CW - 0.36, h: 0.32, margin: 0,
      fontFace: BODY_F, fontSize: 16.5, bold: true, color: INK,
    });
    s.addText(it[1], {
      isTextBox: true, x: M + 0.36, y: y + 0.36, w: CW - 0.5, h: 0.72, margin: 0,
      fontFace: BODY_F, fontSize: 13.5, color: SOFT, lineSpacing: 20,
    });
  });

  s.addText("Stating which figures are measured, which projected and which " +
            "inferred is the point — not a disclaimer.", {
    isTextBox: true, x: M, y: 6.85, w: CW, h: 0.4, margin: 0,
    fontFace: BODY_F, fontSize: 13.5, italic: true, color: NV,
  });
}

/* ─────────────────────────── 10 · stack & next ───────────────────── */
{
  const s = slide();
  eyebrow(s, "THE STACK");
  title(s, "Running today, and what's next.", 1.0, 36);

  const now = [
    ["jina-clip-v2", "local · CPU · open weights, CC BY-NC"],
    ["llama-nemotron-embed-vl-1b-v2", "NVIDIA NIM · fixed-width baseline, 2048d"],
    ["llama-3.2-11b-vision-instruct", "NVIDIA NIM · escalation target"],
    ["parakeet-ctc-0.6b · magpie-tts", "NVIDIA Riva · speech in and out"],
  ];
  const next = [
    ["Jetson Orin Nano Super", "encoder and memory move on-device"],
    ["TensorRT", "collapses the encoder latency that dominates today"],
    ["Isaac GR00T N1", "replaces the VLM stand-in with a real VLA"],
  ];

  card(s, { x: M, y: 2.15, w: 5.95, h: 3.5, fill: PANEL });
  s.addText("RUNNING AND MEASURED", {
    isTextBox: true, x: M + 0.3, y: 2.42, w: 5.35, h: 0.3, margin: 0,
    fontFace: DATA_F, fontSize: 10, bold: true, color: CHEAP, charSpacing: 1.2,
  });
  now.forEach((n, i) => {
    const y = 2.85 + i * 0.68;
    s.addText(n[0], {
      isTextBox: true, x: M + 0.3, y, w: 5.35, h: 0.28, margin: 0,
      fontFace: DATA_F, fontSize: 12.5, bold: true, color: INK,
    });
    s.addText(n[1], {
      isTextBox: true, x: M + 0.3, y: y + 0.27, w: 5.35, h: 0.26, margin: 0,
      fontFace: BODY_F, fontSize: 12, color: SOFT,
    });
  });

  card(s, { x: M + 6.35, y: 2.15, w: 5.58, h: 3.5, fill: PANEL2 });
  s.addText("NEXT STEP · NOT YET RUN", {
    isTextBox: true, x: M + 6.65, y: 2.42, w: 4.98, h: 0.3, margin: 0,
    fontFace: DATA_F, fontSize: 10, bold: true, color: ESC, charSpacing: 1.2,
  });
  next.forEach((n, i) => {
    const y = 2.85 + i * 0.68;
    s.addText(n[0], {
      isTextBox: true, x: M + 6.65, y, w: 4.98, h: 0.28, margin: 0,
      fontFace: DATA_F, fontSize: 12.5, bold: true, color: SOFT,
    });
    s.addText(n[1], {
      isTextBox: true, x: M + 6.65, y: y + 0.27, w: 4.98, h: 0.26, margin: 0,
      fontFace: BODY_F, fontSize: 12, color: FAINT,
    });
  });
  s.addText("The gate does not know what sits behind it — swapping the escalation " +
            "target is one function call.", {
    isTextBox: true, x: M + 6.65, y: 5.0, w: 4.98, h: 0.5, margin: 0,
    fontFace: BODY_F, fontSize: 12.5, italic: true, color: NV, lineSpacing: 18,
  });

  s.addText("44 unit tests · every measurement reproducible · " +
            "github.com/ankursharma435/robot-semantic-memory", {
    isTextBox: true, x: M, y: 6.2, w: CW, h: 0.4, margin: 0,
    fontFace: DATA_F, fontSize: 12, color: FAINT,
  });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
