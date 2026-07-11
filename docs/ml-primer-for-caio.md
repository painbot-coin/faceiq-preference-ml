# ML for a CAIO — a field guide grounded in this repo

Everything in this document is illustrated with real code and real decisions from
**faceiq-preference-ml**, which trains models to rank facial attractiveness from
pairwise comparisons ("which of these two photos is more attractive?"). The repo is
small but it touches nearly every concept that matters: labels, leakage, transfer
learning, loss functions, evaluation, calibration, and experiment discipline. Where
the repo doesn't cover something (LLMs, deployment), the sections say so and explain
the concept standalone.

The goal is not to make you a practitioner. It's to make you the person in the room
who asks the question that saves three months: *"how was the validation set split?"*,
*"what's the baseline?"*, *"who audited the labels?"*

---

## 1. What machine learning actually is

Traditional software: humans write rules, computers apply them to data.
Machine learning: humans supply data and a **goal (loss function)**, and an
optimization algorithm finds the rules (**parameters/weights**) automatically.

The consequence that drives everything else: **the model is only as good as the data
and the objective**. You don't debug ML systems by reading the rules — there are
millions of numeric parameters, not human-readable logic. You debug them by
interrogating the data, the objective, and the evaluation. That is why most of this
document is about data and evaluation, not about network architectures.

**In this repo:** nobody wrote a rule like "symmetric faces score higher." Instead,
52,414 human/VLM judgments of "A beats B" were collected, and two very different
systems learned from them:

- a **statistical model** (Bradley-Terry, `src/faceiq_pref/bt.py`) that assigns each
  face a strength number so the observed wins are maximally probable, and
- a **neural network** (`src/faceiq_pref/model.py`) that learns to predict the winner
  from raw pixels.

Same data, two models, two purposes. Understanding *why both exist* is half of ML
literacy — see §4.

---

## 2. The taxonomy: kinds of learning

| Paradigm | What the data looks like | Example |
|---|---|---|
| **Supervised learning** | inputs + correct answers (labels) | This repo: (photoA, photoB) → winner |
| **Self-supervised learning** | raw data only; the model invents its own labels from structure | DINOv2, used as a backbone here — trained on unlabeled images to make similar crops embed similarly |
| **Unsupervised learning** | raw data; find structure (clusters, dimensions) | Clustering faces by appearance without labels |
| **Reinforcement learning** | an agent acts, gets rewards, improves its policy | Game playing, robotics, RLHF for LLMs |

The pairwise setup in this repo is a flavor of supervised learning called
**learning to rank** (the specific loss is the classic **RankNet** objective — the
research log §5 explicitly names it and documents why the alternative, LambdaRank,
was rejected). Preference pairs, not absolute scores, is also exactly how modern
LLM alignment works (RLHF reward models are trained on "response A is better than
response B"), so this repo's core mechanic is the same one behind ChatGPT's
fine-tuning.

**Nuance to remember:** asking humans (or VLMs) "which is better, A or B?" produces
far more reliable labels than "rate this 1–10." Absolute ratings drift with mood,
anchoring, and rater identity; comparisons are stable. This project's entire design —
collect comparisons, fit Bradley-Terry, *then* map to a /10 scale — exists because of
this psychometric fact.

---

## 3. Data: the part that determines whether anything works

### 3.1 Ground truth and label provenance

"Ground truth" (GT) is the set of answers you treat as correct. It is never actually
perfect — it's a *decision* about what to trust. This repo makes that decision
explicit and machine-checked:

- Each matchup row carries both a `vlmOutcome` (label from Gemini 2.5 Flash) and,
  for ~1,032 audited rows, a `humanOutcome`.
- The **label rule** is written down and frozen: use the human label when a human
  reviewed the pair, else the VLM label (`finalOutcome`).
- `src/faceiq_pref/data.py` **re-derives and asserts** the rule on every load
  (`_assert_finalize_rule`). If an export ever disagrees with the rule, loading fails
  loudly instead of silently training on wrong labels.

**CAIO takeaway:** for any ML initiative, ask *"what is the label rule, where is it
written down, and what enforces it?"* Silent label drift is one of the most common
ways production models rot.

### 3.2 Label noise and model-as-judge

Labels here come mostly from a **VLM (vision-language model) used as a judge** —
an LLM-era pattern for scaling labeling cheaply. The discipline around it matters
more than the trick itself:

- Humans labeled a pilot set *first*; VLMs were then scored against humans (~85%
  agreement for the chosen model at ~$0.004/call — the research log records the
  pass/fail gate before running the experiment).
- A three-model **panel/ensemble** was tested and rejected: unanimity was only ~71%
  and accuracy didn't justify 3× cost. Ensembles (averaging several models) usually
  help accuracy but always multiply cost; measuring whether the lift is worth it is
  exactly the right instinct.
- After full labeling, a **seeded human audit** of 750 pairs measured 84.9% agreement
  — that number bounds how good any model trained on these labels can look.

**Nuance:** ~15% label noise is fine *if it's random* — both BT and the neural model
average over thousands of comparisons per conclusion. It's dangerous if it's
*systematic* (e.g., the VLM consistently favoring a demographic or lighting style),
because models learn systematic bias perfectly. This is why the audit sample was
seeded/stratified rather than casual spot-checking.

### 3.3 Data integrity and provenance

`load_export()` verifies row counts and **sha256 hashes** of every data shard against
a manifest before anything downstream runs. Truncated file, partial copy, hand-edited
row — the pipeline refuses to start.

This sounds like plumbing, but bad-data-in-silently is the #1 unglamorous failure
mode of ML systems. "Always load through the verified entry point" is a hard rule of
this repo for that reason.

### 3.4 Data governance

Note the repo's other hard rule: face photos and model checkpoints never enter git
history. Training data is often the most sensitive asset an AI org holds (privacy,
copyright, biometric regulation). Where data lives, who can export it, and what's
reproducible from what — these are CAIO-level questions, not IT details.

---

## 4. Two families of models — and why you often want both

### 4.1 Classical/statistical models: Bradley-Terry

**Bradley-Terry (BT)** is a 1952 statistical model: give each face \(i\) a strength
\(\theta_i\), and say P(i beats j) = sigmoid(\(\theta_i - \theta_j\)). Fitting means
finding the \(\theta\) values under which the observed 52k outcomes are most probable
— **maximum likelihood estimation (MLE)**. It's the same math family as chess Elo
ratings. `src/faceiq_pref/bt.py` implements it in ~180 lines with no GPU.

Why use a 70-year-old model when you have a neural net? Because BT is:

- **Exact for its job.** It's the theoretically right way to turn pairwise outcomes
  into a ranking of the *faces you have data for*.
- **Interpretable.** Each face's score is directly explainable from its win/loss
  record.
- **Cheap and stable.** Seconds to fit, deterministic, easy to re-audit.

Its limitation: BT **cannot score a new face**. It only ranks entities it observed in
comparisons. That's what the neural model is for — it learns from pixels, so it
generalizes to unseen photos. Here, **BT is the authoritative ranking and the neural
model's evaluation target**; the neural model is the deployable artifact.

**CAIO takeaway:** "do we even need deep learning for this?" is always a legitimate
question. When the answer is yes, a classical model often still belongs in the
system as the ground-truth generator, baseline, or sanity check.

### 4.2 Neural networks in one paragraph

A neural network is a stack of layers, each doing a linear transform (matrix
multiply) followed by a simple nonlinearity (like **ReLU**, which zeroes negatives).
Stacked deep enough, this can approximate essentially any function. The numbers in
the matrices are the **parameters/weights** — ResNet-18 has ~11 million, ResNet-50
~25 million, frontier LLMs hundreds of billions. Training means nudging all of them,
repeatedly, in the direction that reduces the loss (§6).

### 4.3 Architecture concepts you'll hear constantly

All are visible in `src/faceiq_pref/model.py`:

- **Backbone + head.** The *backbone* (ResNet or DINOv2) turns an image into a
  compact vector; the *head* (a tiny 2-layer network) turns that vector into the
  task-specific output — here, one scalar "attractiveness score." Big generic
  backbone + small task head is the dominant pattern across all of applied deep
  learning, including LLMs.
- **Embedding.** The backbone's output vector (512 numbers for ResNet-18, 384 for
  DINOv2 ViT-S) is an *embedding*: a learned coordinate system where semantic
  similarity becomes geometric closeness. Embeddings power search, recommendations,
  RAG, clustering — arguably the most commercially useful single concept in ML.
- **CNN vs Transformer.** ResNet is a **convolutional network** (slides small
  filters across the image; built-in bias toward local visual patterns). DINOv2 is a
  **Vision Transformer (ViT)** — the same attention architecture as LLMs, cutting
  the image into 14×14-pixel patches and treating them like tokens. Transformers
  scale better with data and compute; that's why they've taken over.
- **Siamese architecture.** One scorer network, applied to *both* photos with the
  **exact same weights**; the prediction is the score difference
  (`PairwiseModel.forward` returns `s(A) − s(B)`). This guarantees judging is
  symmetric — swap the photos and the prediction flips its sign by construction.
  The research log flags the classic silent failure of this pattern: **positional
  bias** (if left/right presentation weren't randomized during labeling, the model
  could learn "left wins more often" instead of anything about faces).

### 4.4 Transfer learning: never start from zero

`pretrained=True` in the model constructor is doing enormous work. Instead of
learning what edges, textures, and faces are from 52k pairs (hopeless), the backbone
arrives already trained on millions of images and gets *adapted*. Three regimes,
all present in this repo's configs:

| Regime | What trains | Repo example | When |
|---|---|---|---|
| From scratch | everything, from random init | not used here (would fail) | Only with massive data |
| **Fine-tuning** | everything, starting from pretrained weights | `configs/train-v1/v2.yaml` | Enough data, need max quality |
| **Linear probe** (frozen backbone) | only the small head | `configs/train-v3-dinov2-probe.yaml` (`freeze_backbone: true`) | Fast signal check, small data |

The v3 probe is an idiom worth knowing by name: *"do the pretrained embeddings
already contain the signal we need?"* If a frozen DINOv2 plus a tiny head gets close
to the fine-tuned model, most of the value came free from the foundation model — a
finding with direct budget implications. This is precisely the reasoning behind
"buy a foundation model API vs. train our own" at any company.

Note also the interplay of choices: the probe config uses a **10× higher learning
rate** than the fine-tuning configs, because only a small head is learning.
Hyperparameters are not independent knobs.

---

## 5. The objective: loss functions, probabilities, logits

The model must be given a single number to minimize. Here:

- The raw output `s(A) − s(B)` is a **logit** — an unbounded real number.
- **Sigmoid** squashes the logit into a probability in (0, 1): P(A wins).
- **Binary cross-entropy (BCE)** penalizes confident wrong predictions harshly and
  confident right ones barely at all. `nn.BCEWithLogitsLoss` in `train.py` fuses the
  sigmoid and BCE for numerical stability.

Two nuances with general applicability:

1. **The loss encodes the product decision.** Ties are labeled 0.5 but skipped by
   default (`skip_ties: true`) — a deliberate, documented modeling choice. The
   research log likewise documents choosing RankNet over LambdaRank because the
   product goal is a well-calibrated score across the *whole* distribution, not
   top-k retrieval accuracy. When an ML team argues about a loss function, they are
   really arguing about what the product should optimize. Sit in that meeting.
2. **Goodhart's law applies to models.** The model optimizes the loss *exactly as
   written*, including its loopholes. If any feature correlates with winning in the
   training data — watermark, background, camera quality — the model will use it.
   This is "spurious correlation" / "shortcut learning," and it's why evaluation
   design (§7–8) matters more than architecture.

---

## 6. Training mechanics: what's actually happening in the loop

The whole ritual is ~40 lines in `train.py` (`train()`), and every modern model from
here to GPT is trained by the same loop:

1. Take a **batch** (here 64 pairs) — a compromise between noisy gradients (small
   batch) and memory/compute (large batch).
2. Forward pass → predictions → loss.
3. **Backpropagation** (`loss.backward()`): compute, for every one of the millions of
   parameters, the direction to nudge it to reduce the loss (the **gradient**).
4. **Optimizer step**: nudge them. This is **stochastic gradient descent (SGD)**;
   the variant here, **AdamW**, adapts step size per parameter and is the default
   for most work today.
5. Repeat over the whole dataset (one pass = an **epoch**; configs use 8–10), and
   after each epoch measure performance on held-out data.

The knobs you'll hear about, all in `TrainConfig`:

- **Learning rate (`lr`)** — step size. The single most important hyperparameter.
  Too high: training diverges. Too low: it crawls or gets stuck.
- **Weight decay** — a penalty on large weights, i.e. **regularization** — pressure
  toward simpler solutions that generalize. Note `train-v2.yaml`'s comment: v1
  overfit after epoch 7, so v2 raised weight decay 5× and shortened training. That's
  the textbook diagnosis→treatment loop in one YAML comment.
- **Dropout** (`model.py`, `nn.Dropout(0.2)`) — randomly silence 20% of neurons
  during training so no single pathway is over-relied on. Regularization again.
- **Data augmentation** — random transformations of training inputs (here, random
  horizontal flips) to teach invariances for free. Note the safety comment in
  `train.py`: flips are safe *because presentation order was randomized upstream at
  labeling*. Augmentation must respect the semantics of the label — flipping is fine
  for faces, disastrous for reading text.
- **Hardware**: `pick_device()` chooses CUDA (NVIDIA GPU) → MPS (Apple silicon) →
  CPU. Training is embarrassingly parallel matrix math; GPUs are 10–100× faster,
  which is why compute is a board-level line item in AI strategy.

**Checkpointing / early-stopping** in the loop: after each epoch, the model is saved
*only if validation accuracy improved* (`best.pt`). You deploy the best model along
the way, not whatever the last epoch left behind — because later epochs frequently
get *worse* on held-out data. Which brings us to the most important section in this
document.

---

## 7. Generalization: the core problem of the entire field

**Overfitting** is memorizing the training data instead of learning the underlying
pattern — perfect on data it's seen, useless on new data. **Underfitting** is the
opposite: too simple or undertrained to capture the pattern at all. Every
regularization tool in §6 exists to fight overfitting; every training curve you'll
ever be shown (train loss vs. validation loss, per epoch — this repo writes them to
`artifacts/<run>/metrics.json`) is read by looking for the point where validation
performance stops improving while training performance keeps going.

### 7.1 The train/validation split — and leakage, the silent killer

You measure generalization by hiding data from training (**validation set**) and
scoring on it. The measurement is only honest if there is no information flow from
validation into training. When there is, it's called **leakage**, and it is the
most common way ML results turn out to be fake.

This repo has a beautiful, concrete example — it's a **hard rule in CLAUDE.md** and
implemented in `split_by_face_id()` in `data.py`:

> Split train/val **by face id**, never by random pair.

Why? Each face appears in ~35 different pairs. Split randomly *by pair*, and the same
face lands in both training and validation. The model can then partially memorize
"face #1234 usually wins" during training and cash that in during validation — the
accuracy measures **identity memorization, not attractiveness understanding**.
Splitting by face means validation faces are *never seen in training*, so the metric
answers the question the business actually cares about: does this work on new faces?

Note the cost, accepted deliberately: pairs mixing a train face with a val face are
**thrown away entirely**. Honest evaluation is routinely worth discarding data for.

**CAIO takeaway:** when someone reports a great offline metric, your first question
is *"what's the unit of splitting, and can information about the same underlying
entity (user, patient, document, face) appear on both sides?"* Time-based leakage
(training on the future, validating on the past) is the same disease in temporal
form. Teams with too-good-to-be-true numbers almost always have a leak.

### 7.2 Distribution shift

A model is only valid on data resembling its training distribution. The research log
has a live instance: VLM labeling accuracy was ~85% on curated pilot pairs but
~66–73% on pairs drawn from the raw 20k-face pool — harder, messier matchups. Nothing
"broke"; the distribution changed. In production this appears as silent decay when
the world drifts away from the training snapshot (new demographics, new camera
styles, new user behavior). The countermeasure is monitoring live performance, not
trusting launch metrics forever.

---

## 8. Evaluation: metrics, baselines, calibration, gates

### 8.1 Metrics used here, and their general lessons

- **Pairwise accuracy** — fraction of held-out matchups predicted correctly
  (`evaluate()` in `train.py`). Interpretable, but always ask two questions of any
  accuracy number: *what's the chance baseline?* (here 50% — a coin flip) and
  *what's the ceiling?* (here roughly the ~85% human–VLM agreement rate; you cannot
  reliably outscore your labels' own noise floor).
- **Rank correlations** (`eval.py`): **Spearman's rho** and **Kendall's tau**
  measure agreement between two *orderings* — here the neural model's ranking vs.
  the authoritative BT ranking. Use rank metrics when order matters and raw values
  don't. Tau is roughly "of all pairs, what fraction do the two rankings order the
  same way."
- **Loss vs. metric:** the loss (BCE) is what the machine optimizes; the metric
  (accuracy, tau) is what humans decide with. They usually correlate but are not the
  same thing, and checkpoint selection here is on accuracy, not loss.

### 8.2 Baselines and sanity checks

`validate.py` computes correlation between BT rankings and the legacy Labs
`overall_score` — with an explicit expectation of **moderate** correlation. Too low
would mean something is broken; too high would mean the expensive new pipeline just
reproduced the old formula and added no value. Bracketing the *expected* value of a
sanity metric before looking at it is an underrated discipline.

Also note the hard rule that `overall_score` is **never a training label** — it's
the legacy system being replaced. Training a new model on the old model's outputs
just launders the old model's flaws into the new one. (The same issue appears in
industry as training on another model's outputs generally.)

### 8.3 Calibration and pre-registration

BT produces a relative strength (theta); users want a "7.3/10". `calibrate.py` maps
percentile rank onto a /10 curve through fixed anchor points (median = 5.0, top 10%
= 7.0, top 1% = 8.0). Two governance lessons hide in this small file:

1. The anchors are **pre-registered** — chosen and written down *before* seeing
   results, with a comment saying "do not tweak post-hoc." If you may adjust the
   scoring curve after seeing the scores, you can manufacture any result. The same
   logic applies to success metrics for any AI project: fix them before the
   experiment.
2. When a change *was* later justified (raising the top anchor because the cohort
   contains top-model faces), it was added as an **explicitly labeled post-hoc
   variant** (`ANCHORS_CAP95`) with a dated rationale, leaving the pre-registered
   curve canonical. That's how you make an exception without destroying the norm.

Relatedly: "calibration" in the probabilistic sense means that when a model says 80%
confidence, it's right about 80% of the time. Deep networks are famously
*overconfident* by default; never treat raw model confidence as a probability
without checking.

### 8.4 Acceptance gates

Before BT results are trusted, three pre-registered gates must pass (`bt.py`):

- The comparison **graph is connected** — if two groups of faces were never compared
  (even indirectly), their scores are on incomparable scales. (Here the male and
  female graphs are fit separately for exactly this reason: matchups are same-gender
  only, so the combined graph is disconnected *by construction*.)
- **No face has fewer than 15 comparisons** — small-sample scores are noise.
- **Stability**: refit on two independent 80% subsamples of the data; the two
  rankings must agree (Spearman rho > 0.95). If your conclusions change when you
  drop 20% of the data at random, you don't have conclusions.

Generalizable pattern: *quantitative, pre-registered go/no-go criteria for every
stage*. The research log applies it to the VLM pilot too (≥75% agreement at
≤$0.004/call → proceed). This is the single most transferable practice in the repo
for an executive: demand the gate be written down before the experiment runs.

---

## 9. Experiment discipline: how ML teams avoid fooling themselves

ML progress is empirical — you can't reason your way to the best config, you must
run experiments. That makes ML orgs *research orgs*, and research needs hygiene:

- **Configs as versioned files** (`configs/train-v1/v2/v3.yaml`), each with a header
  comment stating *what changed vs. the previous version and why* ("v1 overfit after
  epoch 7"). One deliberate change per run, or you can't attribute the result.
- **Seeds and reproducibility.** `split_seed: 42` makes the train/val split
  deterministic, so v1, v2, and v3 accuracies are comparable — they're graded on the
  same held-out faces. `eval.py` goes further: it rebuilds the *exact* split from
  the config stored inside the checkpoint. Numbers computed on different splits are
  not comparable, full stop.
- **Artifacts per run**: metrics to `artifacts/<run>/metrics.json`, weights to
  `checkpoints/<run>/best.pt`, config embedded in the checkpoint. Any number in a
  slide deck should be traceable to a run directory.
- **Smoke tests**: `max_pairs` lets a run execute end-to-end on a tiny subsample in
  minutes before committing hours of GPU time. Cheap-fast-wrong before
  expensive-slow-right.
- **A research log** recording decisions, gates, and results — including *rejected*
  approaches with reasons (the panel-of-VLMs rejection, the LambdaRank rejection).
  Six months later, "why didn't we do X?" has an answer.
- **Skills/runbooks** (`.cursor/skills/bt-refit/`, `.cursor/skills/preference-training/`)
  so the two core jobs are executed the same way every time, by humans or agents.

**CAIO takeaway:** you can audit an ML team's health without understanding the math.
Ask to see: the config diff between the last two runs, the metrics file for the
number in the deck, and the log entry for the last rejected idea. Healthy teams
produce all three in five minutes.

---

## 10. Adversarial thinking: poisoning, robustness, bias

The research log mentions a **synthetic poison face** — a deliberately planted bad
node whose matchups were excluded at export, with `fit_bt()` handling the resulting
under-connected stragglers by iteratively dropping them. Planting a known-bad input
to verify your defenses catch it is the ML equivalent of a fire drill.

The general categories a CAIO should keep on a checklist:

- **Data poisoning** — corrupted or adversarial training data shaping model behavior
  (grows in importance the more you train on scraped or user-generated data).
- **Spurious correlation / shortcut learning** — §5's Goodhart problem; probe for it
  with sliced evaluation (per-demographic, per-source, A-wins vs B-wins accuracy —
  the research log's order-symmetry check is exactly this).
- **Bias and fairness** — a model trained on preference labels *inherits the
  preferences of the labelers*. Here the labeler is mostly one VLM, which itself has
  training-data biases. For a face-attractiveness product this is a first-order
  ethical and legal surface (biometrics, demographic disparity), not a footnote.
  The mitigations are the same machinery as everything else: sliced metrics, human
  audits with stratified samples, and documented label rules.
- **Privacy** — embeddings and checkpoints can leak training data; hence "no photos
  or checkpoints in git" as a hard rule.

---

## 11. The LLM era, briefly, and how it maps to what you just read

This repo is small-model computer vision, but every concept transfers:

- **Foundation models** = giant pretrained backbones (DINOv2 is literally one).
  The build/buy question is the linear-probe question (§4.4) at company scale.
- **Fine-tuning an LLM** = §4.4's regimes. LoRA and friends are "train a small head /
  small delta, freeze the backbone" — the v3 probe idea.
- **RLHF / preference tuning** = this repo's exact mechanic: pairwise preference
  data, a Bradley-Terry-style reward model, sigmoid-of-score-difference loss. If you
  understand `model.py`, you understand how reward models for LLM alignment work.
- **LLM-as-judge evaluation** = §3.2's VLM labeling, with the same mandatory
  discipline: audit against humans on a stratified sample, know the agreement rate,
  and never let the judge's systematic biases go unmeasured.
- **Embeddings/RAG** = §4.3's embeddings, applied to text: retrieve relevant
  documents by vector similarity, stuff them into the prompt.
- **Inference vs. training** — training happens once and is expensive; **inference**
  (running the trained model, like `score_all_faces()` in `eval.py`) happens per
  request forever. For LLM products, inference cost usually dominates. The pilot's
  cost gate (accuracy per dollar, not accuracy alone) is the right template.
- **Hallucination** is what §8.3's calibration problem looks like in text: fluent
  confidence uncorrelated with correctness. Same disease, same cure — external
  evaluation, never self-reported confidence.

---

## 12. Glossary (one-liners)

| Term | Meaning |
|---|---|
| **Parameter / weight** | A learned number inside the model; models have millions–billions |
| **Hyperparameter** | A setting chosen by humans (lr, batch size, epochs) — see `TrainConfig` |
| **Label** | The correct answer attached to a training example |
| **Loss function** | The number training minimizes; encodes the objective |
| **Logit** | Raw unbounded model output, pre-probability |
| **Sigmoid / softmax** | Squash logits into probabilities (binary / multi-class) |
| **Gradient descent / backprop** | The nudge-all-weights-downhill algorithm and its gradient computation |
| **Optimizer (AdamW)** | The specific rule for applying gradient nudges |
| **Epoch / batch** | One full pass over training data / one chunk processed per step |
| **Backbone / head** | Generic feature extractor / small task-specific top layer |
| **Embedding** | Learned vector representation where similarity = closeness |
| **CNN / ViT / Transformer** | Convolutional net; Vision Transformer; the attention architecture behind LLMs |
| **Siamese network** | Same weights applied to two inputs, compared |
| **Transfer learning / fine-tuning / linear probe** | Reusing pretrained weights: adapt all / adapt all from pretrained / train only the head |
| **Self-supervised learning** | Pretraining on unlabeled data via invented tasks (DINOv2) |
| **Regularization (weight decay, dropout, augmentation)** | Pressure toward solutions that generalize |
| **Overfitting / underfitting** | Memorizing training data / failing to learn the pattern |
| **Train/val/test split** | Data partitions for learning / model selection / final honest measurement |
| **Leakage** | Information flow from evaluation data into training; fakes your metrics |
| **Distribution shift** | Live data drifting away from training data; silent decay |
| **Checkpoint** | Saved model weights at a point in training (`best.pt`) |
| **Inference** | Running a trained model on new inputs |
| **Calibration** | Whether stated confidence matches empirical accuracy |
| **Baseline** | The simple alternative any model must beat to justify itself |
| **Ablation** | Removing one component to measure its contribution |
| **MLE** | Maximum likelihood estimation — pick parameters making observed data most probable |
| **Bradley-Terry / Elo** | Statistical models turning pairwise outcomes into strength scores |
| **Spearman rho / Kendall tau** | Rank-order agreement metrics between two rankings |
| **RankNet / learning to rank** | Training on pair preferences via sigmoid of score difference |
| **RLHF / reward model** | LLM alignment via a preference model trained exactly like this repo's comparator |
| **LLM/VLM-as-judge** | Using a model to label or grade data; must be audited against humans |
| **Ensemble** | Combining several models' outputs; accuracy up, cost multiplied |
| **Data poisoning** | Adversarial or corrupt training data steering model behavior |
| **Goodhart / shortcut learning** | The model exploits any loophole in the objective or data |

---

## 13. The ten questions to carry into any ML review

1. What exactly is the label, who produced it, and what's the audited agreement rate
   with humans?
2. What's the unit of the train/val split, and can the same entity leak across it?
3. What's the chance baseline, the simple baseline, and the label-noise ceiling for
   the headline metric?
4. Were the success criteria and metric definitions fixed *before* the experiment?
5. Which config changed between this run and the last, and where's the metrics file?
6. What does the metric look like sliced by segment (demographic, source, time)?
7. How will we know when the live distribution has drifted from training?
8. What would this model exploit if it could cheat? Has anyone checked?
9. What does inference cost per request, and does the quality lift justify it over
   the cheaper alternative?
10. Where do the data and checkpoints live, and who can reproduce this result from
    scratch?

If a team has crisp answers to all ten, the math is almost certainly fine.
