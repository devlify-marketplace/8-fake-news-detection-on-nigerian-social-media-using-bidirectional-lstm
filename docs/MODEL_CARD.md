# Model Card — Nigerian Social-Media BiLSTM

## Status

**Implementation complete; model not trained.** There is no checkpoint, vocabulary, or empirical performance result in this delivery because no approved dataset with defensible veracity labels was available. The API reports this state and refuses prediction until training artifacts exist.

## Intended use

Educational and research exploration of binary short-text classification for publicly shared Nigerian social-media claims. Intended only as a screening signal to prioritize human verification after suitable data collection, adjudicated labels, leakage-controlled evaluation, and supervisor review.

## Out-of-scope use

- Establishing the truth or falsity of a claim without evidence.
- Automated moderation, account ranking, censorship, political targeting, surveillance, employment, credit, health, legal, or other high-impact decisions.
- Identifying a person, inferring intent, or treating account verification, writing style, identity, language, location, or political position as truth.

## Design

Word-level vocabulary built on training text only; token embeddings feed a one-layer, bidirectional LSTM; the concatenated final forward/backward states are passed to a dropout and binary-logit head. Training uses class-weighted binary cross-entropy, validation-loss early stopping, and a TF-IDF/logistic-regression comparison on identical grouped data splits. The model output is an **uncalibrated score**, not a probability of truth.

## Data and evaluation

No training data are included. The project's data intake requires `text` and verified `label`, with preferred `claim_id`/`event_id` for group separation. Account-verification status is explicitly rejected as an unsupported label. The candidate public partial corpus is discussed in `DATA_PROVENANCE.md` and excluded due to missing labels, label validity, duplication, and unclear reuse terms.

## Risks and limitations

- Severe domain, platform, temporal, dialect, code-switching, and topic shift can cause failure.
- Label definitions and fact-check coverage can encode institutional, political, or language bias.
- Surface cues (hashtags, named entities, source wording) may be learned instead of factual reasoning.
- A binary label oversimplifies satire, opinion, uncertainty, partial truth, and evolving claims.
- Small datasets and repeated narratives inflate apparent test performance unless grouped carefully.
- Human annotator disagreement and source-verdict errors remain possible.

## Required validation before research claims

Document source rights, collection method, time period, class counts, excluded rows, label mapping, annotator agreement, duplicate removal, group split, confusion matrix, per-class precision/recall/F1, macro F1, balanced accuracy, MCC, ROC-AUC, baseline comparison, error analysis, and limitations. Prefer temporal and external-source/platform validation. Calibrate scores and assess subgroup performance before any public-facing use. Always require human source-based fact-checking.
