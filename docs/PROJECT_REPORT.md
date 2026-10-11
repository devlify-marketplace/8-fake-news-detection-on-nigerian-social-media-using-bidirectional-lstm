# Fake News Detection on Nigerian Social Media Using Bidirectional LSTM

**Final-year project report — Topic 8**  
Department of Computer Science, Federal University Gusau  

## Abstract

The rapid circulation of unverified claims on Nigerian social-media platforms creates a need for tools that help researchers and fact-checkers prioritize content for verification. This project implements a bidirectional long short-term memory (BiLSTM) text-classification prototype for short posts labelled as credible or misleading. The system includes a data-validation workflow, claim/event-aware grouped splitting, a train-only word vocabulary, class-weighted model training, a TF-IDF/logistic-regression baseline, a local inference API and a browser interface. It also defines a data-collection and human-annotation protocol that separates claim veracity from account identity and source popularity. An audit of the Nigerian corpus linked by a relevant 2023 study found substantial missing target values, repeated material, a target described as author-verification status rather than independently established claim truth, and no explicit license in the repository metadata; the corpus is therefore not used as training evidence. No empirical model score is claimed in this report. Training and measured results remain pending a permission-cleared, veracity-labelled Nigerian social-media corpus. The software reports an untrained state rather than invent predictions. The intended role is research screening followed by human fact-checking, not automated truth determination.

**Keywords:** misinformation, fake news, Nigerian social media, bidirectional LSTM, text classification, fact-checking, data provenance

## 1. Introduction

Social platforms allow Nigerian users to publish and redistribute news, claims, and opinions rapidly. The same speed can propagate fabricated quotations, manipulated context, outdated media, and inaccurate claims about public safety, elections, health, and public policy. A human fact-checker must locate a claim, establish its meaning and context, identify evidence, and communicate a verdict. A text classifier can assist by ranking a stream of items for review, but language patterns alone cannot establish whether a factual statement is true.

The selected topic proposes a classifier trained on Nigerian social-media text and a bidirectional LSTM architecture. This implementation makes that topic executable while treating training evidence as a first-class requirement. It deliberately distinguishes **the software prototype** from **the empirical study**: runnable model and application code are provided, but verified performance is not reported until appropriate labelled data are available.

## 2. Problem statement

A misleading post may be short, informal, multilingual, code-switched, and detached from the evidence needed to assess it. Generic fake-news corpora are often based on news articles from other countries and may not represent Nigerian platforms, topics, language varieties, or posting conventions. Labels may also measure source reputation or account verification rather than truth. Training on such a proxy can produce a confident but invalid model and overstate the system's ability to distinguish credible posts from misinformation.

The problem addressed is therefore twofold: implement a BiLSTM classifier and deployment prototype, and establish a defensible data path with veracity labels, provenance, privacy safeguards, and leakage-controlled evaluation.

## 3. Aim and objectives

**Aim:** To implement and prepare for evaluation a Nigerian social-media text classifier based on a bidirectional LSTM that screens posts for potentially misleading content.

**Objectives:**

1. Define a reproducible input schema and validation rules for post text, veracity label, source, and claim/event grouping.
2. Implement text normalization and a vocabulary learned only from the training partition.
3. Build a word-embedding BiLSTM binary classifier with class-weighted loss, early stopping, and deterministic configuration.
4. Compare it with a TF-IDF/logistic-regression baseline on the same grouped test partition.
5. Report class-sensitive metrics and provide a local inference API and browser demo that do not retain user input.
6. Document dataset provenance, ethics, privacy, label limitations, model card, and human-review requirements.

## 4. Research questions

- Can the implemented BiLSTM learn useful text patterns for separating adjudicated credible and misleading Nigerian social-media claims on a leakage-controlled holdout?
- Does the BiLSTM outperform a TF-IDF/logistic-regression baseline on the same test examples?
- How do class imbalance, claim/topic grouping, platform, language, and annotation quality affect measured performance and errors?

These are empirical questions, not findings. They cannot be answered until an appropriate corpus is obtained and evaluated.

## 5. Scope and assumptions

The implemented classifier accepts English and Unicode text and normalizes URLs and user mentions while retaining hashtags and local-language tokens. It is a binary classifier whose class 0 is credible and class 1 is misleading. Multi-level fact-check verdicts, opinions, satire, unresolved claims, and ambiguous assertions require a pre-registered mapping or exclusion rule. The model uses post text only; source metadata is retained for provenance and grouping but is excluded from prediction features.

The project does not scrape private social groups, access direct messages, bypass access controls, identify authors, automatically fact-check evidence, detect image/video manipulation, or perform multilingual fairness validation. A future study may extend it to Hausa, Yoruba, Igbo, Nigerian Pidgin, stance, evidence retrieval, temporal generalization, and external platform validation.

## 6. Related work and data motivation

Otor, Akumba, and Idikwu (2023) describe extracting Nigerian EndSARS-related posts from Twitter and Facebook and report RNN/CNN experiments. The paper is a useful indication that relevant local data can be collected, but the accessible partial repository does not constitute a verified binary truth dataset. The published methods describe an author-verification field as the prediction target and mention modifying some Facebook text to create fake examples. Verification status is not truth, so the accessible file is excluded. A separate repository found during research explicitly calls its Nigerian news examples synthetic and is also excluded.

Public fact-checking organizations such as FactCheckHub and DUBAWA publish Nigeria-related claims, source links, and verdicts. These may support a future claim-level research corpus when reuse rights, sampling, wording, and label mapping are resolved. Such data would be a fact-check-derived proxy, not automatically a representative sample of social-media content. Full provenance and source limitations are recorded in `DATA_PROVENANCE.md`.

## 7. Methodology

### 7.1 Data collection and labelling

The intended corpus consists of permitted public posts or self-contained claims originating from Nigerian social platforms. Each item should have a veracity label based on cited evidence, not a user's identity or verification badge. The protocol recommends independent double annotation, adjudication, recording of original verdicts and evidence, and exclusion of opinions or unresolved assertions. Source URLs, platform, date, collection method, claim/event group, and text-verbatim status should be retained where permitted. Personal identifiers are minimized or removed.

### 7.2 Preprocessing and leakage control

Text is HTML-decoded and case-folded; URLs and account mentions are replaced with generic tokens; whitespace is normalized; tokenization retains Unicode word characters, hashtags, and punctuation. Exact normalized duplicates are removed. A human-assigned `claim_id`, `event_id`, or `group_id` keeps related paraphrases together. If none exists, exact normalized text is the fallback grouping key. A deterministic grouped-stratified split allocates approximately 60%/20%/20% for training, validation, and test. The vocabulary is fit on training text only. This reduces exact claim leakage but does not by itself guarantee independence across similar narratives; stronger future work should use time-based and external-source/platform tests.

### 7.3 Model architecture

Each token is mapped to a trainable 100-dimensional embedding. The sequence is processed by one 64-unit-per-direction BiLSTM. The final forward and backward states are concatenated, dropout is applied, and a linear head produces one binary logit. Training uses weighted binary cross-entropy, AdamW, gradient clipping, deterministic seeds, and validation-loss early stopping. Sequence length and other values can be adjusted in `configs/default.json`.

### 7.4 Baseline and evaluation

TF-IDF unigrams/bigrams with logistic regression and balanced class weights are fitted on the same training partition and evaluated on the same held-out test partition. Planned measures are accuracy, balanced accuracy, misleading-class precision/recall/F1, macro F1, Matthews correlation coefficient, ROC-AUC, and a confusion matrix. Accuracy is not sufficient for an imbalanced corpus. Error analysis should group mistakes by topic, platform, language, and claim type, and investigate high-confidence false positives/negatives. Confidence intervals and inter-annotator agreement should be reported when supported by the data.

### 7.5 Application

The FastAPI service exposes health and prediction routes. The browser demo posts text to the local API. If no checkpoint exists, prediction returns HTTP 503 and the interface explains that training data are required. With a trained checkpoint, output is a non-calibrated model score and may abstain for human review. Requests are processed in memory and are not saved by the application.

## 8. System requirements and implementation

The package supports Python 3.10–3.12, PyTorch, pandas, scikit-learn, FastAPI, and Uvicorn. The repository contains setup metadata, configuration, source modules, tests, web assets, provenance documentation, and an architecture diagram. Run the test suite with `pytest -q`. Add an approved `data/raw/posts.csv`, validate it with `python scripts/validate_data.py data/raw/posts.csv`, then train using `python -m nigerian_fake_news.train --root . --data data/raw/posts.csv`.

## 9. Results and status

The engineering implementation is testable. **No training experiment was run and no predictive metric is reported.** The candidate local corpus is excluded because its label semantics and completeness do not support the intended target. Synthetic examples appear only as isolated unit-test fixtures to verify software behavior. They are not used to fit a model or claim research performance. When approved data are supplied, results will be generated under `artifacts/run_*`; the report must then be updated using the actual test results, data counts, annotation agreement, and error analysis. See `RESULTS.md` for the reporting checklist.

## 10. Ethical, privacy, and societal considerations

Incorrectly flagging credible posts can undermine trust; missing a harmful false claim can allow misinformation to spread. Political, ethnic, religious, and security topics can create particular risks. Labels may reflect fact-checker coverage or annotator disagreement. Dataset imbalances in platform, topic, language, or author style can cause systematic error. The model must not infer truth from an account's verification status, identity, dialect, or political position. Use only data covered by platform terms, source rights, and institutional approval; remove unnecessary personal data; restrict raw-dataset access; define retention and deletion; and avoid using outputs for surveillance or punitive decisions. Human reviewers must examine cited evidence before acting.

## 11. Limitations and future work

The main limitation is the lack of an approved training corpus, so the system has no trained checkpoint and no empirical performance claim. Even after data are supplied, an LSTM trained on a small fact-check set may learn lexical/topic artifacts, perform poorly on new events, and fail on code-switching or languages not represented in training. Future work should secure a rights-cleared Nigerian post corpus with claim-level evidence; validate label quality through multiple annotators; report time- and platform-held-out results; calibrate scores; test generalization across regions/languages; add evidence retrieval and stance detection; and conduct bias and privacy review before any broader use.

## 12. Conclusion

This project delivers the core BiLSTM classifier, data-validation and grouped-evaluation pipeline, baseline comparison, local API, browser demo, tests, and academic documentation. It does not claim to solve Nigerian misinformation detection without suitable data. The next research dependency is a permission-cleared, correctly labelled social-media corpus. Once supplied, the included commands perform training and evaluation, and results can be reported transparently without changing the stated limitations.

## References

1. Otor, S. U., Akumba, B. O., & Idikwu, J. S. (2023). A Novel Fake-News Dataset and Detection System to Mitigate Cyber War with Emphasis on Nigerian News Events. *International Journal of Scientific Research in Computer Science, Engineering and Information Technology, 9*(4), 24–32. [https://doi.org/10.32628/CSEIT23903146](https://doi.org/10.32628/CSEIT23903146).
2. [FactCheckHub](https://factcheckhub.com/), public Nigeria-focused fact-check archive.
3. [DUBAWA](https://dubawa.org/), public West African fact-checking project.
