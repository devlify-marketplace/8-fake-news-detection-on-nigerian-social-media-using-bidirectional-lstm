# Data Collection and Annotation Protocol

## Research question and unit of analysis

The unit is a short Nigerian public social-media post (or the smallest self-contained factual claim within it), paired with a documented evidence-based verdict. The target is **claim veracity**, not whether a user/account is verified, whether a post is popular, or whether the post expresses a political opinion. Keep posts and paraphrases about the same underlying claim in one `claim_id` group so they cannot cross data partitions.

## Permitted acquisition

1. Obtain institutional/supervisor approval and check current platform terms, data-protection obligations, and source licenses before collection.
2. Prefer official platform research APIs or a platform-provided export where access and permitted use are documented. Do not scrape private groups, direct messages, deleted/blocked content, or bypass access controls. Do not retain usernames, phone numbers, email addresses, profile photos, or unnecessary user IDs.
3. For public posts submitted for fact-checking, preserve only the minimum text and provenance required for the research. A fact-check report is not itself proof of the original claim's wording; check its linked/archived source when available. Record whether text is verbatim, lightly normalized, or a claim paraphrase.
4. Keep a source register with collection date, platform/source, URL or stable reference, license/permission, topic/claim group, and any transformation. Store any re-identification keys separately with restricted access; they are not model input.

## Annotation

- Use a written rubric and at least two independent annotators for each item. Resolve disagreements with an adjudicator; preserve original disagreements in a separate restricted annotation log.
- `credible` means the factual assertion is supported by the cited evidence at the time of assessment; `misleading` means demonstrably false, fabricated, materially incomplete, or contextually misrepresented. Exclude opinions, satire, unverifiable claims, and ambiguous claims from binary training rather than forcing labels.
- Keep the original fact-check verdict/category and a short evidence citation in the working annotation record. Map a multi-level verdict to the binary classes only under a documented rule approved before modeling; report exclusions and mapping decisions.
- Measure inter-annotator agreement (Cohen's kappa for two annotators or an appropriate multi-rater statistic) before adjudication. Report the sample size and agreement. Do not claim ground truth if annotation is weak.

## Dataset fields

Required: `text`, `label` (0 credible / 1 misleading). Recommended: `platform`, `source_url`, `timestamp`, `claim_id` or `event_id`, `original_verdict`, `annotation_evidence`, `language`, `text_is_verbatim`, `collection_method`, `annotator_agreement`. Use hashed/opaque IDs where needed. Keep source/provenance columns out of model features.

## Quality and leakage checks

- Remove empty rows and exact normalized duplicates; investigate conflicting labels for repeated text instead of silently choosing one.
- Group split by claim/event; do not randomly split reposts or near-identical claims across train/test. For stronger validation, use a chronological holdout and an external source/platform test set when size permits.
- Inspect class balance, language distribution, post-length distribution, source/platform mix, duplicate rate, and topic overlap. Detect labels accidentally encoded in source, URLs, hashtags, templates, or annotator notes.
- Report precision, recall and F1 for the misleading class, macro F1, balanced accuracy, MCC, ROC-AUC, confusion matrix, confidence intervals where sample size allows, and baseline comparison. Accuracy alone is insufficient.

## Privacy and harm reduction

Use only content allowed for the project; follow the Nigeria Data Protection Act and institutional ethics requirements as applicable. Minimize retention, remove personal identifiers, keep only necessary claims, restrict raw data access, and define a deletion/retention schedule. Do not publish datasets or model artifacts that expose sensitive information or repeat harmful content without a justified, reviewed research need. Predictions should not be used for surveillance, employment, political targeting, or punitive moderation.
