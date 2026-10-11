# Raw data goes here — not included

Place a permission-cleared UTF-8 CSV here (or pass another path to the CLI). Required columns:

- `text`: the public post text, or a short claim text copied with documented research permission.
- `label`: `credible` / `0` or `misleading` / `1`, based on a cited fact-check or a documented annotation protocol. A platform account verification badge is **not** a truth label.

Recommended provenance/grouping columns: `platform`, `source_url`, `post_id` (hashed or removed before analysis), `timestamp`, `claim_id` or `event_id`, and `annotator_id` (pseudonymous). Avoid names, handles, phone numbers, private-group content, direct messages, or unnecessary user metadata. Review platform terms and your institution's ethics requirements before collecting or retaining posts.

No dataset is bundled because the publicly linked Nigerian corpus assessed for this project has missing labels, severe duplicate concentration, a verification-status label rather than a fact-veracity label, and no explicit reuse license. Do not treat it as a credible/misleading benchmark. See `docs/DATA_PROVENANCE.md` and `docs/DATA_COLLECTION_PROTOCOL.md`.
