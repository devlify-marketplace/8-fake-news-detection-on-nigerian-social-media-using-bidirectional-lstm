# Data Provenance and Reuse Notes

## Current status

**No labelled corpus or trained model is bundled.** This is deliberate: a misleading/credible classifier needs labels that represent claim veracity, and the candidate dataset available during implementation did not meet that requirement. A correct, permission-cleared corpus is a prerequisite for empirical results; this project does not substitute invented examples or foreign news data.

## Nigerian corpus examined

Otor, Akumba, and Idikwu (2023), “A Novel Fake-News Dataset and Detection System to Mitigate Cyber War with Emphasis on Nigerian News Events,” describes Twitter and Facebook data around EndSARS and herdsmen-related events. The paper reports approximately 750,710 tweets and 1,169 Facebook posts and states that the full dataset is available from the authors on request, with a partial file linked in the paper. The paper's methods describe selecting an `author-verified`/verification-status field as the prediction class, and also describe modifying some Facebook write-ups to generate fake examples. An author/account verification status does not establish whether a post is factually true.

The publicly linked repository [sammy-stack/Research-Repo](https://github.com/sammy-stack/Research-Repo) has a partial CSV but no explicit license in its repository metadata. A temporary inspection of that CSV found 750,581 parsed rows with columns `text,label`: 650,582 rows had blank labels; 99,698 had label `0`; 301 had label `1`. Repeated/overlapping text was substantial, and the target-label semantics do not provide independently verified claim truth. The CSV is **not included, used for training, or treated as a benchmark**. Do not infer veracity from these labels. Seek written clarification/permission and a corrected annotation scheme from the original data authors if considering it for future work.

A separate public GitHub project [`smksean/nigeria-fake-news-detection`](https://github.com/smksean/nigeria-fake-news-detection) explicitly describes its bundled dataset as synthetic; it was not used as real-world evidence.

## Public fact-check archives

FactCheckHub and DUBAWA articles can document Nigerian social-media claims, their platform/source links, and verdicts. Examples inspected include:

- [FactCheckHub: fabricated quote attributed to Peter Obi](https://factcheckhub.com/did-peter-obi-say-his-son-is-gay-no-quote-fabricated/) — page identifies a Facebook-group post and gives a false verdict.
- [FactCheckHub: EU-ACP/Samoa Agreement claim](https://factcheckhub.com/how-true-is-the-claim-that-nigeria-has-not-signed-the-eu-acp-agreement/) — page quotes an X post and assesses its specific claim as true.
- [DUBAWA: Sokoto mass-casualty claim](https://dubawa.org/false-terrorists-did-not-kill-100-in-sokoto-as-facebook-user-claimed/) — page cites a Facebook post and a false verdict.

These pages demonstrate that source-linked claim-level material exists, but do not constitute a complete or balanced research corpus by themselves. Quoted third-party post text may have separate rights; the FactCheckHub page expressly notes it has not obtained rights from third-party content. This code bundle does not republish those claims. If a supervisor approves a fact-check-derived proxy corpus, document its limited sampling frame, licensing/reuse position, verbatim/paraphrase status, annotation mapping, and platform/topic bias. Such a corpus must not be described as a representative sample of all Nigerian social media.

## Rights and collection

The project's software is not permission to collect or republish platform content. Before obtaining data, check the current platform terms, source-page reuse terms, institutional ethics review, data-protection requirements, and authors'/platform permissions. Follow [`DATA_COLLECTION_PROTOCOL.md`](DATA_COLLECTION_PROTOCOL.md). Keep source URLs and annotations in the working dataset as required, but avoid personal data and never place raw posts into public source control. The paper's open-access article is licensed CC BY 4.0; that license does not automatically grant rights to user posts reproduced in the paper or its external repository.

## References

1. Otor, S. U., Akumba, B. O., & Idikwu, J. S. (2023). *A Novel Fake-News Dataset and Detection System to Mitigate Cyber War with Emphasis on Nigerian News Events*. International Journal of Scientific Research in Computer Science, Engineering and Information Technology, 9(4), 24–32. [Article](https://ijsrcseit.com/CSEIT23903146), [PDF](https://ijsrcseit.com/paper/CSEIT23903146.pdf), DOI: [10.32628/CSEIT23903146](https://doi.org/10.32628/CSEIT23903146).
2. [Partial corpus repository](https://github.com/sammy-stack/Research-Repo) (accessed for schema/quality inspection; no explicit license found).
3. [FactCheckHub](https://factcheckhub.com/) and [DUBAWA](https://dubawa.org/) public fact-check archives (source examples only; not redistributed here).
