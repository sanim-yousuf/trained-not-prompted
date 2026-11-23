# Data

No email text is redistributed in this repository. The notebook reads each source from where its publishers host it.

## Main corpus

[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) (Al-Subaiey et al., 2024), which merges six public collections:

| Source | Legitimate | Malicious |
|---|---:|---:|
| CEAS-08 | 17,061 | 15,662 |
| Enron | 14,383 | 13,893 |
| SpamAssassin | 4,075 | 1,585 |
| Nigerian-Fraud | 0 | 3,229 |
| Ling | 2,395 | 455 |
| Nazario | 0 | 1,522 |
| **Total after de-duplication** | **37,914** | **36,346** |

The positive class mixes phishing, advance-fee fraud and spam; the paper calls it *malicious* and reports phishing and fraud (Nazario, Nigerian-Fraud) separately. Exact and near duplicates are removed (82,486 to 74,260 emails), and the data are split 70/10/20, stratified by source and label. Nazario and Nigerian-Fraud emails carry source-identifying tokens (for example the collector's mailbox and 2004/2005 dates), which a detector can exploit; see the paper, §4.2.

## Modern replication

PhishFuzzer (Toth et al., 2025), CC BY-NC 4.0, used non-commercially. The notebook uses 3,000 LLM-rephrased emails after removing 1,340 near-duplicates of training emails, and the 1,199 human-written emails from 2022 onward for the analysis in §4.2.

## Identifiers in the released files

`id` in `results/scores_*.parquet` and `audit/*_ratings.csv` is the email's position in the de-duplicated corpus as the notebook builds it (`e000000` onward), which is deterministic for a given version of the Kaggle dataset; manipulated emails carry their own `id` and the `base_id` of the clean email they derive from. Running the notebook's data stage on the same dataset version recovers the text for any identifier locally.

## Score files

Each `results/scores_<set>.parquet` has one row per email with `id`, `base_id` (the clean email a manipulated one derives from), `label`, `source`, `valid` (the manipulation passed its quality filters), and one `s_<detector>` column per detector: a malicious-class score in [0, 1] for most detectors, or the normalized label-word probability for prompted ones.
