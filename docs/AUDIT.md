# Validity audit

## Procedure

Fixed random samples were drawn from five sets: 50 naturally padded emails (all padding variants), 30 emails that evaded a detector under the score-guided padding attack, 30 under the adaptive injection attack, 50 paraphrases, and 30 legitimate emails with an ordinary-instruction paragraph (190 items).

An LLM judge (Claude Opus 5.5) rated every item against the rubric below. One author then reviewed every rating against the item and agreed with all 190. The second step is a review of the first, not an independent annotation, so the paper reports validity rates and no inter-rater agreement.

## Rubric (binary, 1 = yes)

| Set | Column | Question |
|---|---|---|
| Padding, score-guided attack, adaptive injection | `still_malicious_yn` | Is the email still malicious, with its request and payload (links, contact details, amounts) intact? |
| Paraphrase | `intent_preserved_yn` | Does the rewrite make the same request for the same purpose? |
| Paraphrase | `fluent_yn` | Is the rewrite fluent and free of generator artifacts (placeholders such as `[Your Name]`, notes about the rewrite, duplicated versions, leaked prompt text)? |
| Paraphrase | `payload_kept_yn` | Are the links, contact details and amounts of the original kept? |
| Ordinary instructions | `plausible_legitimate_yn` | Is the email a plausible legitimate email, with the added paragraph fitting it? |

## Files

`audit/*_ratings.csv` hold identifiers, the attack target or variant, automatic quality measures where available, the judge's rating (`judge_*`) and the reviewing author's rating (`author_review_*`). The email text is omitted (see RESPONSIBLE_RELEASE.md).

## Results

| Set | Criterion | Valid |
|---|---|---:|
| Natural padding | still malicious | 50/50 |
| Score-guided padding attack | still malicious | 30/30 |
| Adaptive injection attack | still malicious | 30/30 |
| Paraphrase | intent preserved | 48/50 |
| Paraphrase | fluent, no artifacts | 33/50 |
| Paraphrase | payload kept | 48/50 |
| Ordinary instructions | plausible legitimate | 23/30 |

Fixed patterns that flag the observed artifacts match the audit (16 of 17 disfluent paraphrases, 7 of 7 implausible instruction emails, no false flags) and are applied to the full sets in notebook section 14b; excluding flagged emails changes no conclusion (`results/v53_*`).
