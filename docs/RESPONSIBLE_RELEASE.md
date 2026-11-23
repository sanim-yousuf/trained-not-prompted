# Responsible release

This work studies how email filters built on language models can be evaded, and it is dual-use. The attack types it uses are known in principle; the contribution is a measurement protocol and an analysis of defenses, including where they fail.

## Released

- All code, including the attack implementations and the injection templates, so that the measurements can be checked and extended.
- Per-email detector scores for every evaluation set, all aggregate results, figures and tables.
- Validity-audit ratings with identifiers only.

## Withheld

- Generated adversarial emails from the adaptive attacks.
- Paraphrases, LLM-written padding and LLM-written instruction paragraphs.
- Audit sheets that contain email text.

These are ready-to-use evasion material against deployed filters, and releasing them would add little to reproducibility beyond the code and scores. The notebook writes them to a separate `cbs/private/` folder that is never included in its public output.

## Access on request

Researchers who need the withheld material, for example to evaluate a defense on the same adversarial emails, can request it after the review period through the contact given in the published paper. Please describe the intended use; material is shared for non-commercial research under the source datasets' terms.

## Source data

All source corpora are public. Some contain real correspondence released publicly (for example Enron); the paper and this repository report only aggregates and quote no email.
