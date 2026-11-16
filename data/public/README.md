# Public medical simplification data

## Selected dataset: GEM Cochrane Simplification

The project includes the English-language [GEM Cochrane Simplification dataset](https://huggingface.co/datasets/GEM/cochrane-simplification), downloaded from the pinned revision `75a92ae445171fa1b7641a229bfe3c77c0d8723d`. Its dataset card declares **CC BY 4.0**. Attribute the original authors and dataset, retain the license notice, and review the current card and source terms again before redistribution or submission.

The corpus has 4,459 source/target pairs: 3,568 train, 411 validation, and 480 test. Sources are excerpts from Cochrane review abstracts; targets are roughly aligned excerpts from their plain-language summaries. Each record contains `gem_id`, `doi`, `source`, and `target`. The downloader checks the pinned SHA-256 digests, required field types, unique DOI grouping within each supplied split, and DOI separation across the supplied splits.

| Included file | Rows | Bytes |
|---|---:|---:|
| `data/public/cochrane/train.json` | 3,568 | 13,615,698 |
| `data/public/cochrane/validation.json` | 411 | 1,538,508 |
| `data/public/cochrane/test.json` | 480 | 1,830,276 |

This is **secondary text-simplification data, not a post-telehealth instruction dataset**. It can help probe whether a medical research paragraph's selected numbers and other explicitly annotated details survive a plain-language rewrite. It cannot establish preservation of medication instructions, clinical meaning, or safety. The reference targets are only roughly aligned; they may omit details or include information absent from the source. They are therefore not treated as fact-preservation gold labels. The project does not train or fine-tune on these rows.

### Attribution

Devaraj, Ashwin, Iain J. Marshall, Byron C. Wallace, and Junyi Jessy Li. 2021. “Paragraph-level Simplification of Medical Texts.” *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 4972–4984. https://doi.org/10.18653/v1/2021.naacl-main.395.

Dataset card and files: [GEM/cochrane-simplification](https://huggingface.co/datasets/GEM/cochrane-simplification). Dataset source repository: [Paragraph-level Simplification of Medical Texts](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts). License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

## Creating reviewed test labels

The source-only annotation workflow uses a deterministic sample of 30 DOI-disjoint examples from the official test split. It does not show annotators the target summary or use the project's fact extractor to create expected details.

Blank source-only sheets for the fixed 30-example sample are already prepared as `data/public/cochrane/reviewer_a.csv` and `reviewer_b.csv`. If the sheets are missing, generate them from the project root in VS Code:

```powershell
.\.venv\Scripts\python.exe data\public\download_cochrane.py
.\.venv\Scripts\python.exe data\public\cochrane\prepare_annotations.py
```

The generator refuses to replace existing sheets. Use `--force` only if you intentionally want to discard their contents and recreate them. Give `reviewer_a.csv` and `reviewer_b.csv` to two reviewers independently. For each row, each reviewer should:

1. Select one or more details from the source that a plain-language rewrite should preserve.
2. Enter exact source snippets grouped by supported category in `facts_json`, for example `{"number_or_dose":["0.73"]}`.
3. Set `complete` to `yes` only after reviewing the entire source. Use `reviewer_notes` to record uncertainty.

Supported categories are `number_or_dose`, `frequency`, `duration`, `date`, and `negation`. The validator checks that each selected detail is a literal span in the source. This workflow is for a small research benchmark, not clinical annotation; have a qualified clinician review labels if making any clinical claims. The current project must not make such claims.

After both reviews are complete, validate and build the gold file:

```powershell
.\.venv\Scripts\python.exe data\public\cochrane\build_adjudicated_annotations.py
```

If reviewers disagree, the script creates `adjudication.csv` and stops. Discuss each disagreement and enter the agreed final fact JSON, set `adjudication_complete` to `yes`, and rerun the script. It writes `adjudicated_annotations.jsonl` only when all 30 cases have two completed reviews and all disagreements are resolved. It never silently treats missing or empty annotations as valid gold labels.

## Running experiments

The held-out synthetic checker test requires no API key and does not fabricate a model result:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset synthetic --split test --checker-only
```

The latest local run flagged all 30 deliberately altered test cases and did not flag any of the 30 unchanged-source controls (precision/recall/F1: 1.00; false-alarm rate: 0.00 on those unchanged-source controls). The saved result is [synthetic_test_checker_metrics.json](../../results/synthetic_test_checker_metrics.json). This measures literal checks on one selected fact per synthetic scenario; unchanged-source controls do not estimate false alarms on paraphrased AI outputs or real instructions.

With `GROQ_API_KEY` set in the local `.env`, compare AI-only simplification to the same AI output plus deterministic checking and at most one repair:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset synthetic --split test
```

To run the secondary Cochrane test after the annotation file exists:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset cochrane --split test
```

The ordinary AI condition receives no gold-fact hints. The verified condition begins from that same draft, compares it with the independently authored required facts, and calls Groq for one repair only if a check fails. Both initial and final checks are saved. The experiment writes outputs, aggregate and per-category preservation/omission/alteration rates, and up to three actual successes and failures under `results/`. No live-model results exist until these commands complete successfully. API calls may incur cost.

The synthetic controlled-mutation false-alarm rate measures only the deterministic checker against clean synthetic source text and deliberately altered versions. It is not a false-alarm estimate on real instructions, on model outputs, or on patients. Public-data preservation metrics are limited to the small set of reviewer-selected literal details; they are not clinical outcomes.

## Excluded candidate: Med-EASi

[Med-EASi](https://huggingface.co/datasets/cbasu/Med-EASi) is task-relevant and has paired medical simplification examples, but its dataset card does not declare a reuse license and the upstream repository does not provide a dataset license. The code license does not establish rights for the data. It is not downloaded, bundled, sent to an API, or used in the experiments unless the data authors clarify permission.

No private or patient-level data is used. The included post-telehealth instruction benchmark is project-authored synthetic text and is kept separate from this public research-summary dataset.
