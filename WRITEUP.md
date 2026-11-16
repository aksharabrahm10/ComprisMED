# comprisMED — hackathon write-up

## The problem

Older adults managing instructions after an appointment may find medical terms and multi-step directions difficult to follow. comprisMED explores whether source-grounded verification can reduce omission or alteration of important details when AI simplifies those instructions. This is a research prototype, not a clinical tool.

## What we built

A Gradio app accepts fictional instructions, sends them to Groq for a plain-language draft, and checks selected amounts, frequency, duration, dates, and negation wording against the source. When literal checks fail, the app may ask the model for one repair and displays any remaining issues for human review. Users can select a translation language; translations are not independently verified. A separate experiment compares AI-only with AI plus checking and repair; it is not shown in the regular app.

## What we added

The research contribution is a paired comparison of ordinary AI simplification and AI plus source-grounded checking. The local benchmark contains 60 project-authored synthetic cases in 12 family-disjoint groups (20 train, 10 validation, 30 test). Expected facts are authored separately from the parser, with controlled alterations for checking detection. They have not been independently adjudicated; no real patient data are used. Public dataset candidates and reuse limitations are documented in README.md.

## How it works

The experiment makes one Groq draft per example. The AI-only condition uses that output; the verified condition checks the same output and, when needed, sends the source and flagged details back to the model once. A passing check means only that the implemented string/regex comparison passed. It does not establish meaning, clinical accuracy, or safety.

## Results

On 2026-11-15, the live run used the 30 held-out synthetic test cases with `openai/gpt-oss-120b`. Each example has one project-authored selected fact. The same AI draft was evaluated in both conditions; AI + verification checked it and made at most one repair call (four repairs were attempted).

| Result across selected facts | AI-only | AI + verification |
|---|---:|---:|
| Preserved | 26/30 (86.7%) | 29/30 (96.7%) |
| Flagged omissions | 1/30 (3.3%) | 1/30 (3.3%) |
| Flagged alterations | 3/30 (10.0%) | 0/30 |

The checker-only controlled mutation test is a separate deterministic test: it flagged 30/30 deliberately altered synthetic strings and 0/30 unchanged controls (precision, recall, and F1 = 1.00; false-alarm rate = 0.00). Those results are not Groq experiment results.

These model-condition rates are **literal-checker outcomes, not human-judged semantic or clinical outcomes**. The baseline was flagged for rewriting “2 weeks” as “14 days” and “2 readings” as “two measurements,” both meaning-preserving paraphrases. A date was also flagged as omitted even though it appeared in the output with different spacing/punctuation. After repair, 29/30 selected facts passed the matcher; the remaining flagged date appears in the output in a different format. These examples show that the observed increase in match rate does not prove a reduction in medically meaningful errors. The synthetic expected facts have not been independently reviewed by a clinician; no clinical validation or medical accuracy is claimed.

Full records and metrics are saved in `results/synthetic_test_outputs.json` and `results/synthetic_test_metrics.json`. The separate checker-only result is `results/synthetic_test_checker_metrics.json`. Re-run the paired experiment with `.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset synthetic --split test`; the key is read from the local `.env`. Local unit tests and browser smoke tests establish functionality only, not clinical validity.

## What we'd do next

Independently review annotations, freeze code and prompts before the held-out run, inspect missed changes and false alarms by detail type, and only then report measured results. Seek explicit reuse permission and privacy review before using real-world data. Obtain clinician and older-adult feedback, and evaluate translation separately before presenting it as checked.

## Sources

- Synthetic benchmark and annotations: project-authored, `data/synthetic/benchmark.jsonl`; generator and annotation method: `data/synthetic/build_benchmark.py`.
- Synthetic jargon starter data: project-authored, `data/jargon_training.csv` (40 rows; not externally validated).
- Med-EASi candidate reviewed but not used because dataset reuse rights are unclear: [dataset card](https://huggingface.co/datasets/cbasu/Med-EASi), [paper](https://arxiv.org/abs/2302.09155), [authors' repository](https://github.com/Chandrayee/CTRL-SIMP).
- Model/API: Groq Chat Completions API; this experiment used `openai/gpt-oss-120b`: [API docs](https://console.groq.com/docs/api-reference#chat-create), [model catalog](https://console.groq.com/docs/models).
- Libraries: Gradio, Hugging Face Hub client, NumPy, pandas, python-dotenv, requests, scikit-learn, pytest; versions/ranges are listed in `requirements.txt`.
- No tutorial source, external benchmark rows, or AI-generated dataset is currently used. Add any AI tools used by the team to prepare the submission before submitting.

**This project is a prototype and is not medical advice.**
