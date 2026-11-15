# comprisMED — hackathon write-up

## The problem

Older adults managing instructions after an appointment may find medical terms and multi-step directions difficult to follow. comprisMED explores whether source-grounded verification can reduce omission or alteration of important details when AI simplifies those instructions. This is a research prototype, not a clinical tool.

## What we built

A Gradio app accepts fictional instructions, sends them to Groq for a plain-language draft, and compares the draft with the source for selected amounts, frequency, duration, dates, and negation wording. It shows an independently generated AI-only draft beside an AI + verification result. When literal checks fail, the verified path asks the model for one repair and presents remaining issues for human review. Users can select a translation language; translations are not independently verified.

## What we added

The research contribution is a paired comparison of ordinary AI simplification and AI plus source-grounded checking. The local benchmark contains 60 project-authored synthetic cases in 12 family-disjoint groups (20 train, 10 validation, 30 test). Expected facts are authored separately from the parser, with controlled alterations for checking detection. They have not been independently adjudicated; no real patient data are used. Public dataset candidates and reuse limitations are documented in README.md.

## How it works

The Groq model creates both drafts. A deterministic checker extracts a limited set of literal details from the source and compares them against each draft. If a checked detail is missing or appears altered, the verified path sends the source and flagged details back to the model once. A passing check means only that the implemented string/regex comparison passed. It does not establish meaning, clinical accuracy, or safety.

## Results

On 2026-11-15, a live Groq smoke test with fictional text successfully returned a simplified response. A live app-callback check produced both an AI-only draft and an AI + detail-check draft for fictional instructions containing a 5 mg dose, twice-daily frequency, 7-day duration, and “do not stop early”; the implemented literal checks marked those selected details as preserved. This is one qualitative functional check, not a held-out evaluation, and it does not demonstrate that verification improves preservation.

No live-model test-set metrics have been run and inspected yet. Do not claim a benefit until the held-out experiment is run after development is frozen. Run `python experiments\\run_experiment.py` with `GROQ_API_KEY` configured; results are written to `results/test_outputs.json` and `results/test_metrics.json`. The output includes overall and per-detail-type omission/alteration and mutation detection metrics. Record actual examples, including a failure, here after inspecting those files. Local unit tests and the single smoke test are software/functionality checks, not experiment results.

## What we'd do next

Independently review annotations, freeze code and prompts before the held-out run, inspect missed changes and false alarms by detail type, and only then report measured results. Seek explicit reuse permission and privacy review before using real-world data. Obtain clinician and older-adult feedback, and evaluate translation separately before presenting it as checked.

## Sources

- Synthetic benchmark and annotations: project-authored, `data/synthetic/benchmark.jsonl`; generator and annotation method: `data/synthetic/build_benchmark.py`.
- Synthetic jargon starter data: project-authored, `data/jargon_training.csv` (40 rows; not externally validated).
- Med-EASi candidate reviewed but not used because dataset reuse rights are unclear: [dataset card](https://huggingface.co/datasets/cbasu/Med-EASi), [paper](https://arxiv.org/abs/2302.09155), [authors' repository](https://github.com/Chandrayee/CTRL-SIMP).
- Model/API: Groq Chat Completions API; current local `GROQ_MODEL` is `openai/gpt-oss-120b` (verify the model again for any final experiment): [API docs](https://console.groq.com/docs/api-reference#chat-create), [model catalog](https://console.groq.com/docs/models).
- Libraries: Gradio, Hugging Face Hub client, NumPy, pandas, python-dotenv, requests, scikit-learn, pytest; versions/ranges are listed in `requirements.txt`.
- No tutorial source, external benchmark rows, or AI-generated dataset is currently used. Add any AI tools used by the team to prepare the submission before submitting.

**This project is a prototype and is not medical advice.**
