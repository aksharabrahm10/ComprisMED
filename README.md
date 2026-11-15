# comprisMED

comprisMED is an AI + Health research prototype exploring whether source-grounded checks can help preserve selected important details when simplifying post-telehealth instructions. It is **not medical advice, not a medical device, and not clinically validated**. Use only fictional/synthetic examples; never enter real patient text or identifiers.

## What it does

- Detects jargon with a TF-IDF + logistic-regression sentence classifier trained on the small, synthetic `data/jargon_training.csv` starter set, and highlights terms in a small locally authored glossary.
- Uses the Groq API for AI simplification. The current `.env` selects `openai/gpt-oss-120b`; the API key remains local and is never displayed. The prompt includes a style example from the synthetic jargon set, while still instructing the model to handle new wording generally.
- Checks whether selected dose/number, frequency, duration, date, and negation markers appear unchanged in the output.
- Compares an AI-only draft with an AI + detail-check draft. In the checked path, a failed check can trigger one Groq repair, after which remaining failures are shown for human review.
- Optionally translates the checked draft with Groq. Translation is not independently checked, so do not treat it as verified.
- Runs a paired experiment on held-out, explicitly synthetic instructions and reports omissions and alterations separately, plus detection precision, recall, F1, and false alarms for deliberate mutations.

The checks are deliberately conservative string/regex checks. They do not understand clinical meaning, dose equivalence, ambiguous wording, context, units conversion, or whether the output added unsafe advice. A green check means only that the included literal checks passed.

## Start the actual app (Windows)

**Normal use does not require typing commands in a terminal.**

1. Install Python 3.9+ if it is not already installed.
2. Open the project folder in File Explorer and double-click `Start_comprisMED.bat`.
3. The first launch creates the private Python environment and installs dependencies. It then starts the local app and opens it in your browser. The usual address is `http://127.0.0.1:7860`; if that port is already in use, Gradio may select the next available port and prints the address in the launcher window.
4. Leave the launcher window open while using the app. Close it to stop the local server.

### One-time Groq key setup

The project already has a local `.env` file. Check that it contains a non-empty `GROQ_API_KEY=...` line; if you need to replace the key, create one in the Groq console and paste it after the equals sign without quotes or spaces. Save the file and restart the app. **Do not paste the key into chat, screenshots, source code, or a commit.** The app never displays the key.

The current configured model is `openai/gpt-oss-120b`. To change it, set `GROQ_MODEL` in `.env` to an active model from the [Groq model catalog](https://console.groq.com/docs/models). Groq may change its model catalog, API terms, limits, and pricing, so confirm those before the hackathon demo.

### Run from VS Code or PowerShell

Open a terminal in the project folder and run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for the local URL printed by Gradio and open it in your browser. Keep the terminal open while using the app; press `Ctrl+C` there to stop it. To use the regular `python` command instead, activate the environment first with `.\.venv\Scripts\Activate.ps1`.

The app's **Use example** button fills in fictional text. Select **Simplify and compare** to send requests to Groq and see the AI-only draft beside the AI + detail-check result. The request makes at least two model calls; a repair may make another. API availability, quotas, and account billing apply. Do not use real patient information.

If the app reports an error:

- **Groq is not configured:** check that `.env` is in the project root and `GROQ_API_KEY` is non-empty, then restart the app.
- **Groq rejected the API key (401):** check for an expired, revoked, or mistyped key; create a replacement in Groq if needed.
- **Rate limit or quota (429):** wait or check the Groq account limits and billing.
- **Model not found (404):** set `GROQ_MODEL` to an available model in the Groq model catalog.
- **Could not connect:** check internet access and retry. A successful local page load does not by itself mean the external API is reachable.

### What a user does in the app

1. Double-click `Start_comprisMED.bat`; the app opens in a browser.
2. Select **Use example** or paste fictional instructions into the input. Never use real patient information.
3. Select **Simplify instructions**. The app generates an ordinary AI draft, checks selected details against the source, and shows the checked/possibly repaired draft as short bullet notes.
4. Read the detail-check summary below the notes. “Preserved” means only that a supported literal check matched; it does not prove medical correctness. Expand **Compare AI-only and AI + detail check** to inspect both versions.
5. The translation controls are visible below the simplified result. Choose a language and select **Translate**. Translation is not independently checked.

Both drafts come from the AI; the comparison reuses the first draft as its baseline and shows the result after the verifier has checked it and, when needed, requested one repair. The detail checks are not a guarantee of accuracy, and text submitted for AI features is sent to Groq.

Run the local tests from VS Code or PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pytest -q` directly.

## Experiment

With the Groq key set, run the held-out split:

```powershell
python experiments\run_experiment.py
```

By default this evaluates only the `test` split and writes generated outputs and aggregate/category metrics to `results/`. The experiment makes model API calls and may incur costs. For development, set `$env:CARETRACE_SPLIT="train"` or `$env:CARETRACE_SPLIT="validation"` in PowerShell before running. Do not tune on test examples; if test-driven changes are made, report the leakage and make a new held-out split. It compares an unprompted AI-only output against an AI + source-check + one repair output and reports checker metrics overall and by category.

The benchmark has 60 fictional instruction scenarios across 12 instruction families: 20 train, 10 validation, and 30 held-out test. Entire families are assigned to a single split to reduce template leakage. Expected facts are project-authored separately from the parser under evaluation; see `annotate()` in the generator. The annotations are not independently adjudicated and do not establish annotation validity. Every scenario contains one controlled mutation; the test set includes 30 mutation cases. Warnings/follow-up/context are present in source texts but only the explicit fact categories are checked. The train split is a benchmark partition, not LLM training; Groq is used through inference only. This small authored benchmark is not an external validation set. Scripted metrics assess only annotated literal facts; they are not a clinical safety score.

### Data included in this repository

| File | Size (current) | Contents and use | Data type |
|---|---:|---|---|
| `data/jargon_training.csv` | 2,047 bytes; 40 labeled rows | Tiny starter set for fitting TF-IDF + logistic regression jargon classifier | Synthetic, project-authored |
| `data/synthetic/benchmark.jsonl` | 84,942 bytes; 60 scenarios | Train/validation/test evaluation cases with parser-extracted facts and altered variants | Synthetic, project-authored |
| `data/synthetic/build_benchmark.py` | Generator source | Rebuilds the JSONL benchmark from authored scenario definitions and derives extracted-fact labels using the project parser | Project code; no external data |

No public dataset files are bundled or used for training or evaluation. Med-EASi and other candidates below are cited for investigation only.

## Intended demo story / exact user flow

1. Tell viewers: “This is comprisMED, an AI + Health research prototype for older adults reviewing instructions after a telehealth appointment. This is a fictional example, not real patient information.”
2. Paste a **fictional** instruction such as: “Take 5 mg by mouth twice daily for 7 days. Do not stop early.”
3. Explain that the jargon panel uses a tiny synthetic starter classifier and that its term matches are clues, not a complete medical vocabulary.
4. Run **baseline** and show the draft plus the detail-check table.
5. Run **verified** with the same source; show the initial draft, any repair, and any remaining flags. Describe the verifier as limited checks for literal details—not an understanding of clinical meaning.
6. If the output is flagged, say that the prototype would direct a user to confirm the instruction with a clinician or pharmacist. Do not demonstrate by giving the audience a real treatment recommendation.
7. Optionally translate the simplified output, then point out that the translation has not been independently verified and should be reviewed by a fluent speaker.
8. End by showing the experiment metrics and one miss/false alarm, if present. Never claim clinical validation or invent successful results.

## Data sources and reuse status (checked 2026-11-14)

- **Synthetic preservation benchmark:** `data/synthetic/benchmark.jsonl` is authored for this project and labeled synthetic. It is not clinical data and is not drawn from a patient record.
- **Synthetic jargon examples:** `data/jargon_training.csv` contains the starter labeled examples used by the local TF-IDF + logistic-regression classifier. It is not public or clinician-reviewed data.
- **Med-EASi:** [Hugging Face dataset card](https://huggingface.co/datasets/cbasu/Med-EASi), [paper](https://arxiv.org/abs/2302.09155), and [authors' repository](https://github.com/Chandrayee/CTRL-SIMP). The card reports 1,979 short medical simplification pairs and transformation annotations. The public dataset card has no data license declared; the repository code's MIT license does not establish the dataset's reuse rights. We therefore do **not** download, bundle, or train on its rows. Ask the dataset authors for permission/terms before using it. It is relevant to simplification, not a post-telehealth fact-preservation benchmark.
- **Medical texts simplification (Hugging Face):** [dataset card](https://huggingface.co/datasets/liliya-makhmutova/medical_texts_simplification) reports 30 triples (around 800 sentences) with original, human-simplified, and ChatGPT-simplified text; metadata labels it CC-BY-NC-4.0. The card says the source is the Kaggle [Medical Notes Classification competition](https://www.kaggle.com/competitions/medicalnotes-2019/rules), that human simplifications were not medically reviewed, and that use for treatment is not appropriate without professional supervision. The records are clinical notes, not synthetic telehealth instructions; the underlying Kaggle terms/de-identification could not be verified here. We do **not** bundle or send these notes to an API. It is a possible later research candidate only after checking both source-data permission and privacy/de-identification.
- **Other Hugging Face result:** `Sadou/medical-reports-simplification-dataset` is described as generated by Gemini with automated validation and carries an Apache-2.0 metadata label, but that does not make its outputs human- or clinician-validated. It is not used.
- **Hugging Face `huhucheck/medical_jargon`:** search result indicates a tiny JSON artifact; the dataset card/reuse license and suitability are not established. We do not use it.
- **Kaggle and UCI searches:** Kaggle surfaced the upstream Medical Notes Classification competition, but its rules and data privacy status could not be verified from the accessible page; UCI's text search did not surface a relevant dataset. No external dataset is included.
- **Glossary:** prototype explanations are authored paraphrases in `comprismed/glossary.py`, not bulk-copied MedlinePlus text. [MedlinePlus Health Definitions](https://medlineplus.gov/healthdefinitions/) is a public reference to investigate, not a dataset bundled here. Verify the source and reuse terms before incorporating glossary text.
- **Groq:** the app uses [Groq's OpenAI-compatible Chat Completions API](https://console.groq.com/docs/api-reference#chat-create) and the model configured in `.env`. Cite Groq and the exact model used in the final submission; recheck current model/provider terms. Only send fictional/synthetic text from this prototype.

No external dataset is currently included or required to run the prototype. The local jargon training set is synthetic and too small to support claims about real-world jargon detection; it is a scaffold to replace with an appropriately licensed and independently labeled dataset. No API experiment results exist until a user runs the Groq-backed script; the unit tests only exercise local checks and mocked API behavior.

## Research plan and interpretation

Primary comparison: baseline simplification versus one-check-and-repair verified simplification on the same held-out synthetic examples, using the same model settings. Report source-fact omission rate and alteration rate separately. Run injected-error tests and report precision, recall, F1, and false alarms. Include the examples and state the small sample size. Do not tune prompts on the held-out test cases.

Known limitations: synthetic English-only scenarios; tiny jargon seed corpus; literal matching can miss paraphrases or flag safe rewordings; incomplete number/date/frequency/negation patterns; warning/restriction meaning and clinical context are not comprehensively checked; model outputs vary; an LLM repair can introduce new errors; translation is not verified; no older-adult usability study, clinician review, clinical validation, or safety guarantees. For real use, require human confirmation and clinician/pharmacist contact when anything is unclear.

## Suggested weekend completion checklist

- [ ] Add the Groq key locally; never commit it.
- [ ] Expand synthetic examples and have another person review source facts and deliberate mutations.
- [ ] Ask Med-EASi authors about dataset license/reuse before adopting its data.
- [ ] Run baseline and verified modes on frozen held-out data; save outputs.
- [ ] Inspect false alarms and missed injected errors; report results as measured.
- [ ] Have fluent speakers review demo translations.
- [ ] Make the one-page write-up and <2 minute video from actual runs and include the limits above.
