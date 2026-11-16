# ComprisMED

### Making medical instructions easier to understand

**Try the app:** [Open the ComprisMED demo](https://YOUR-DEPLOYED-APP-URL.vercel.app)
_Replace this placeholder with the public app URL after deployment._

ComprisMED is an AI + Health research prototype for older adults reviewing complicated instructions after an appointment. It turns supplied text into simpler bullet notes and checks selected details against the original. It is **not medical advice or a medical device, and has not been clinically validated**.

> **For the demo, use fictional instructions only. Never enter real patient information.**

The current application is a Python/Gradio app. A Vercel link can point to it, but a static Vercel page alone will not run the Python app; deploy the app backend on a Python-capable host or connect the Vercel frontend to a separately hosted backend. Keep the Groq API key on the server, never in browser code.

## What it does

- Includes a TF-IDF + logistic-regression jargon classifier trained on the small, synthetic `data/jargon_training.csv` starter set as a research component; it is not shown in the simplified user interface.
- Uses the Groq API for AI simplification. The prompt includes a style example from the synthetic jargon set, while still instructing the model to handle new wording generally.
- Lets users type/paste instructions or upload a fictional JPG/JPEG/PNG. RapidOCR runs locally in the app process; the image is not sent to Groq. The uploaded file is removed after preview preparation, and the preview/OCR image stay in session memory only. Users must review and confirm extracted text before simplification; only the text is sent to Groq when they select Simplify.
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

Create a file named `.env` in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholder with your own key from Groq, save the file, and restart the app. **Never share or commit your key.** The app does not display it. For a hosted deployment, configure the key as a private server-side environment variable instead of putting it in the repository or frontend.

The current configured model is `openai/gpt-oss-120b`. To change it, set `GROQ_MODEL` in `.env` to an active model from the [Groq model catalog](https://console.groq.com/docs/models). Groq may change its model catalog, API terms, limits, and pricing, so confirm those before the hackathon demo.

### Run from VS Code or PowerShell

Open a terminal in the project folder and run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for the local URL printed by Gradio and open it in your browser. Keep the terminal open while using the app; press `Ctrl+C` there to stop it. To use the regular `python` command instead, activate the environment first with `.\.venv\Scripts\Activate.ps1`.

The app's **Use sample instructions** button fills in fictional text. Or upload a fictional JPG, JPEG, or PNG, compare the editable OCR text with its preview, correct it, and confirm the review checkbox. OCR text cannot be simplified until this confirmation is checked; editing it afterward clears the confirmation. Select **Simplify** to generate a checked explanation. If selected details appear missing or changed, the app may request one repair and flags any remaining issues. Choose a language beside the result, then select **Translate**. A repair or translation makes an additional model call. API availability, quotas, and account billing apply. Images are processed locally and are not sent to Groq. Only instruction text is sent when simplification or translation is requested. Never use real patient information.

If the app reports an error:

- **Groq is not configured:** check that `.env` is in the project root and `GROQ_API_KEY` is non-empty, then restart the app.
- **Groq rejected the API key (401):** check for an expired, revoked, or mistyped key; create a replacement in Groq if needed.
- **Rate limit or quota (429):** wait or check the Groq account limits and billing.
- **Model not found (404):** set `GROQ_MODEL` to an available model in the Groq model catalog.
- **Could not connect:** check internet access and retry. A successful local page load does not by itself mean the external API is reachable.

### What a user does in the app

1. Double-click `Start_comprisMED.bat`; the app opens in a browser.
2. Select **Use sample instructions**, type fictional instructions, or upload a fictional JPG/JPEG/PNG.
3. For an image, inspect the preview, select **Extract text**, compare and correct the editable text, then check **I checked the extracted text against the photo**. If the photo is blurry or unreadable, retake it or type the text; no partial OCR result is silently simplified.
4. Select **Simplify**.
5. Read the simplified bullet notes. The detail area shows only missing or changed supported details; otherwise it gives a brief status. A clear status is not proof of medical correctness.
6. Choose a language beside the result and select **Translate**. Translation is not independently checked.

The main app is focused on the checked explanation and does not display the research comparison. The separate experiment below compares AI-only with AI + source-check + repair on held-out synthetic examples. Text submitted for AI features is sent to Groq.

Run the local tests from VS Code or PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pytest -q` directly.

## Experiment

The held-out synthetic checker test runs without Groq:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset synthetic --split test --checker-only
```

With `GROQ_API_KEY` set in the local `.env`, run the paired model experiment:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset synthetic --split test
```

The AI-only condition receives no gold-fact hints. The verified condition starts from the same AI output, checks explicitly labeled details, and makes at most one Groq repair call. Outputs, aggregate/category preservation, omission and alteration metrics, and actual example successes/failures are saved under `results/`. To test the secondary public set, first complete its independent annotation workflow in [data/public/README.md](./data/public/README.md), then run:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --dataset cochrane --split test
```

The synthetic benchmark contains 60 fictional scenarios across 12 instruction families: 20 train, 10 validation, and 30 held-out test examples. Entire families stay in one split. Each scenario has one project-author-selected important detail, explicitly paired with its controlled mutation; labels are not generated by the extractor. These labels have not been independently reviewed by a clinician. The test mutation results are controlled software tests, not evidence of clinical accuracy. Warnings and follow-up are included in source text but not covered by the current detail checker. Groq is used only through inference; no model is trained on the benchmark.

The latest deterministic test run flagged 30/30 deliberately altered test cases and 0/30 unchanged-source controls; this exact-string synthetic check is not a result about AI simplifications or clinical safety. See the saved [checker metrics](./results/synthetic_test_checker_metrics.json).

### Live Groq held-out experiment (2026-11-15)

The command above completed on the 30-example synthetic test split using `openai/gpt-oss-120b`. Each example has one project-authored selected fact. The two conditions use the same initial AI draft; the verified condition checks it and makes at most one repair call. Four repairs were attempted.

| Selected-fact outcome | AI-only | AI + detail verification |
|---|---:|---:|
| Examples / selected facts | 30 / 30 | 30 / 30 |
| Preserved | 28 / 30 (93.3%) | 29 / 30 (96.7%) |
| Flagged omissions | 1 / 30 (3.3%) | 1 / 30 (3.3%) |
| Flagged alterations | 1 / 30 (3.3%) | 0 / 30 (0%) |

| Detail type | AI-only preserved | AI + verification preserved |
|---|---:|---:|
| Dates | 8 / 10 (80.0%) | 9 / 10 (90.0%) |
| Durations | 7 / 7 (100%) | 7 / 7 (100%) |
| Frequencies | 3 / 3 (100%) | 3 / 3 (100%) |
| Negations | 4 / 4 (100%) | 4 / 4 (100%) |
| Amounts / quantities | 6 / 6 (100%) | 6 / 6 (100%) |

These are **literal-checker preservation results, not clinically adjudicated omission or alteration rates**. The matcher can flag meaning-preserving paraphrases or formatting differences as changes; therefore, the increase from 93.3% to 96.7% does not establish a reduction in medically meaningful errors. The test facts have not been independently reviewed by a clinician.

The separate controlled-mutation checker test detected 30/30 deliberately altered strings and raised 0 alarms on 30 unchanged-source controls (precision, recall, and F1: 1.00; false-alarm rate: 0.00). This is a deterministic software test on synthetic strings, **not** an AI-only versus AI-plus result, not a model-generated output evaluation, and not a clinical metric. False-alarm rates are not estimated for the live Groq conditions.

The exact generated records and aggregate metrics are [synthetic_test_outputs.json](./results/synthetic_test_outputs.json) and [synthetic_test_metrics.json](./results/synthetic_test_metrics.json). The metrics file includes examples of AI-only failures, repairs, and remaining verified failures. The output file contains all 30 paired examples.

The Cochrane dataset is a secondary source of medical research summaries, not telehealth instructions. Source-only reviewer templates are present, but independent reviews and adjudication are **not complete**; no Cochrane evaluation results are reported. Neither benchmark establishes clinical safety or clinical validation.

### Data included in this repository

| Path | Rows / scope | Contents and use | Data type |
|---|---:|---|---|
| `data/jargon_training.csv` | 40 | Small starter set for the TF-IDF + logistic-regression jargon classifier | Synthetic, project-authored |
| `data/synthetic/benchmark.jsonl` | 60 | Family-disjoint train/validation/test scenarios with one selected fact and a controlled mutation each | Synthetic, project-authored |
| `data/public/cochrane/train.json` | 3,568 | Research-summary source/target pairs; not post-telehealth instructions | Public, GEM Cochrane; CC BY 4.0 |
| `data/public/cochrane/validation.json` | 411 | Official validation split; not used as instruction data | Public, GEM Cochrane; CC BY 4.0 |
| `data/public/cochrane/test.json` | 480 | Official held-out split; 30 examples selected for annotation, which is not yet complete | Public, GEM Cochrane; CC BY 4.0 |
| `data/public/cochrane/reviewer_a.csv`, `reviewer_b.csv` | 30 each | Blank source-only templates; independent reviews are pending | Public source text; project annotation templates |
| `data/public/README.md` | — | Dataset scope, license, attribution, limitations, and annotation/run commands | Project documentation |

The Cochrane files are downloaded at a pinned revision with checksums by `data/public/download_cochrane.py`. Dataset counts, sizes, and attribution are documented in [data/public/README.md](./data/public/README.md).

## Intended demo story / exact user flow

1. Tell viewers: “This is ComprisMED, an AI + Health research prototype for older adults reviewing instructions after an appointment. This is a fictional example, not real patient information.”
2. Paste a **fictional** instruction such as: “Take 5 mg by mouth twice daily for 7 days. Do not stop early.”
3. Explain that the jargon panel uses a tiny synthetic starter classifier and that its term matches are clues, not a complete medical vocabulary.
4. Read the simplified bullet notes and the details flagged by the checker. Describe it as a limited literal check, not an understanding of clinical meaning.
5. Optionally translate the simplified output, then point out that the translation has not been independently verified and should be reviewed by a fluent speaker.
6. If the output is flagged, say that the prototype would direct a user to confirm the instruction with a clinician or pharmacist. Do not give the audience a real treatment recommendation.
7. For the research portion, run the separate experiment command and show its actual output metrics and examples, if available. The AI-only versus AI + verification comparison is not part of the regular app page. Never claim clinical validation or invent successful results.

## Data sources and reuse status

- **GEM Cochrane Simplification:** public medical research-summary source/target pairs, declared CC BY 4.0 and retained as secondary research data. It is not used for current model training or reported results, and it is not a medication-instruction benchmark. Scope, citation, pinned revision and limitations are in [data/public/README.md](./data/public/README.md).
- **Med-EASi:** task-relevant, but the dataset card does not declare a reuse license and upstream repository has no data license. Not downloaded or used.
- **Project benchmark and jargon examples:** fictional and synthetic, not patient data and not clinician-reviewed.
- **Glossary:** prototype explanations are authored paraphrases in `comprismed/glossary.py`, not bulk-copied MedlinePlus text. [MedlinePlus Health Definitions](https://medlineplus.gov/healthdefinitions/) is a public reference to investigate, not a dataset bundled here. Verify the source and reuse terms before incorporating glossary text.
- **Groq:** the app uses [Groq's OpenAI-compatible Chat Completions API](https://console.groq.com/docs/api-reference#chat-create). The reported run used `openai/gpt-oss-120b`; recheck current model/provider terms. Only send fictional/synthetic text from this prototype.

One live-model synthetic test-split run is now saved and summarized above. It is not clinical evidence or proof of medical accuracy. The local jargon training set is too small to support claims about real-world jargon detection.

## Research plan and interpretation

Primary comparison: baseline simplification versus one-check-and-repair verified simplification on the same held-out synthetic examples, using the same model settings. The 2026-11-15 run and its outputs are reported above. The metrics reflect only one labeled detail per instruction and a limited literal matcher. Do not present them as semantic or clinical correctness, and do not tune prompts on the held-out test cases.

Known limitations: synthetic English-only scenarios; tiny jargon seed corpus; literal matching can miss paraphrases or flag safe rewordings; incomplete number/date/frequency/negation patterns; warning/restriction meaning and clinical context are not comprehensively checked; model outputs vary; an LLM repair can introduce new errors; translation is not verified; no older-adult usability study, clinician review, clinical validation, or safety guarantees. For real use, require human confirmation and clinician/pharmacist contact when anything is unclear.

## Credits and citations

- Devaraj et al. (2021). [Paragraph-level Simplification of Medical Texts](https://aclanthology.org/2021.naacl-main.395/). NAACL-HLT 2021.
- [GEM Cochrane Simplification dataset](https://huggingface.co/datasets/GEM/cochrane-simplification), CC BY 4.0. This is research-summary data, not telehealth instructions; see [dataset notes and attribution](./data/public/README.md).
- [Groq API documentation](https://console.groq.com/docs/api-reference#chat-create) and [model catalog](https://console.groq.com/docs/models). Confirm current availability and terms before use.
- [RapidOCR](https://github.com/RapidAI/RapidOCR), [scikit-learn](https://scikit-learn.org/stable/), and [Gradio](https://www.gradio.app/docs).

No private patient data is included. The synthetic benchmark and jargon examples are project-authored. Med-EASi is not included because its dataset reuse license is unclear.
