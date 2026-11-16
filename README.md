
# ComprisMED

**Track:** AI + Health  
**Team:** Individual Project / Solo

## The Problem

Both of my grandparents regularly receive medical documents filled with unfamiliar terminology and complicated instructions. Almost every time, they come to me to translate the documents and explain what they actually mean in simpler words.

It made me wonder what happens when someone doesn't have a person available to help them understand their medical instructions.

ComprisMED explores how AI can make these documents easier to understand while helping preserve important details.

## What I Built

ComprisMED is an AI-powered application that simplifies complex medical instructions into clearer, more accessible language.

Users can:
- Paste medical instructions directly into the application.
- Upload a JPG, JPEG, or PNG image and extract its text using OCR.
- Review and correct extracted text before processing.
- Simplify instructions using AI.
- Translate the simplified explanation into another language.
- Review verification results that flag potentially missing or changed details.

**Research question:** Does adding source-grounded verification to AI simplification improve the preservation of important details?

## Try ComprisMED Yourself

You can run ComprisMED locally on your computer.

### Requirements
- Python 3.11
- A Groq API key
- An internet connection for AI processing

### 1. Download the project

Clone this repository or download it as a ZIP file.

```bash
git clone https://github.com/aksharabrahm10/ComprisMED.git
cd ComprisMED
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a file named `.env` in the project folder and add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholder with your own API key from [Groq](https://console.groq.com/).

Never share your API key or commit your `.env` file to GitHub.

### 4. Start the application

```bash
python app.py
```

Open the local URL printed in your terminal, usually:

`http://127.0.0.1:7860`

### Using the application

1. Enter fictional medical instructions or use the sample instructions.
2. Alternatively, upload an image of a fictional medical document.
3. If uploading an image, extract the text and review or correct it before confirming.
4. Select **Simplify** to generate a clearer explanation.
5. Review the simplified output and any details flagged by the verifier.
6. Optionally, select a language to translate the explanation.

**Note:** A hosted version is not currently available, so running the application locally requires your own Groq API key. Use fictional instructions only. Never enter real patient information.

## How It Works

ComprisMED compares two approaches:

| Approach | Description |
|---|---|
| AI-only baseline | Uses AI to simplify the original instructions. |
| AI + verification | Simplifies the instructions, then checks whether selected important details were preserved. |

The verification step checks five categories:

- **Numbers and dosages:** Checks whether important numerical details remain.
- **Frequency:** Checks how often an instruction should be followed.
- **Duration:** Checks how long an instruction applies.
- **Dates:** Checks whether relevant dates are preserved.
- **Negation:** Checks whether negative instructions, such as "do not", remain.

If a check fails, the application can attempt one repair using the original instructions. Any remaining issues are shown for human review.

### Technologies

| Component | Technology |
|---|---|
| Application | Python, Gradio |
| AI model | Groq API (`openai/gpt-oss-120b`) |
| OCR | RapidOCR |
| Verification | Python string matching and regular expressions |
| Jargon classification | TF-IDF and Logistic Regression |
| Classifier data | 40 synthetic training examples |

## Data

### Synthetic Benchmark

I created a benchmark of 60 synthetic scenarios across 12 instruction families to test whether selected important details were preserved.

- 20 training examples
- 10 validation examples
- 30 held-out test examples

Each scenario contains a selected important detail and a controlled variation. The benchmark was created for this project and has not been clinically reviewed.

### Cochrane Simplification Dataset

I also explored the GEM Cochrane Simplification dataset, which contains medical research summaries paired with simplified versions.

It provides additional context for medical text simplification, but research summaries are different from the post-appointment instructions ComprisMED focuses on. It was not used to fine-tune the Groq model or calculate the reported benchmark results.

## Results

The AI-only baseline was compared with AI + verification on 30 held-out synthetic examples.

| Metric | AI-only baseline | AI + verification |
|---|---:|---:|
| Selected details preserved | 28/30 (93.33%) | 29/30 (96.67%) |
| Omissions | 1 | 1 |
| Alterations | 1 | 0 |

The verification approach showed a raw improvement of 3.33 percentage points in selected-detail preservation.

These are preliminary, checker-based results. They are not proof of clinical accuracy or safety. The benchmark is synthetic, and the results have not been clinically adjudicated.

### Checker Testing

I also tested the deterministic checker separately using controlled synthetic strings:

- 30/30 deliberately altered examples were flagged.
- 30/30 unchanged-source examples were accepted.
- Precision: 1.00
- Recall: 1.00
- F1 score: 1.00

This tests the checker's performance on controlled examples, not the accuracy of AI-generated medical explanations.

## Limitations

ComprisMED is an early prototype, and several limitations remain:

- **Literal matching:** The verifier relies on string matching and regular expressions. It may flag equivalent wording, such as "2 weeks" and "14 days", as different.
- **No clinical verification:** It cannot determine whether medical advice or a dosage is clinically correct.
- **Limited context understanding:** It cannot reliably interpret ambiguous instructions, medical context, or equivalent units.
- **Translation:** Translated output is not independently verified.
- **Synthetic evaluation:** The benchmark is small, project-created, and not clinically reviewed.
- **Privacy:** The application has not undergone a security audit or clinical validation.

A successful verification check only means the selected literal checks passed. It does not guarantee that the explanation is medically accurate or complete.

## Privacy and Safety

Uploaded images are processed locally using RapidOCR and are not sent to Groq. Only extracted text is sent to Groq when AI processing is requested. Users can review and correct extracted text before sending it.

The application is designed to minimize information shared and does not intentionally persist uploaded documents. Groq is an external service, and its data-handling policies apply.

ComprisMED is an educational prototype, not a medical device or substitute for professional medical advice. Its outputs should always be reviewed by a qualified healthcare professional.

## Next Steps

- Improve verification to recognize equivalent wording and date formats.
- Expand the benchmark with more diverse instruction types.
- Evaluate the system with more examples and independent reviewers.
- Explore ways to improve accessibility for older adults.

## Sources

- Devaraj, Ashwin, Iain J. Marshall, Byron C. Wallace, and Junyi Jessy Li. (2021). "Paragraph-level Simplification of Medical Texts." NAACL-HLT 2021. [Paper](https://aclanthology.org/2021.naacl-main.395/)
- [GEM Cochrane Simplification Dataset](https://huggingface.co/datasets/GEM/cochrane-simplification) (CC BY 4.0)
- [Groq Documentation](https://console.groq.com/docs)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## AI Assistance

AI coding assistants were used during development for debugging, implementation support, and documentation. The project design, research question, benchmark construction, experiments, and evaluation were directed by me.
