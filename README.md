# MediVoice

**A multimodal lab-report interpreter built on Gemma 4, with a deterministic safety layer that never hallucinates an emergency into routine.**

Submitted to the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Health + Digital Equity tracks).

- **Kaggle notebook:** https://www.kaggle.com/code/kallurivenkatesh4416/medivoice-gemma-4-v19-lab-report
- **License:** Apache 2.0 (see `LICENSE`)
- **Deadline:** 2026-05-18

## What it does

Upload one or more phone photos of a CBC or CMP lab report. MediVoice produces a plain-English explanation at a sixth-grade reading level, classifies every value deterministically against the report's own reference range (with a curated fallback), cross-checks printed flags, and returns an urgency level (`routine` / `see_doctor_soon` / `er_now`) that cannot be overridden by free-form generation.

Supports five output languages (English, Spanish, Hindi, French, German), large-print mode, and voice input with Whisper fallback.

## Architecture

Four stages. Safety-critical decisions are pure Python, never LLM prose.

1. **Read** — Gemma 4 vision head is the primary reader. When its output is empty or low-signal, the same production path falls back to Tesseract OCR on the same image variants and picks whichever read yields more usable lab rows. The `selected_reader` field is recorded honestly in the trace.
2. **Structure** — Gemma 4 chat-template tool-calling produces schema-valid JSON.
3. **Decide** — Pure-Python classification, escalation, flag-mismatch detection, and citations against published critical-value tables (Mayo Clinic Laboratories, URMC, Texas DSHS, Interpath).
4. **Explain** — Gemma 4 prose grounded in the structured facts; deterministic fallback template in CPU smoke mode.

## Evaluation

22 scenario cases across CBC / CMP / pediatric / critical-value / flag-mismatch / OCR-noise edge cases. Five layered evaluations:

- Deterministic decision layer — 22/22 status match, 1.0 value-extraction accuracy, 1.0 classification accuracy, 1.0 citation grounding.
- Perception ablation — OCR-only vs Gemma-multimodal vs full production, same synthetic renders.
- Degraded synthetic-photo simulation — skew / blur / JPEG / uneven lighting.
- Multilingual validation — disclaimer + action text preserved 1.0/1.0 across all five languages.
- Safe-failure scenarios — pregnancy refusal, missing-context clarification, prompt injection, hard-emergency keyword. 1.0 pass rate.

## Honest limitations

- Multimodal extraction on real phone photos is weak; the Tesseract fallback exists for this reason. Synthetic eval corpus is not a substitute for clinical validation.
- No clinician-in-the-loop review yet. Deterministic safety guarantees stand; the impact claim is architectural, not empirical.
- Pediatric coverage is intentionally narrow (7 analytes × 4 age bands). Any analyte outside that grid returns a `pediatric_coverage_gap` flag rather than reusing adult ranges.
- On GPU allocations below compute capability 7.0 (e.g. P100), the full image path falls into CPU smoke mode.

## Repo contents

- `build_notebook_v19.py` — source of truth; emits the Kaggle notebook.
- `medivoice_gemma4_v19_lab_report.ipynb` — generated notebook submitted to Kaggle.
- `_validate_eval.py` — CPU-safe validator covering the deterministic, multilingual, and safe-failure layers.
- `kaggle_v19_run/` — kernel metadata and staged notebook copy used for pushes.
- `kaggle_output_*/medivoice_v19_outputs/` — sample export bundles (metrics, eval tables, writeup, judge readme, checksums).
- `assets/` — cover image and any redacted sample inputs.
- `LICENSE` — Apache 2.0.

## Reproducing

```bash
pip install -r requirements.txt            # or install transformers==5.5.3, huggingface_hub, hf_transfer, gradio, pytesseract, etc.
python build_notebook_v19.py                # regenerate the ipynb
python _validate_eval.py                    # CPU-safe deterministic validation
cd kaggle_v19_run && kaggle kernels push -p .   # push to Kaggle (requires configured kaggle CLI)
```

On Kaggle, open the notebook, **Run all**, and wait for a T4 allocation for the full multimodal path. The export cell writes a timestamped ZIP under `/kaggle/working/medivoice_v19_outputs_*.zip`.

## Disclaimer

MediVoice is educational software, not a medical device. All output is for patient understanding, never diagnosis. Always review lab results with a qualified healthcare professional.
