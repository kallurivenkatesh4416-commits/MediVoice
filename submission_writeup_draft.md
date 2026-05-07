# MediVoice: Plain-English Lab Report Companion Powered by Gemma 4

- **Kaggle notebook:** https://www.kaggle.com/code/kallurivenkatesh4416/medivoice-gemma-4-v19-lab-report
- **GitHub repo:** https://github.com/kallurivenkatesh4416-commits/MediVoice
- **Demo video:** *(to be added before submission)*
- **License:** Apache 2.0

MediVoice is a safety-first lab-report and voice-assistant companion for people who receive results outside a clinic and need a calm, trustworthy explanation before they can reach a clinician. A user uploads a photo of a lab report, MediVoice reads the report with **Gemma 4 multimodal input as the primary reader**, falls back to OCR when Gemma's read is weak, converts the extracted rows into structured lab values, applies a **deterministic Python safety layer** for classification and escalation, and then uses Gemma 4 again to produce a plain-English explanation with a fixed medical disclaimer.

The notebook also includes a **Voice Chat** path. A patient can speak a symptom question or follow-up, MediVoice transcribes the audio through Gemma-native audio when available or Whisper fallback when needed, checks emergency guardrails, and returns a structured, plain-language response. The goal is not to replace a doctor. The goal is to reduce confusion, surface urgency safely, and make lab results and health questions more understandable for patients, caregivers, and community health workers.

This project fits three tracks directly:

- **Health & Sciences:** it focuses on CBC, CMP, and common chemistry/electrolyte-style reports, where patients often see dense tables before they get a callback.
- **Safety & Trust:** life-threatening decisions do not depend on free-form generation. Critical values, clarification rules, pregnancy refusal, and flag-mismatch detection are all handled by deterministic code.
- **Digital Equity & Inclusivity:** the interface supports large-print mode, multilingual output, microphone/uploaded audio, voice-chat follow-ups, and a calm explanation style designed for non-clinical readers.

## Why Gemma 4 specifically

Gemma 4 matters here in three ways.

First, MediVoice uses **Gemma 4's native multimodal input** to read lab-report images directly. The notebook gives Gemma the report image as an image content block, asks for row-preserving transcription, and then structures the result for downstream interpretation. This is the core “picture understanding” capability the project is built around.

Second, Gemma 4 is used for the **structure and explanation stages**. After the image/text perception step, Gemma helps turn noisy report text into schema-shaped JSON and then writes a calm, patient-facing explanation in plain language. This is where the model adds usability, multilingual reach, and a more supportive user experience.

Third, Gemma 4 is used in a way that stays honest about medical safety. On hard images or unsupported formats, MediVoice does not trust free-form model output to make critical decisions. Instead, the system pairs Gemma with a deterministic escalation layer and an OCR fallback. In practice, this makes the project a **Gemma-first multimodal workflow with explicit safeguards**, rather than a “model does everything” demo.

## Architecture

MediVoice follows a four-stage pipeline:

1. **Read**  
   Gemma 4 attempts to read the uploaded image first. If the read is weak or malformed, the production path falls back to Tesseract OCR on the same image variants and records which reader won.

2. **Structure**  
   The Structure stage uses Gemma 4's chat-template tool-calling pattern to produce schema-shaped JSON from noisy report text, with Python validation enforcing the final schema. Each row includes analyte name, value, unit, reference range, and printed flag.

3. **Decide**  
   A deterministic Python layer classifies each row, applies fallback ranges only when needed, checks for critical thresholds, detects printed-flag mismatches, and computes the final escalation level: `routine`, `see_doctor_soon`, or `er_now`.

4. **Explain**  
   Gemma 4 produces a short plain-English explanation grounded in the structured results and the deterministic decision layer. The fixed disclaimer remains present in every output.

This separation is the main design choice of the project: **Gemma handles perception and explanation, while safety-critical decisions remain deterministic.**

The demo exposes three judge-facing tabs:

- **Lab Report:** upload or select sample CBC/CMP-style report images.
- **Voice Chat:** ask medical education questions by microphone/audio upload or text, with emergency guardrails.
- **Eval Dashboard:** inspect deterministic metrics, multilingual validation, and safe-failure evidence.

## What the latest clean T4 run shows

The strongest evidence in the current submission is a clean Kaggle T4 run with non-null multimodal metrics. The evaluation corpus covers 22 scenario cases spanning CBC, CMP, pediatric, critical-value, flag-mismatch, OCR noise, and edge-case categories.

### Deterministic / decision-layer evaluation

The deterministic layer stayed perfect on all 22 cases:

- `status_match = 1.0`
- `value_extraction_accuracy = 1.0`
- `classification_accuracy = 1.0`
- `flag_mismatch_accuracy = 1.0`
- `citation_grounding_rate = 1.0`
- `safety_escalation_pass_rate = 1.0`

This matters because the deterministic layer is the final safety backstop. If a dangerous value is present in the structured rows, the escalation decision does not depend on how persuasive or cautious the model sounds.

### End-to-end image path

The latest clean T4 run also produced real end-to-end image metrics instead of null placeholders:

- `multimodal_status_match = 1.0`
- `multimodal_value_extraction_accuracy = 0.649`
- `multimodal_classification_accuracy = 0.675`
- `multimodal_escalation_pass_rate = 0.864`

*(Latest confirmed clean T4 run. These will be updated to the final v31 numbers once that run completes.)*

These are materially better than the earlier weak multimodal run and show that the production image path is no longer collapsing on the full synthetic benchmark.

### Perception ablation

The perception ablation compares three methods on the same images:

- `ocr_text_baseline`
- `gemma_multimodal_structured`
- `full_medivoice`

On the current ablation subset, all three methods reached comparable row-level score bands. That is not a claim that Gemma vision alone beats OCR. Instead, it supports a more honest conclusion:

> Gemma is a meaningful part of the production workflow, but for hard report images the strongest system today is the **hybrid pipeline**: Gemma-first, OCR fallback, deterministic safety.

### Real-world report formats

The project was extended to handle more than textbook CBC/CMP names. Recent coverage expansions add support for:

- CO2 / bicarbonate variants
- magnesium, phosphorus, ionized calcium, anion gap
- Indian-format analyte names and units such as `Random Blood Sugar`, `Blood Urea`, `cells/cumm`, and `lakhs/cumm`
- common CBC differential names and supporting chemistry items such as `PCV`, `Hematocrit`, `RBC`, `MCV`, `MCH`, `MCHC`, `Uric Acid`, and `Lactate`

This is important because many real reports do not use clean US-style labels.

## Accessibility and equity

MediVoice is designed to reduce friction for non-expert users:

- multilingual output with preserved safety text
- large-print mode
- voice assistant path for spoken questions and follow-ups
- calm, companion-style explanations
- open-weight, local-first model stack rather than external inference APIs

The multilingual validation remains strong:

- `disclaimer_preserved = 1.0`
- `action_text_preserved = 1.0`

That makes the project more than a visual OCR demo. It is an accessibility-oriented communication tool.

## Safety features

Several safeguards are deliberate and central to the submission:

- deterministic escalation thresholds for dangerous values
- fixed disclaimer that remains present in output
- `needs_clarification` for missing patient context where fallback ranges would be unsafe
- refusal path for pregnancy
- `flag_mismatch` warnings when a printed flag disagrees with the numeric interpretation
- safe-failure demonstrations covering unreadable rows, prompt-injection text, pediatric coverage gaps, and other failure modes

The safe-failure validation remains at `1.0` pass rate.

The deterministic layer covers 35 analytes with fallback reference ranges, 12 critical-value thresholds sourced from Mayo Clinic Laboratories, URMC, Texas DSHS, and Interpath, and 22 plain-English explanations.

## Honest limitations

This is still a prototype, and the limitations are important:

1. **Gemma vision is not consistently strong enough on its own for hard report photos.**  
   The latest clean T4 run showed that the production path often selected Tesseract as the winning reader on the synthetic image benchmark. That is why the project should be framed as a **Gemma-first hybrid system**, not as a pure Gemma OCR replacement.

2. **Real phone-photo robustness is still limited.**  
   The strongest benchmark evidence comes from synthetic report renders and degraded synthetic photo simulation. Those are useful for controlled comparison, but they are not the same as broad real-world validation on many phone photos.

3. **CBC/CMP-style reports are still the strongest supported format.**  
   The project now covers more chemistry/electrolyte and Indian-format report variants, but extraction quality is still uneven on blood-gas strips, highly custom hospital layouts, and images with tiny text or curved perspective. CMP and CBC extraction quality can also be asymmetric depending on report layout.

4. **The system is not clinically validated.**  
   There is no clinician-in-the-loop study, no deployment trial, and no patient-outcomes evidence. The strongest claim here is architectural safety and measured benchmark behavior, not clinical efficacy.

5. **Pediatric coverage is intentionally narrow.**  
   When pediatric reference data is not encoded, the system surfaces a `pediatric_coverage_gap` rather than silently applying adult fallback ranges.

## What makes MediVoice different

MediVoice is not just a prompt wrapped around a model. It is a carefully scoped healthcare prototype with:

- a meaningful Gemma 4 multimodal use case
- deterministic safety logic for escalation
- honest fallback behavior instead of fabricated confidence
- multilingual accessibility features
- a strong evaluation and artifact package for judges

The project’s core claim is not “Gemma solves clinical interpretation by itself.” The stronger and more credible claim is:

> Gemma 4 can power a useful multimodal patient-facing workflow when paired with OCR fallback and a deterministic safety layer that refuses to hallucinate urgency.

That combination is exactly what makes MediVoice a practical, trustworthy hackathon submission rather than a flashy but brittle demo.
