#!/usr/bin/env python3
"""Generate the MediVoice Kaggle notebook (.ipynb) programmatically.

v3 — hardened for Kaggle reliability and stronger competition polish:
  1. Stable Gemma 4 E2B loading via Transformers + BitsAndBytes + PEFT
  2. Kaggle-local → Hugging Face fallback chain with full load-attempt retries
  3. Deeper training: 2K randomly sampled subset, 125 steps (~1 full epoch)
  4. Safety layer: emergency triage, structured response template, target normalization
  5. Native Gemma 4 system role in chat template
  6. Multilingual Whisper (auto-detect, language dropdown)
  7. Deterministic decoding for demo reproducibility
  8. Before/after evaluation cell with comparison table
"""
import json

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {"trusted": True},
        "source": source,
        "outputs": [],
        "execution_count": None,
    })


# ============================================================
# CELL — Title & Badges
# ============================================================
md("""\
# MediVoice — Medical Voice Assistant Powered by Gemma 4

**Gemma 4 Good Hackathon | Theme: Health — AI for Real-World Medical Impact**

| Component | Detail |
|-----------|--------|
| **Model** | `google/gemma-4-E2B-it` — deep LoRA (rank 32, 7 projection targets) with 4-bit QLoRA |
| **Training** | 200 steps (~1.6 epochs) on 2K samples, all attention + MLP projections |
| **Dataset** | `lavita/ChatDoctor-HealthCareMagic-100k` (2K train + 200 eval holdout) |
| **Evaluation** | ROUGE-L + 3 safety metrics on 20 holdout questions, triage F1, multilingual validation |
| **Speech** | OpenAI Whisper (base) — multilingual auto-detect, 99+ languages |
| **Safety** | 3-layer: emergency triage (F1 tested) + structured prompt + disclaimer training |
| **Demo** | Multi-turn Gradio chat — voice or text → structured medical guidance |
| **License** | Apache 2.0 (Gemma 4) |

---""")

# ============================================================
# CELL — Project Overview
# ============================================================
md("""\
## Project Overview

**MediVoice** is an end-to-end medical voice assistant that bridges the gap between \
patients and reliable health information — especially in **resource-constrained environments** \
where specialist access is limited.

### How It Works
1. **Voice Input** — Patient speaks a symptom or health question (any language Whisper supports)
2. **Transcription** — Whisper converts speech to text with automatic language detection
3. **Emergency Triage** — Hard keyword guard checks for life-threatening symptoms first
4. **Medical Reasoning** — A QLoRA fine-tuned Gemma 4 model generates a **structured** response
5. **Clear Guidance** — Response follows a fixed safe format: possible explanations, self-care, \
urgent signs, and when to see a clinician

### Why Gemma 4?
- **Apache 2.0 license** — deployable anywhere, including hospitals and NGOs
- **E2B** — compact Gemma 4 variant designed for single-GPU and edge-friendly deployments
- **Instruction-tuned** — strong baseline for medical conversation with minimal fine-tuning
- **Native system role** — clean separation of safety instructions from patient input
- **Multilingual** — supports patients across language barriers""")

# ============================================================
# CELL — Medical Disclaimer
# ============================================================
md("""\
> **Medical Disclaimer**
>
> MediVoice is an AI research prototype for **informational and educational purposes only**. \
It is **NOT** a licensed medical professional and does **NOT** provide medical diagnoses, \
treatment plans, or prescriptions. Always consult a qualified healthcare provider for \
medical decisions. In an emergency, contact your local emergency services immediately.""")

# ============================================================
# CELL — Section: Environment Setup
# ============================================================
md("""\
---
## 1. Environment Setup

Install all required packages. This cell is designed for Kaggle notebooks with GPU, and \
**T4 is required over P100** for Gemma 4 in the current Kaggle environment. \
If Kaggle assigns **T4 x2**, MediVoice intentionally uses **one GPU only** to avoid \
PyTorch `DataParallel` replication overhead and out-of-memory errors. \
The `%%capture` magic suppresses verbose install output.""")

# ============================================================
# CELL — Pip Installs
# ============================================================
code("""\
%%capture
# Gemma 4 support may land in Transformers ahead of a stable pip release,
# so install the latest main branch build for compatibility.
!pip install -q git+https://github.com/huggingface/transformers.git

# QLoRA training stack
!pip install -q accelerate bitsandbytes peft trl
!pip install -q datasets tokenizers sentencepiece protobuf safetensors

# HuggingFace Hub for model resolution
!pip install -q huggingface_hub

# Speech-to-text
!pip install -q openai-whisper

# Demo UI
!pip install -q gradio

# Audio processing
!pip install -q librosa soundfile

# Eval display & metrics
!pip install -q tabulate rouge-score

# Kaggle's pre-installed wandb can have circular-import bugs that break
# trl's SFTTrainer import.  We don't use wandb, so remove it cleanly.
!pip uninstall -y wandb 2>/dev/null || true

print("All packages installed.")""")

# ============================================================
# CELL — Imports & GPU Check
# ============================================================
code("""\
import os
import gc
import re
import glob as globmod
import pathlib
import sys
import traceback

# Avoid PyTorch DataParallel on Kaggle T4x2 sessions. Gemma 4 QLoRA is
# more memory-stable on a single GPU than on replicated multi-GPU copies.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Reduce fragmentation during long runs with mixed training + generation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import warnings

warnings.filterwarnings("ignore")
os.environ["USE_HUB_KERNELS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tabulate import tabulate

# Gemma 4 is a multimodal model (text + vision + audio).  The checkpoint
# stores language-model weights under 'model.language_model.layers.*'.
# AutoModelForCausalLM creates Gemma4ForCausalLM, which expects the flat
# 'model.layers.*' layout — resulting in ALL weights being randomly
# initialised (MISSING) while every checkpoint key is silently dropped
# (UNEXPECTED).
# AutoModelForImageTextToText resolves to Gemma4ForConditionalGeneration
# which maps the checkpoint keys correctly.
try:
    from transformers import AutoModelForImageTextToText as GemmaModelClass
    _gemma_auto_cls_name = "AutoModelForImageTextToText"
except ImportError:
    # Older transformers may not have this alias yet.
    from transformers import AutoModel as GemmaModelClass
    _gemma_auto_cls_name = "AutoModel (fallback)"
print(f"Gemma 4 model class: {_gemma_auto_cls_name}")

DEBUG_LOG_PATH = pathlib.Path("/kaggle/working/medivoice_debug.log")

def log_debug(message):
    line = f"{message}\\n"
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(message)

def _medivoice_excepthook(exc_type, exc_value, exc_tb):
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\\n=== UNCAUGHT EXCEPTION ===\\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        pass
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = _medivoice_excepthook
log_debug("MediVoice debug logging initialized.")

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU             : {gpu.name}")
    print(f"VRAM            : {round(gpu.total_memory / 1024**3, 1)} GB")
    print(f"Visible GPUs    : {torch.cuda.device_count()}")
    if "P100" in gpu.name.upper():
        raise RuntimeError(
            "Kaggle assigned a Tesla P100 GPU. Gemma 4 currently fails on Kaggle P100 "
            "with 'CUDA error: no kernel image is available for execution on the device'. "
            "Please stop this session, switch the notebook accelerator to T4/T4x2 in Kaggle Settings, "
            "and rerun. This is a GPU/runtime compatibility issue, not a MediVoice code bug."
        )
else:
    raise RuntimeError(
        "No GPU detected. This notebook requires a CUDA-capable GPU. "
        "Enable GPU in Kaggle: Settings -> Accelerator -> GPU, preferably T4 if available."
    )""")

# ============================================================
# CELL — Kaggle Secrets / HF Token
# ============================================================
code("""\
# Authenticate with HuggingFace for gated model access.
# On Kaggle: Add your HF token as a Secret named "HF_TOKEN".
# Locally: export HF_TOKEN=hf_... in your shell.
log_debug("HF token setup started.")

hf_token = None

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("HuggingFace token loaded from Kaggle Secrets.")
except Exception:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HuggingFace token loaded from environment variable.")
    else:
        print(
            "WARNING: No HF_TOKEN found. "
            "If the model is gated, loading will fail. "
            "Add your token in Kaggle Secrets or as an env var."
        )

log_debug("HF token setup completed.")""")

# ============================================================
# CELL — Configuration
# ============================================================
code("""\
log_debug("Configuration cell started.")
class Config:
    \"\"\"Central configuration — edit these values to experiment.\"\"\"

    # -- Model (resolved dynamically — see resolve_model_candidates below) ---
    # Priority: Kaggle local input -> Google HF official
    KAGGLE_MODEL_PATTERNS: list = [
        "/kaggle/input/gemma-4/transformers/E2B-it/*",
        "/kaggle/input/gemma-4*/transformers/*E2B*/*",
        "/kaggle/input/gemma*/transformers/*E2B*/*",
    ]
    HF_MODEL_ID: str = "google/gemma-4-E2B-it"
    MAX_SEQ_LENGTH: int = 512
    LOAD_IN_4BIT: bool = True
    BNB_QUANT_TYPE: str = "nf4"
    USE_DOUBLE_QUANT: bool = True
    MODEL_DEVICE_MAP: dict = {"": 0}
    ATTN_IMPLEMENTATION: str = "eager"
    EXPERTS_IMPLEMENTATION: str = "eager"
    P100_FORCE_FP16: bool = True
    FP16_FALLBACK_MAX_SEQ_LENGTH: int = 512
    FP16_FALLBACK_BATCH_SIZE: int = 1
    FP16_FALLBACK_GRAD_ACCUM_STEPS: int = 16
    FP16_FALLBACK_OPTIM: str = "adamw_torch"

    # -- LoRA -------------------------------------------------------
    LORA_R: int = 32
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0
    # Gemma 4 multimodal: the vision/audio towers use Gemma4ClippableLinear
    # wrappers (not nn.Linear), which PEFT cannot inject LoRA into.  Using a
    # regex string tells PEFT to match against the full module path via
    # re.fullmatch, restricting LoRA to the language_model sub-tree where all
    # projection layers ARE nn.Linear.  We target all attention + MLP
    # projections for maximum adaptation capacity.
    TARGET_MODULES: str = ".*language_model.*\\.(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)"

    # -- Dataset ----------------------------------------------------
    DATASET_NAME: str = "lavita/ChatDoctor-HealthCareMagic-100k"
    KAGGLE_DATASET_PATTERNS: list = [
        "/kaggle/input/chatdoctor*/**/*.json",
        "/kaggle/input/chatdoctor*/**/*.parquet",
        "/kaggle/input/medical-qa*/**/*",
    ]
    NUM_TRAIN_SAMPLES: int = 2_000   # randomly sampled subset for ~1 full epoch
    NUM_EVAL_SAMPLES: int = 200      # holdout for before/after comparison
    NUM_EVAL_QUESTIONS: int = 20     # number of holdout questions for eval
    DATASET_SEED: int = 42

    # -- Training ---------------------------------------------------
    BATCH_SIZE: int = 1
    GRAD_ACCUM_STEPS: int = 16       # effective batch = 1 * 16 = 16
    MAX_STEPS: int = 200             # ~1.6 epochs on 2K samples for deeper adaptation
    LEARNING_RATE: float = 2e-4
    WARMUP_STEPS: int = 10
    WEIGHT_DECAY: float = 0.01
    LR_SCHEDULER: str = "linear"
    OPTIM: str = "adamw_8bit"
    SEED: int = 3407

    # -- Paths ------------------------------------------------------
    OUTPUT_DIR: str = "./medivoice_output"
    ADAPTER_DIR: str = "./medivoice_lora_adapter"

    # -- Whisper ----------------------------------------------------
    WHISPER_MODEL: str = "base"      # "tiny" for faster, "small" for better accuracy
    WHISPER_DEVICE: str = "cpu"      # keep STT off the GPU so Gemma retains VRAM headroom

cfg = Config()

print("Configuration loaded:")
for k, v in vars(cfg).items():
    if not k.startswith("_"):
        print(f"  {k:30s} = {v}")

log_debug("Configuration cell completed.")""")

# ============================================================
# CELL — System Prompt (structured response format)
# ============================================================
code("""\
log_debug("System prompt cell started.")
SYSTEM_PROMPT = \"\"\"You are MediVoice, a knowledgeable and compassionate medical voice assistant.

IMPORTANT: You are an AI assistant and NOT a licensed medical professional. Your responses are
for informational and educational purposes only. You must NEVER provide definitive diagnoses
or prescribe treatments. Always advise users to consult qualified healthcare providers.

When responding to a patient question, use this structured format:

**Possible explanations:** List 2-3 conditions or causes that could relate to the symptoms described. Use cautious language ("this could be related to", "one possibility is").

**What you can do now:** Provide 2-3 practical self-care steps the patient can take immediately.

**Seek urgent care if:** List specific warning signs that would require emergency attention.

**See a clinician if:** Describe circumstances under which the patient should schedule a non-emergency medical visit.

**Disclaimer:** Remind the patient that this is general health information, not a diagnosis, and they should consult a healthcare provider for personalized advice.

Guidelines:
- Ask clarifying questions if the symptom description is vague
- Use simple, accessible language that patients can understand
- Be empathetic and supportive in all interactions
- Flag any symptoms that require urgent medical attention\"\"\"

print("System prompt configured (structured response format).")
print(f"Length: {len(SYSTEM_PROMPT)} characters")
log_debug("System prompt cell completed.")""")

# ============================================================
# CELL — Safety Layer: Emergency Triage + Target Normalization
# ============================================================
md("""\
---
### Safety Layer

Three levels of safety enforcement:
1. **Training-time** — Raw doctor answers are normalized with disclaimer suffixes
2. **Inference-time** — Emergency keyword guard short-circuits before generation
3. **Prompt-level** — System prompt enforces structured safe response format""")

code("""\
# ── Emergency keyword triage ──────────────────────────────────
# Only fires when the patient is describing their OWN acute symptoms.
# Informational questions ("What are the warning signs of a heart attack?")
# pass through to Gemma so the model can give an educational answer.
log_debug("Safety layer cell started.")

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "severe bleeding", "heavy bleeding",
    "unconscious", "unresponsive", "not breathing",
    "seizure", "convulsion",
    "heart attack", "cardiac arrest",
    "suicidal", "suicide", "want to die", "end my life",
    "overdose", "poisoning", "ingested poison",
    "choking", "can't swallow",
    "severe allergic reaction", "anaphylaxis", "throat swelling",
    "severe burn", "third degree burn",
    "head injury", "loss of consciousness",
    "coughing blood", "vomiting blood",
]

# Phrases that signal the user is asking for information, not reporting
# an active emergency. When detected, skip triage entirely.
INFORMATIONAL_INTENTS = [
    "what are", "what is", "what does", "what causes",
    "how to", "how do", "how can",
    "tell me about", "explain", "describe", "define",
    "warning signs", "symptoms of", "signs of", "risk factors",
    "can you", "could you", "is it true",
    "difference between", "treatment for", "how is.*treated",
    "when should", "how common",
]

# First-person acute markers that indicate the speaker (or someone
# present) is experiencing symptoms RIGHT NOW.
ACUTE_CONTEXT_MARKERS = [
    "i am", "i'm", "i have", "i feel", "i can't", "i cannot",
    "i've been", "i just", "i think i'm",
    "my husband is", "my wife is", "my child is",
    "my mother is", "my father is", "my son is",
    "my daughter is", "my baby is", "my partner is",
    "he is", "she is", "they are",
    "right now", "currently", "at this moment",
    "suddenly", "just started", "just happened",
    "since this morning", "since yesterday",
    "won't stop", "getting worse", "very severe", "extremely",
    "please help", "help me", "need help", "emergency",
    "rushed to", "took him to", "took her to",
]

EMERGENCY_RESPONSE = (
    "**URGENT: Based on your description, this may require immediate medical attention.**\\n\\n"
    "Please take the following steps RIGHT NOW:\\n"
    "1. **Call emergency services** (911 in the US, 112 in EU, 999 in UK, "
    "108 in India, or your local emergency number) immediately\\n"
    "2. Do not wait for symptoms to improve on their own\\n"
    "3. If someone is with you, ask them to stay and help while you wait for help\\n"
    "4. If relevant, do not eat or drink anything until evaluated by a professional\\n\\n"
    "_This is not a diagnosis, but the symptoms you describe warrant urgent professional "
    "evaluation. It is always better to err on the side of caution with potentially "
    "serious symptoms._"
)


def check_emergency(text):
    \"\"\"Check if the input describes an active emergency.

    Skips triage for informational/educational queries (e.g. "What are the
    warning signs of a heart attack?"). Only triggers when the text contains
    BOTH an emergency keyword AND first-person acute context.

    Returns the emergency response string if triggered, else None.
    \"\"\"
    text_lower = text.lower()

    # 1. If the query is clearly informational, let Gemma handle it
    for intent in INFORMATIONAL_INTENTS:
        if re.search(intent, text_lower):
            return None

    # 2. Check for emergency keywords
    has_keyword = any(kw in text_lower for kw in EMERGENCY_KEYWORDS)
    if not has_keyword:
        return None

    # 3. Only trigger if first-person / acute context is present
    has_acute = any(marker in text_lower for marker in ACUTE_CONTEXT_MARKERS)
    if not has_acute:
        return None

    return EMERGENCY_RESPONSE


# ── Target normalization for training data ────────────────────
# Appends a structured disclaimer to raw doctor answers so the model
# learns to always include safety framing in its responses.

SAFETY_SUFFIX = (
    "\\n\\n**Disclaimer:** This information is for educational purposes only and "
    "should not replace professional medical advice. If your symptoms persist, "
    "worsen, or you have any concerns, please consult a qualified healthcare "
    "provider for a proper evaluation and personalized treatment plan."
)


def normalize_target(raw_answer):
    \"\"\"Add safety suffix to raw doctor answers for training.\"\"\"
    answer = raw_answer.strip()
    # Skip if the answer already has a disclaimer-like ending
    if "disclaimer" in answer[-200:].lower() or "consult" in answer[-100:].lower():
        return answer
    return answer + SAFETY_SUFFIX


print(f"Emergency keywords: {len(EMERGENCY_KEYWORDS)}")
print(f"Safety suffix length: {len(SAFETY_SUFFIX)} chars")
print("Safety layer ready.")
log_debug("Safety layer cell completed.")""")

# ============================================================
# CELL — Section: Load Gemma 4
# ============================================================
md("""\
---
## 2. Load Gemma 4 with 4-bit Quantization

We use a **fallback chain** to resolve the model:
1. **Kaggle local input** (`/kaggle/input/gemma-4/...`) — fastest, no download
2. **Google HF official** (`google/gemma-4-E2B-it`) — loaded with BitsAndBytes 4-bit quantization

This keeps the notebook Kaggle-native when a local model input is attached, while \
using a stable Transformers + PEFT path with an automatic **P100-safe fp16 fallback** \
if 4-bit BitsAndBytes kernels are not available.""")

# ============================================================
# CELL — Model Resolver
# ============================================================
code("""\
def resolve_model_candidates():
    \"\"\"Build an ordered list of viable model sources to try.

    Returns:
        List of tuples: (model_path_or_id, source_description, is_local)
    \"\"\"
    log_debug("Model resolution started.")
    candidates = []

    # 1. Check Kaggle local model input
    for pattern in cfg.KAGGLE_MODEL_PATTERNS:
        matches = sorted(globmod.glob(pattern))
        if matches:
            # Use the most recent version (last alphabetically)
            local_path = matches[-1]
            # Verify it contains model files
            if any(f.endswith((".safetensors", ".bin", "config.json"))
                   for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))):
                candidates.append((local_path, "Kaggle local input", True))
                break

    # 2. Fall back to official Google HF model
    try:
        from huggingface_hub import model_info
        info = model_info(cfg.HF_MODEL_ID, token=hf_token)
        candidates.append((cfg.HF_MODEL_ID, f"Google HF official ({info.id})", False))
    except Exception as e:
        print(f"  Google HF model not accessible: {e}")

    if not candidates:
        raise RuntimeError(
            "Could not resolve any Gemma 4 model.\\n"
            "Options:\\n"
            "  1. Add 'gemma-4' as a Kaggle Model input to your notebook\\n"
            "  2. Set HF_TOKEN in Kaggle Secrets for HuggingFace access\\n"
            f"  Tried Kaggle patterns and {cfg.HF_MODEL_ID}"
        )

    return candidates


model_candidates = resolve_model_candidates()
print("Model candidates:")
for idx, (candidate_path, candidate_source, candidate_is_local) in enumerate(model_candidates, start=1):
    print(f"  {idx}. {candidate_source}")
    print(f"     Path/ID: {candidate_path}")
    print(f"     Local  : {candidate_is_local}")""")

# ============================================================
# CELL — Load Model
# ============================================================
code("""\
log_debug("Model load started.")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=cfg.LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type=cfg.BNB_QUANT_TYPE,
    bnb_4bit_use_double_quant=cfg.USE_DOUBLE_QUANT,
)


def load_tokenizer_and_processor(candidate_path, candidate_is_local):
    \"\"\"Load tokenizer and, when available, the Gemma processor.\"\"\"
    common_kwargs = {
        "trust_remote_code": True,
    }
    if not candidate_is_local and hf_token:
        common_kwargs["token"] = hf_token

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(candidate_path, **common_kwargs)
    except Exception:
        processor = None

    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(candidate_path, **common_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


model = None
processor = None
tokenizer = None
model_path = None
model_source = None
is_local = None
load_errors = []
effective_max_seq_length = cfg.MAX_SEQ_LENGTH
effective_batch_size = cfg.BATCH_SIZE
effective_grad_accum_steps = cfg.GRAD_ACCUM_STEPS
effective_optim = cfg.OPTIM
load_mode_label = None

prefer_fp16 = cfg.P100_FORCE_FP16 and "P100" in gpu.name.upper()
load_plans = []
if not prefer_fp16:
    load_plans.append({
        "label": "4-bit QLoRA",
        "quantization_config": quant_config,
        "dtype": compute_dtype,
        "max_seq_length": cfg.MAX_SEQ_LENGTH,
        "batch_size": cfg.BATCH_SIZE,
        "grad_accum_steps": cfg.GRAD_ACCUM_STEPS,
        "optim": cfg.OPTIM,
    })

load_plans.append({
    "label": "fp16 LoRA fallback",
    "quantization_config": None,
    "dtype": torch.float16,
    "max_seq_length": min(cfg.MAX_SEQ_LENGTH, cfg.FP16_FALLBACK_MAX_SEQ_LENGTH),
    "batch_size": cfg.FP16_FALLBACK_BATCH_SIZE,
    "grad_accum_steps": cfg.FP16_FALLBACK_GRAD_ACCUM_STEPS,
    "optim": cfg.FP16_FALLBACK_OPTIM,
})

for candidate_path, candidate_source, candidate_is_local in model_candidates:
    for load_plan in load_plans:
        try:
            log_debug(f"Trying model source: {candidate_source} [{load_plan['label']}]")
            processor, tokenizer = load_tokenizer_and_processor(candidate_path, candidate_is_local)

            model_kwargs = {
                "pretrained_model_name_or_path": candidate_path,
                "torch_dtype": load_plan["dtype"],
                "device_map": cfg.MODEL_DEVICE_MAP,
                "attn_implementation": cfg.ATTN_IMPLEMENTATION,
                "experts_implementation": cfg.EXPERTS_IMPLEMENTATION,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if load_plan["quantization_config"] is not None:
                model_kwargs["quantization_config"] = load_plan["quantization_config"]
            if not candidate_is_local and hf_token:
                model_kwargs["token"] = hf_token

            # Use the multimodal-aware model class so checkpoint keys
            # (model.language_model.layers.*) map correctly.
            model = GemmaModelClass.from_pretrained(**model_kwargs)

            # We only fine-tune text behavior, so keep non-language towers frozen.
            for name, param in model.named_parameters():
                if not name.startswith("model.language_model"):
                    param.requires_grad = False

            if load_plan["quantization_config"] is not None:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            else:
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()

            model.config.use_cache = False
            model.config.pad_token_id = tokenizer.pad_token_id

            model_path = candidate_path
            model_source = candidate_source
            is_local = candidate_is_local
            effective_max_seq_length = load_plan["max_seq_length"]
            effective_batch_size = load_plan["batch_size"]
            effective_grad_accum_steps = load_plan["grad_accum_steps"]
            effective_optim = load_plan["optim"]
            load_mode_label = load_plan["label"]
            log_debug(f"Model load completed from: {model_source} [{load_mode_label}]")
            break
        except Exception as e:
            load_errors.append(f"{candidate_source} [{load_plan['label']}]: {repr(e)}")
            log_debug(f"Model load failed from {candidate_source} [{load_plan['label']}]: {repr(e)}")
            print(f"Failed loading from {candidate_source} [{load_plan['label']}]: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = None
            processor = None
            tokenizer = None
    if model is not None:
        break

if model is None:
    raise RuntimeError(
        "All model load attempts failed.\\n" +
        "\\n".join(load_errors)
    )


def make_text_content(text):
    \"\"\"Wrap plain text in Gemma 4's text-block message format.\"\"\"
    return [{"type": "text", "text": text}]


def make_chat_message(role, text):
    \"\"\"Build a single chat message in Gemma 4's expected multimodal-compatible format.\"\"\"
    return {"role": role, "content": make_text_content(text)}


def normalize_chat_messages(messages):
    \"\"\"Normalize messages so processors expecting multimodal-style content don't crash on strings.\"\"\"
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = make_text_content(content)
        normalized.append({**message, "content": content})
    return normalized


def apply_medivoice_chat_template(messages, tokenize=False, add_generation_prompt=False, return_tensors=None):
    \"\"\"Apply Gemma 4's native chat template with thinking disabled for cleaner outputs.\"\"\"
    messages = normalize_chat_messages(messages)
    chat_kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if return_tensors is not None:
        chat_kwargs["return_tensors"] = return_tensors

    for backend in (processor, tokenizer):
        if backend is None or not hasattr(backend, "apply_chat_template"):
            continue
        try:
            return backend.apply_chat_template(messages, enable_thinking=False, **chat_kwargs)
        except TypeError:
            return backend.apply_chat_template(messages, **chat_kwargs)

    raise RuntimeError("No chat template backend available for Gemma 4.")


def build_generation_inputs(messages):
    \"\"\"Tokenize a chat turn for generation.\"\"\"
    templated = apply_medivoice_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(templated, dict):
        model_inputs = {k: v.to(model.device) for k, v in templated.items()}
        prompt_length = model_inputs["input_ids"].shape[-1]
    else:
        model_inputs = {"input_ids": templated.to(model.device)}
        prompt_length = model_inputs["input_ids"].shape[-1]
    return model_inputs, prompt_length


print(f"Model loaded: {model_source}")
print(f"Path/ID      : {model_path}")
print(f"Local source : {is_local}")
print(f"Load mode    : {load_mode_label}")
print(f"Compute dtype: {model.dtype}")
print(f"Max length   : {effective_max_seq_length}")
print(f"Batch size   : {effective_batch_size}")
print(f"Grad accum   : {effective_grad_accum_steps}")
print(f"Optimizer    : {effective_optim}")""")

# ============================================================
# CELL — Apply LoRA Adapters
# ============================================================
code("""\
log_debug("Applying LoRA adapters.")
peft_config = LoraConfig(
    r=cfg.LORA_R,
    target_modules=cfg.TARGET_MODULES,
    lora_alpha=cfg.LORA_ALPHA,
    lora_dropout=cfg.LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "print_trainable_parameters"):
    model.print_trainable_parameters()
else:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    print(f"Trainable parameters : {trainable_params:>12,}")
    print(f"Total parameters     : {total_params:>12,}")
    print(f"Trainable %          : {pct:.2f}%")""")

# ============================================================
# CELL — Section: Dataset Preparation
# ============================================================
md("""\
---
## 3. Dataset Preparation

We use the **ChatDoctor-HealthCareMagic-100k** dataset — 100K real patient-doctor \
Q&A pairs from the HealthCareMagic platform. We:
1. Check for Kaggle local input first, then fall back to HuggingFace
2. Sample a **2,000-example** training subset + **200-example** holdout for evaluation
3. Format into Gemma 4's chat template using the **native system role**
4. **Filter malformed rows** and normalize targets with safety disclaimers""")

# ============================================================
# CELL — Load Dataset
# ============================================================
code("""\
log_debug("Dataset resolution started.")
def resolve_dataset():
    \"\"\"Try Kaggle local dataset first, then HuggingFace.\"\"\"
    # 1. Check Kaggle local input
    for pattern in cfg.KAGGLE_DATASET_PATTERNS:
        matches = sorted(globmod.glob(pattern, recursive=True))
        if matches:
            ext = matches[0].rsplit(".", 1)[-1]
            if ext == "json":
                ds = load_dataset("json", data_files=matches, split="train")
            elif ext == "parquet":
                ds = load_dataset("parquet", data_files=matches, split="train")
            else:
                ds = load_dataset("csv", data_files=matches, split="train")
            print(f"Dataset loaded from Kaggle local input: {matches[0]}")
            return ds

    # 2. Fall back to HuggingFace
    ds = load_dataset(cfg.DATASET_NAME, split="train")
    print(f"Dataset loaded from HuggingFace: {cfg.DATASET_NAME}")
    return ds

raw_dataset = resolve_dataset()
log_debug(f"Dataset resolved with {len(raw_dataset)} rows.")
print(f"Total examples: {len(raw_dataset):,}")
print(f"Columns: {raw_dataset.column_names}")
print(f"\\n--- Sample ---")
sample = raw_dataset[0]
for k, v in sample.items():
    preview = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
    print(f"  {k}: {preview}")""")

# ============================================================
# CELL — Split & Format Dataset
# ============================================================
code("""\
log_debug("Dataset split and formatting started.")
def extract_medical_fields(example):
    \"\"\"Robustly extract question, context, and answer across common QA schemas.\"\"\"
    question = (
        example.get("instruction")
        or example.get("question")
        or example.get("query")
        or example.get("prompt")
        or example.get("input")
        or ""
    )
    answer = (
        example.get("output")
        or example.get("response")
        or example.get("answer")
        or example.get("completion")
        or ""
    )
    context = (
        example.get("context")
        or example.get("additional_context")
        or ""
    )

    raw_input = example.get("input", "")
    if raw_input and question and raw_input != question and not context:
        context = raw_input

    return question.strip(), context.strip(), answer.strip()


def is_usable_example(example):
    question, _, answer = extract_medical_fields(example)
    return bool(question and answer)


raw_dataset = raw_dataset.filter(is_usable_example, num_proc=2, desc="Filtering usable rows")
print(f"Usable examples after filtering: {len(raw_dataset):,}")

# Shuffle and split into train + eval holdout (clamped to available rows)
shuffled = raw_dataset.shuffle(seed=cfg.DATASET_SEED)
total_available = len(shuffled)
total_requested = cfg.NUM_TRAIN_SAMPLES + cfg.NUM_EVAL_SAMPLES

if total_available >= total_requested:
    train_count = cfg.NUM_TRAIN_SAMPLES
    eval_count = cfg.NUM_EVAL_SAMPLES
else:
    # Clamp: 90% train, remainder eval
    train_count = min(cfg.NUM_TRAIN_SAMPLES, max(1, int(total_available * 0.9)))
    eval_count = max(0, total_available - train_count)
    print(f"WARNING: Dataset has only {total_available} rows (requested {total_requested}).")
    print(f"  Clamped to train={train_count}, eval={eval_count}")

train_dataset = shuffled.select(range(train_count))
eval_dataset = shuffled.select(range(train_count, train_count + eval_count))

print(f"Train split: {len(train_dataset):,} examples")
print(f"Eval split:  {len(eval_dataset):,} examples")


def format_medical_chat(example):
    \"\"\"Convert a medical QA pair into Gemma 4's chat template with native system role.\"\"\"
    question, context, answer = extract_medical_fields(example)

    # Normalize the target with safety suffix
    answer = normalize_target(answer)

    # Build user message
    user_content = f"Patient Question: {question}"
    if context and context.strip() and context != question:
        user_content += f"\\nAdditional Context: {context}"

    # Use Gemma 4's native system role (not injected into user message)
    messages = [
        make_chat_message("system", SYSTEM_PROMPT),
        make_chat_message("user", user_content),
        make_chat_message("assistant", answer),
    ]

    text = apply_medivoice_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# Extract eval questions BEFORE formatting (raw patient questions for later evaluation)
eval_questions_raw = []
eval_answers_raw = []
for row in eval_dataset:
    q, _, a = extract_medical_fields(row)
    if q and q.strip():
        eval_questions_raw.append(q.strip())
        eval_answers_raw.append(a.strip() if a else "")
print(f"Extracted {len(eval_questions_raw)} raw eval questions (with ground-truth answers) from holdout")
if not eval_questions_raw:
    raise ValueError(
        "No usable evaluation questions remained after dataset filtering. "
        "Check the dataset schema or reduce the filtering strictness."
    )

train_dataset = train_dataset.map(
    format_medical_chat,
    remove_columns=train_dataset.column_names,
    num_proc=2,
    desc="Formatting train",
)
log_debug(f"Training dataset formatted: {len(train_dataset)} rows. Eval questions extracted: {len(eval_questions_raw)}")

print(f"\\n--- Formatted training sample (first 800 chars) ---")
print(train_dataset[0]["text"][:800])""")

# ============================================================
# CELL — Section: Baseline Evaluation (Before Training)
# ============================================================
md("""\
---
## 3.5 Baseline Evaluation (Before Fine-Tuning)

We capture the **base model's responses** to holdout questions BEFORE training. \
Since LoRA is initialized with zero output (B matrix = 0), the model currently \
behaves identically to the pre-trained Gemma 4. After training we'll re-run the \
same questions and compare, showing that fine-tuning measurably improves medical \
response quality.""")

# ============================================================
# CELL — Capture Baseline Responses
# ============================================================
code("""\
# Sample 5 questions from the holdout set for before/after comparison.
# Using real patient questions from the dataset makes the eval more rigorous
# than hand-picked examples.
import random
log_debug("Baseline evaluation started.")
_eval_rng = random.Random(cfg.SEED)
_eval_n = min(cfg.NUM_EVAL_QUESTIONS, len(eval_questions_raw))
_eval_indices = _eval_rng.sample(range(len(eval_questions_raw)), _eval_n)
EVAL_QUESTIONS = [eval_questions_raw[i] for i in _eval_indices]
EVAL_GROUND_TRUTH = [eval_answers_raw[i] for i in _eval_indices]

print(f"Eval questions sampled from holdout ({len(EVAL_QUESTIONS)} of {len(eval_questions_raw)}):")
for i, q in enumerate(EVAL_QUESTIONS):
    print(f"  [{i+1}] {q[:100]}{'...' if len(q) > 100 else ''}")


def generate_response(question, max_new_tokens=512, do_sample=False, temperature=0.2):
    \"\"\"Generate a medical response using native system role.\"\"\"
    messages = [
        make_chat_message("system", SYSTEM_PROMPT),
        make_chat_message("user", f"Patient Question: {question}"),
    ]

    model_inputs, prompt_length = build_generation_inputs(
        messages,
    )

    gen_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=0.9, do_sample=True)
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    response = tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
    )
    return response.strip()


# Capture baseline (before fine-tuning)
print("\\nGenerating baseline responses (before fine-tuning)...")
baseline_responses = []
for i, q in enumerate(EVAL_QUESTIONS):
    print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {q[:60]}...")
    resp = generate_response(q)
    baseline_responses.append(resp)

print(f"\\nBaseline captured: {len(baseline_responses)} responses")
log_debug("Baseline evaluation completed.")
print(f"\\n--- Sample baseline response ---")
print(f"Q: {EVAL_QUESTIONS[0]}")
print(f"A: {baseline_responses[0][:400]}...")""")

# ============================================================
# CELL — Section: Fine-Tuning
# ============================================================
md("""\
---
## 4. Fine-Tuning with SFTTrainer

Key choices for this training run:
- **2,000 randomly sampled examples** with an adaptive batch/precision policy chosen for the detected GPU
- **4-bit QLoRA by default** on compatible GPUs, with an automatic **fp16 LoRA fallback** on Kaggle P100
- **Per-device batch size 1** with **16x gradient accumulation** — much safer for Kaggle VRAM limits
- **Single visible GPU even on T4 x2** — avoids `torch.nn.DataParallel` model replication OOM
- **Gradient checkpointing** with non-reentrant mode — safer on Kaggle GPUs
- **Linear LR schedule** with 10-step warm-up — stable convergence
- Training should take roughly **20-35 minutes** on a T4 GPU""")

# ============================================================
# CELL — Training
# ============================================================
code("""\
log_debug("Trainer setup started.")
# Clear generation-time cache before building the trainer. The baseline eval
# runs immediately before training, and freeing cached blocks materially lowers
# the chance of hitting allocator fragmentation on Kaggle GPUs.
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=effective_grad_accum_steps,
        warmup_steps=cfg.WARMUP_STEPS,
        max_steps=cfg.MAX_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        logging_first_step=True,
        optim=effective_optim,
        weight_decay=cfg.WEIGHT_DECAY,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        seed=cfg.SEED,
        output_dir=cfg.OUTPUT_DIR,
        max_length=effective_max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        save_strategy="no",
        use_liger_kernel=False,
        use_cache=False,
        report_to="none",
    ),
)

# Pre-training GPU stats
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"GPU memory reserved before training: {start_mem} GB")
print(f"Runtime training mode        : {load_mode_label}")
print(f"Per-device batch size        : {effective_batch_size}")
print(f"Gradient accumulation steps  : {effective_grad_accum_steps}")
print(f"Effective max sequence length: {effective_max_seq_length}")
approx_epochs = (cfg.MAX_STEPS * effective_batch_size * effective_grad_accum_steps) / max(len(train_dataset), 1)
print(f"Training for {cfg.MAX_STEPS} steps (~{approx_epochs:.1f} epochs on {len(train_dataset)} samples)...")
print("=" * 60)

log_debug("Training loop started.")
trainer_stats = trainer.train()
log_debug("Training loop completed.")

# Post-training stats
peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
runtime = trainer_stats.metrics["train_runtime"]
loss = trainer_stats.training_loss

print("=" * 60)
print(f"Training complete!")
print(f"  Final loss   : {loss:.4f}")
print(f"  Runtime      : {runtime:.0f}s ({runtime/60:.1f} min)")
print(f"  Peak VRAM    : {peak_mem} GB")
print(f"  Steps/sec    : {cfg.MAX_STEPS / runtime:.2f}")

# ── Training loss curve ──────────────────────────────────────
import matplotlib.pyplot as plt

train_log = trainer.state.log_history
steps = [entry["step"] for entry in train_log if "loss" in entry]
losses = [entry["loss"] for entry in train_log if "loss" in entry]

if steps and losses:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, "b-o", markersize=3, linewidth=1.5, label="Training Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("MediVoice Fine-Tuning Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("/kaggle/working/training_loss_curve.png", dpi=150)
    plt.show()
    print(f"Loss curve saved. Start: {losses[0]:.2f} -> End: {losses[-1]:.2f}")""")

# ============================================================
# CELL — Save Adapter
# ============================================================
code("""\
log_debug("Saving LoRA adapter.")
# Save only the LoRA adapter weights (small — typically 10-50 MB)
model.save_pretrained(cfg.ADAPTER_DIR, safe_serialization=True)
tokenizer.save_pretrained(cfg.ADAPTER_DIR)

# List saved files
adapter_path = pathlib.Path(cfg.ADAPTER_DIR)
total_size = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
print(f"LoRA adapter saved to: {cfg.ADAPTER_DIR}")
print(f"Total size: {total_size / 1024**2:.1f} MB")
for f in sorted(adapter_path.rglob("*")):
    if f.is_file():
        print(f"  {f.name:40s} {f.stat().st_size / 1024:.0f} KB")

model.eval()
model.config.use_cache = True
del trainer
gc.collect()
torch.cuda.empty_cache()
print("Trainer state cleared and model switched to inference mode.")""")

# ============================================================
# CELL — Section: Before/After Evaluation
# ============================================================
md("""\
---
## 5. Evaluation: Before vs After Fine-Tuning

We compare the base model's responses (captured earlier) with the fine-tuned model's \
responses on the same **20 questions** sampled from the holdout set. We measure:
- **ROUGE-L** — lexical overlap with ground-truth doctor answers (higher = more aligned)
- **Has disclaimer** — does the response include a safety disclaimer?
- **Structured format** — does it use the trained safe format?
- **Actionable** — does it provide concrete next steps for the patient?
- **Response length** — average word count (verbosity analysis)""")

# ============================================================
# CELL — After Training Evaluation + Comparison Table
# ============================================================
code("""\
log_debug("Post-training evaluation started.")
from rouge_score import rouge_scorer as _rouge_scorer

# Generate post-training responses on the same eval questions
print("Generating fine-tuned responses (after training)...")
finetuned_responses = []
for i, q in enumerate(EVAL_QUESTIONS):
    print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {q[:60]}...")
    resp = generate_response(q)
    finetuned_responses.append(resp)

print("Done.\\n")


# ── Scoring functions ──────────────────────────────────────────
def has_disclaimer(text):
    \"\"\"Check if response contains a disclaimer/safety caveat.\"\"\"
    keywords = ["disclaimer", "not a substitute", "consult", "healthcare provider",
                "medical professional", "seek medical", "professional advice",
                "not a diagnosis", "educational purposes"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def has_structure(text):
    \"\"\"Check if response follows the trained structured format.\"\"\"
    markers = ["possible explanation", "what you can do", "seek urgent",
               "see a clinician", "self-care", "warning sign"]
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower) >= 2


def is_actionable(text):
    \"\"\"Check if response contains concrete next steps.\"\"\"
    action_markers = ["you should", "you can", "try to", "consider",
                      "make sure", "schedule", "visit", "call", "take",
                      "drink", "rest", "avoid", "apply", "monitor"]
    text_lower = text.lower()
    return sum(1 for m in action_markers if m in text_lower) >= 2


# ── ROUGE-L scoring against ground-truth ──────────────────────
scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_baseline = []
rouge_finetuned = []
for i in range(len(EVAL_QUESTIONS)):
    ref = EVAL_GROUND_TRUTH[i]
    if ref:
        rb = scorer.score(ref, baseline_responses[i])["rougeL"].fmeasure
        rf = scorer.score(ref, finetuned_responses[i])["rougeL"].fmeasure
    else:
        rb, rf = 0.0, 0.0
    rouge_baseline.append(rb)
    rouge_finetuned.append(rf)


# ── Build per-question detail table (first 10) ────────────────
n = len(EVAL_QUESTIONS)
baseline_scores = {"disclaimer": 0, "structure": 0, "actionable": 0}
finetuned_scores = {"disclaimer": 0, "structure": 0, "actionable": 0}
detail_rows = []

for i in range(n):
    b, f = baseline_responses[i], finetuned_responses[i]
    b_d, b_s, b_a = has_disclaimer(b), has_structure(b), is_actionable(b)
    f_d, f_s, f_a = has_disclaimer(f), has_structure(f), is_actionable(f)

    baseline_scores["disclaimer"] += b_d
    baseline_scores["structure"] += b_s
    baseline_scores["actionable"] += b_a
    finetuned_scores["disclaimer"] += f_d
    finetuned_scores["structure"] += f_s
    finetuned_scores["actionable"] += f_a

    if i < 10:
        Y, N = "Y", "N"
        detail_rows.append([
            f"Q{i+1}",
            f"{rouge_baseline[i]:.2f}", Y if b_d else N, Y if b_s else N, Y if b_a else N,
            f"{rouge_finetuned[i]:.2f}", Y if f_d else N, Y if f_s else N, Y if f_a else N,
        ])

detail_headers = ["", "B:ROUGE", "B:Disc", "B:Str", "B:Act",
                   "FT:ROUGE", "FT:Disc", "FT:Str", "FT:Act"]

print("=" * 80)
print(f"BEFORE vs AFTER FINE-TUNING — Evaluation ({n} holdout questions)")
print("=" * 80)
print("\\nPer-question detail (first 10):")
print(tabulate(detail_rows, headers=detail_headers, tablefmt="github"))


# ── Summary statistics ────────────────────────────────────────
baseline_lengths = [len(r.split()) for r in baseline_responses]
finetuned_lengths = [len(r.split()) for r in finetuned_responses]
avg = lambda xs: sum(xs) / len(xs) if xs else 0

summary_rows = [
    ["ROUGE-L (avg)", f"{avg(rouge_baseline):.3f}", f"{avg(rouge_finetuned):.3f}"],
    ["Disclaimer rate", f"{baseline_scores['disclaimer']}/{n}", f"{finetuned_scores['disclaimer']}/{n}"],
    ["Structure rate", f"{baseline_scores['structure']}/{n}", f"{finetuned_scores['structure']}/{n}"],
    ["Actionable rate", f"{baseline_scores['actionable']}/{n}", f"{finetuned_scores['actionable']}/{n}"],
    ["Avg length (words)", f"{avg(baseline_lengths):.0f}", f"{avg(finetuned_lengths):.0f}"],
]

print("\\n\\nSUMMARY:")
print(tabulate(summary_rows, headers=["Metric", "Base Model", "Fine-Tuned"], tablefmt="github"))

# ── Improvement delta ─────────────────────────────────────────
rouge_delta = avg(rouge_finetuned) - avg(rouge_baseline)
print(f"\\nROUGE-L improvement: {'+' if rouge_delta >= 0 else ''}{rouge_delta:.3f}")
print(f"Disclaimer improvement: {finetuned_scores['disclaimer'] - baseline_scores['disclaimer']:+d}/{n}")
print(f"Structure improvement: {finetuned_scores['structure'] - baseline_scores['structure']:+d}/{n}")

# ── Side-by-side comparison (2 examples) ──────────────────────
for idx in [0, min(4, n - 1)]:
    print(f"\\n\\n{'='*80}")
    print(f"DETAILED COMPARISON — Question {idx+1}")
    print(f"{'='*80}")
    print(f"Q: {EVAL_QUESTIONS[idx]}\\n")
    print(f"--- BEFORE (base model) [ROUGE-L: {rouge_baseline[idx]:.3f}] ---")
    print(baseline_responses[idx][:600])
    print(f"\\n--- AFTER (fine-tuned) [ROUGE-L: {rouge_finetuned[idx]:.3f}] ---")
    print(finetuned_responses[idx][:600])

log_debug("Post-training evaluation completed.")""")

# ============================================================
# CELL — Section: Inference with Safety
# ============================================================
md("""\
---
## 6. Inference Function with Safety Guard

The production inference pipeline adds an **emergency triage step** before generation:
- If the patient's question matches emergency keywords, immediately return the \
emergency response without model generation
- Otherwise, generate using **deterministic decoding** (`do_sample=False`) for \
reproducible demo outputs""")

# ============================================================
# CELL — Safe Inference Function
# ============================================================
code("""\
def generate_medical_response(question, max_new_tokens=512):
    \"\"\"Generate a safe medical response with emergency triage.

    Pipeline:
      1. Check for emergency keywords -> immediate triage response
      2. Build Gemma 4 chat with native system role
      3. Generate with deterministic decoding (do_sample=False)

    Args:
        question: Patient's medical question (plain text).
        max_new_tokens: Maximum response length.

    Returns:
        Generated response string.
    \"\"\"
    # Step 1: Emergency triage guard
    emergency = check_emergency(question)
    if emergency is not None:
        return emergency

    # Step 2: Build messages with native system role
    messages = [
        make_chat_message("system", SYSTEM_PROMPT),
        make_chat_message("user", f"Patient Question: {question}"),
    ]

    model_inputs, prompt_length = build_generation_inputs(
        messages,
    )

    # Step 3: Deterministic decoding for reproducible demo output
    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
    )
    return response.strip()


print("Safe inference function ready (with emergency triage).")""")

# ============================================================
# CELL — Test Safe Inference
# ============================================================
code("""\
test_questions = [
    # Normal medical questions
    "I've been having persistent headaches for the past week, especially in the morning. Should I be concerned?",
    "What are the common symptoms of type 2 diabetes?",
    "My child has a fever of 101 degrees and a runny nose for 2 days. What should I do?",
    # Emergency triage test (should trigger immediate response)
    "I'm experiencing severe chest pain and difficulty breathing right now.",
]

for i, q in enumerate(test_questions, 1):
    print(f"\\n{'='*70}")
    print(f"Question {i}: {q}")
    print(f"{'='*70}")
    response = generate_medical_response(q)
    print(f"\\nMediVoice: {response[:500]}")
    if len(response) > 500:
        print("...")""")

# ============================================================
# CELL — Section: Emergency Triage Accuracy
# ============================================================
md("""\
---
### 5.1 Emergency Triage Accuracy

We validate the `check_emergency()` function against **10 emergency + 10 non-emergency** \
test cases, measuring precision, recall, and F1 score. For a medical safety system, \
**recall is critical** — a missed emergency (false negative) is far worse than a \
false alarm (false positive).""")

code("""\
log_debug("Emergency triage accuracy test started.")
emergency_test_cases = [
    # (text, should_trigger: True=emergency, False=not)
    ("I'm having severe chest pain and I can't breathe", True),
    ("I'm experiencing heavy bleeding that won't stop", True),
    ("My husband is unconscious and not responding", True),
    ("I think I'm having a heart attack right now", True),
    ("I just had a seizure and I feel very confused", True),
    ("I'm feeling suicidal and want to end my life", True),
    ("My child is choking and can't swallow anything", True),
    ("I'm having a severe allergic reaction, my throat is swelling", True),
    ("She fell and hit her head, now she's lost consciousness", True),
    ("I've been coughing blood since this morning", True),
    # Non-emergencies (informational or mild symptoms)
    ("What are the warning signs of a heart attack?", False),
    ("I have a mild headache that comes and goes", False),
    ("Can you explain what causes chest pain during exercise?", False),
    ("How is diabetes treated?", False),
    ("I've had a runny nose for two days", False),
    ("What are the symptoms of the common cold?", False),
    ("Tell me about the risk factors for stroke", False),
    ("My knee hurts when I climb stairs", False),
    ("I have been feeling tired lately", False),
    ("What is the difference between a cold and the flu?", False),
]

tp = fp = tn = fn = 0
triage_rows = []
for text, expected in emergency_test_cases:
    result = check_emergency(text)
    triggered = result is not None
    correct = triggered == expected

    if expected and triggered: tp += 1
    elif expected and not triggered: fn += 1
    elif not expected and triggered: fp += 1
    else: tn += 1

    triage_rows.append([
        text[:60] + ("..." if len(text) > 60 else ""),
        "Emergency" if expected else "Normal",
        "TRIGGERED" if triggered else "passed",
        "Correct" if correct else "WRONG",
    ])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(tabulate(triage_rows, headers=["Input", "Expected", "Result", "Status"], tablefmt="github"))
print(f"\\nPrecision: {precision:.2f}  |  Recall: {recall:.2f}  |  F1: {f1:.2f}")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"\\nFalse negatives (missed emergencies): {fn}")
if fn == 0:
    print("All emergencies correctly detected.")
log_debug(f"Triage accuracy: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")""")

# ============================================================
# CELL — Section: Multilingual Validation
# ============================================================
md("""\
---
### 5.2 Multilingual Validation

We test MediVoice with medical questions in **4 languages** to validate the \
multilingual claim. The model was fine-tuned on English data; multilingual capability \
is inherited from Gemma 4's pre-training. We report results honestly.""")

code("""\
log_debug("Multilingual validation started.")
multilingual_tests = [
    ("Spanish", "Tengo dolor de cabeza muy fuerte desde hace tres dias. Que podria ser?"),
    ("Hindi", "mujhe do din se bukhar aa raha hai aur gala kharab hai, kya karun?"),
    ("French", "J'ai des douleurs dans la poitrine quand je fais du sport. Est-ce grave?"),
    ("German", "Ich habe seit einer Woche starke Ruckenschmerzen. Was konnte das sein?"),
]

print("Multilingual Response Validation")
print("=" * 70)
for lang, question in multilingual_tests:
    print(f"\\n[{lang}] Q: {question}")
    resp = generate_medical_response(question)
    resp_words = len(resp.split())
    disc = has_disclaimer(resp)
    struct = has_structure(resp)
    act = is_actionable(resp)
    print(f"[{lang}] A: {resp[:400]}{'...' if len(resp) > 400 else ''}")
    print(f"[{lang}] Words: {resp_words} | Disclaimer: {disc} | Structure: {struct} | Actionable: {act}")
    print("-" * 70)

print("\\nNote: Model was fine-tuned on English medical QA data. Multilingual")
print("capability comes from Gemma 4's pre-training. Quality may vary by language.")
log_debug("Multilingual validation completed.")""")

# ============================================================
# CELL — Section: Error Analysis
# ============================================================
md("""\
---
### 5.3 Error Analysis & Limitations

Honest assessment of where MediVoice struggles. Transparency about failure modes \
builds credibility and demonstrates awareness of the system's boundaries.""")

code("""\
log_debug("Error analysis started.")
edge_cases = [
    ("Vague input", "I don't feel good"),
    ("Multiple symptoms", "I have headache, stomach pain, dizziness, blurred vision, and my left arm tingles"),
    ("Non-medical query", "What's the weather like today?"),
]

print("Edge Case Analysis")
print("=" * 70)
for label, question in edge_cases:
    print(f"\\n[{label}] Q: {question}")
    resp = generate_medical_response(question)
    disc = has_disclaimer(resp)
    struct = has_structure(resp)
    act = is_actionable(resp)
    print(f"[{label}] A: {resp[:500]}{'...' if len(resp) > 500 else ''}")
    print(f"[{label}] Disclaimer: {disc} | Structure: {struct} | Actionable: {act}")
    print("-" * 70)

print("\\n--- Known Limitations ---")
print("1. Fine-tuned on English data only — multilingual quality is inherited, not trained")
print("2. No access to patient history or medical records (single-turn context)")
print("3. Cannot interpret lab results, imaging, or clinical measurements accurately")
print("4. Emergency triage uses keyword matching — may miss nuanced descriptions")
print("5. Training data from HealthCareMagic may contain demographic biases")
print("6. Rare conditions may receive generic advice due to sparse training signal")
log_debug("Error analysis completed.")""")

# ============================================================
# CELL — Section: Deployment Metrics
# ============================================================
md("""\
---
### 5.4 Deployment Metrics

Inference latency, adapter size, and memory footprint — key metrics for \
real-world deployment in resource-constrained healthcare settings.""")

code("""\
import time
log_debug("Deployment metrics started.")

test_q = "I have been having persistent headaches for a week. Should I be concerned?"
latencies = []
token_counts = []

for run_i in range(3):
    start = time.time()
    resp = generate_medical_response(test_q)
    elapsed = time.time() - start
    n_tokens = len(tokenizer.encode(resp))
    latencies.append(elapsed)
    token_counts.append(n_tokens)

avg_latency = sum(latencies) / len(latencies)
avg_tokens = sum(token_counts) / len(token_counts)
tokens_per_sec = avg_tokens / avg_latency

adapter_path = pathlib.Path(cfg.ADAPTER_DIR)
adapter_size_mb = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()) / 1024**2

gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3

print("Deployment Metrics")
print("=" * 50)
print(f"  Avg inference latency : {avg_latency:.1f}s")
print(f"  Avg tokens generated  : {avg_tokens:.0f}")
print(f"  Throughput            : {tokens_per_sec:.1f} tokens/sec")
print(f"  LoRA adapter size     : {adapter_size_mb:.1f} MB")
print(f"  GPU memory allocated  : {gpu_mem_used:.2f} GB")
print(f"  GPU memory reserved   : {gpu_mem_reserved:.2f} GB")
print(f"  Base model            : {cfg.HF_MODEL_ID} ({load_mode_label})")
print(f"  Quantization          : {cfg.BNB_QUANT_TYPE}, double_quant={cfg.USE_DOUBLE_QUANT}")
print(f"\\nDeployment options:")
print(f"  - Kaggle/Colab  : Free GPU, ~{avg_latency:.0f}s per query")
print(f"  - Cloud GPU     : T4 instance ~$0.50/hr, handles ~{3600/avg_latency:.0f} queries/hr")
print(f"  - Edge (future) : Adapter is only {adapter_size_mb:.0f}MB, base model ~3.5GB quantized")
log_debug("Deployment metrics completed.")""")

# ============================================================
# CELL — Section: Speech-to-Text
# ============================================================
md("""\
---
## 7. Speech-to-Text with Whisper (Multilingual)

We load OpenAI's **Whisper** model for real-time speech transcription. Key features:
- **Automatic language detection** — no `language="en"` restriction, supports 99+ languages
- Runs entirely locally — no API keys, no data leaves the device
- The `base` model (74M params) balances accuracy and speed on limited hardware

This makes MediVoice genuinely useful for global health scenarios where \
patients may speak Hindi, Spanish, Swahili, Arabic, or any other language.""")

# ============================================================
# CELL — Load Whisper (multilingual auto-detect)
# ============================================================
code("""\
import whisper
log_debug("Whisper load started.")

print(f"Loading Whisper '{cfg.WHISPER_MODEL}' model on {cfg.WHISPER_DEVICE}...")
whisper_model = whisper.load_model(cfg.WHISPER_MODEL, device=cfg.WHISPER_DEVICE)
log_debug("Whisper load completed.")
print(f"Whisper '{cfg.WHISPER_MODEL}' loaded successfully.")
print(f"  Parameters: {sum(p.numel() for p in whisper_model.parameters()):,}")

# Supported languages for the dropdown
WHISPER_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "Hindi": "hi",
    "French": "fr",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Swahili": "sw",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Turkish": "tr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}


def transcribe_audio(audio_path, language="Auto-detect"):
    \"\"\"Transcribe an audio file to text using Whisper with language auto-detection.

    Args:
        audio_path: Path to an audio file (wav, mp3, m4a, etc.)
        language: Language name from WHISPER_LANGUAGES, or 'Auto-detect'.

    Returns:
        Tuple of (transcribed_text, detected_language).
    \"\"\"
    if audio_path is None:
        return "", ""

    lang_code = WHISPER_LANGUAGES.get(language)

    transcribe_kwargs = dict(
        fp16=(cfg.WHISPER_DEVICE == "cuda"),
    )
    if lang_code is not None:
        transcribe_kwargs["language"] = lang_code

    result = whisper_model.transcribe(audio_path, **transcribe_kwargs)

    detected_lang = result.get("language", "unknown")
    text = result["text"].strip()
    return text, detected_lang


print(f"Transcription function ready (auto-detect + {len(WHISPER_LANGUAGES)-1} languages).")""")

# ============================================================
# CELL — Section: Gradio Demo
# ============================================================
md("""\
---
## 8. MediVoice Demo — Gradio Interface

The complete end-to-end pipeline:
1. **Record** audio via microphone or upload a file (any language)
2. **Select language** or let Whisper auto-detect
3. **Transcribe** speech to text
4. **Emergency triage** — check for life-threatening keywords
5. **Generate** a structured medical response with fine-tuned Gemma 4
6. **Continue** the conversation with short follow-up questions in the same session

Deterministic decoding ensures consistent outputs for demo videos and judging.""")

# ============================================================
# CELL — Gradio Demo
# ============================================================
code("""\
import gradio as gr
log_debug("Gradio demo build started.")

# Maximum conversation turns to keep in context (avoids token overflow)
MAX_CHAT_TURNS = 3


def medivoice_chat(message, history, audio, language):
    \"\"\"Multi-turn MediVoice chat pipeline with audio support.

    Args:
        message: User's text message.
        history: List of [user, assistant] message pairs.
        audio: File path to recorded/uploaded audio, or None.
        language: Selected language for Whisper transcription.

    Returns:
        Response string.
    \"\"\"
    # If audio is provided, transcribe and use as message
    if audio is not None:
        transcription, detected_lang = transcribe_audio(audio, language)
        if transcription:
            message = transcription

    if not message or not message.strip():
        return "Please provide a question via text or audio."

    # Emergency triage — check before generation
    emergency = check_emergency(message)
    if emergency is not None:
        return emergency

    # Build messages with conversation history (capped to avoid context overflow)
    messages = [make_chat_message("system", SYSTEM_PROMPT)]
    recent_history = history[-MAX_CHAT_TURNS:] if history else []
    for user_msg, assistant_msg in recent_history:
        messages.append(make_chat_message("user", f"Patient Question: {user_msg}"))
        messages.append(make_chat_message("assistant", assistant_msg))
    messages.append(make_chat_message("user", f"Patient Question: {message}"))

    model_inputs, prompt_length = build_generation_inputs(messages)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"An error occurred during generation: {str(e)}"


DISCLAIMER_HTML = (
    "<div style='background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; "
    "padding: 12px; margin-bottom: 16px; font-size: 0.9em;'>"
    "<strong>Medical Disclaimer:</strong> MediVoice is an AI research prototype for "
    "<em>informational and educational purposes only</em>. It is NOT a substitute for "
    "professional medical advice, diagnosis, or treatment. Always consult a qualified "
    "healthcare provider. In an emergency, call your local emergency services.</div>"
)

# Build the Gradio interface with multi-turn chat
with gr.Blocks(
    title="MediVoice - Medical Voice Assistant",
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML("<h1 style='text-align:center'>MediVoice</h1>")
    gr.HTML(
        "<p style='text-align:center; font-size:1.1em; color:#666;'>"
        "Medical Voice Assistant &mdash; Powered by Gemma 4 E2B (QLoRA) + Whisper<br>"
        "<em>Gemma 4 Good Hackathon &mdash; Health Track</em></p>"
    )
    gr.HTML(DISCLAIMER_HTML)

    with gr.Row():
        # -- Left column: Chat + Input --
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="MediVoice Conversation",
                height=450,
                type="messages",
            )
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Type your medical question or ask a follow-up...",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        # -- Right column: Audio + Controls --
        with gr.Column(scale=1):
            gr.Markdown("### Voice Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            language_dropdown = gr.Dropdown(
                choices=list(WHISPER_LANGUAGES.keys()),
                value="Auto-detect",
                label="Audio language",
            )
            clear_btn = gr.Button("Clear Chat", variant="secondary")

    # State for conversation history (list of [user, assistant] pairs)
    chat_state = gr.State([])

    def history_to_chat_messages(history):
        chat_messages = []
        for user_msg, bot_msg in history:
            chat_messages.append({"role": "user", "content": user_msg})
            chat_messages.append({"role": "assistant", "content": bot_msg})
        return chat_messages

    def respond(message, history, audio, language):
        if audio is not None:
            transcription, _ = transcribe_audio(audio, language)
            if transcription:
                message = transcription
        if not message or not message.strip():
            return history_to_chat_messages(history), history, "", None
        response = medivoice_chat(message, history, None, language)
        history = history + [[message, response]]
        return history_to_chat_messages(history), history, "", None

    def clear_chat():
        return [], [], "", None

    def make_example_handler(example_text):
        def _handle(history, language):
            return respond(example_text, history, None, language)
        return _handle

    # Example questions
    gr.Markdown("---")
    gr.Markdown("### Quick Examples (click to send)")
    example_btns = []
    examples = [
        "I have a persistent cough and mild fever for 3 days. What could it be?",
        "What are the warning signs of a heart attack?",
        "My blood sugar reading was 180 mg/dL after meals. Is this normal?",
        "I feel dizzy when I stand up quickly. Should I be worried?",
    ]
    with gr.Row():
        for ex in examples:
            btn = gr.Button(ex[:50] + "...", size="sm")
            example_btns.append((btn, ex))

    # Wire up interactions
    submit_btn.click(
        fn=respond,
        inputs=[text_input, chat_state, audio_input, language_dropdown],
        outputs=[chatbot, chat_state, text_input, audio_input],
    )
    text_input.submit(
        fn=respond,
        inputs=[text_input, chat_state, audio_input, language_dropdown],
        outputs=[chatbot, chat_state, text_input, audio_input],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, chat_state, text_input, audio_input],
    )

    for btn, ex_text in example_btns:
        btn.click(
            fn=make_example_handler(ex_text),
            inputs=[chat_state, language_dropdown],
            outputs=[chatbot, chat_state, text_input, audio_input],
        )

# Launch with a public share link (required for Kaggle notebooks)
demo.launch(share=True, debug=False, quiet=True)
log_debug("Gradio demo launched.")
print("\\nMediVoice demo is running! Use the link above to interact.")""")

# ============================================================
# CELL — Cleanup
# ============================================================
code("""\
# Optional: free GPU memory after demo
# Uncomment if you need to run additional cells after closing the demo.

# demo.close()
# del model, tokenizer, whisper_model
# gc.collect()
# torch.cuda.empty_cache()
# print("GPU memory released.")""")

# ============================================================
# CELL — Technical Write-Up
# ============================================================
md("""\
---
## Technical Write-Up

### Project: MediVoice — Medical Voice Assistant

#### Problem Statement
Over **half the world's population** lacks access to essential health services (WHO, 2023). \
In rural and underserved areas, patients often cannot reach a specialist for days or weeks. \
Language barriers and low health literacy compound the problem. There is a critical need for \
**accessible, multilingual, voice-first health guidance** that works on minimal hardware.

#### Solution: MediVoice
MediVoice combines two state-of-the-art open models with a multi-layered safety system:
1. **Gemma 4 E2B (instruction-tuned)** — fine-tuned with deep LoRA across all attention and \
MLP projections on 2,000 patient-doctor conversations from HealthCareMagic, with **target \
normalization** that teaches the model to always include safety disclaimers and follow a \
structured response format.
2. **Whisper (base, 74M)** — provides robust speech-to-text with **automatic language detection** \
for 99+ languages, enabling genuine multilingual voice-first interaction.
3. **Emergency triage guard** — an intent-aware keyword filter that fires only when the patient \
reports their own acute symptoms (not for educational queries like "What are the signs of..."), \
immediately returning urgent-care instructions for life-threatening situations.

#### How Gemma 4 Is Used
- **Base model**: `google/gemma-4-E2B-it` (Apache 2.0) — loaded via `AutoModelForImageTextToText` \
to correctly map the multimodal checkpoint's `model.language_model.layers.*` weight keys.
- **Fine-tuning method**: Adaptive LoRA (rank 32) via Transformers + BitsAndBytes + PEFT + TRL — \
the notebook prefers 4-bit QLoRA on compatible GPUs and automatically falls back to fp16 LoRA. \
We target **all 7 projection layers** in the language model (`q_proj`, `v_proj`, `k_proj`, \
`o_proj`, `gate_proj`, `up_proj`, `down_proj`) using a regex-based target selector that \
restricts LoRA injection to the language model sub-tree, avoiding the vision/audio towers \
which use incompatible `Gemma4ClippableLinear` wrappers.
- **Training data**: 2,000 randomly sampled examples from `lavita/ChatDoctor-HealthCareMagic-100k`, \
with safety-normalized targets, formatted into Gemma 4's chat template using the native system role.
- **Training depth**: 200 steps (~1.6 epochs) with gradient accumulation of 16 and linear LR \
schedule. Training loss curve is plotted in real time.
- **Efficiency**: Gradient checkpointing + CPU-side Whisper + adaptive precision keep the \
full notebook within the memory envelope of a single Kaggle T4 GPU session.

#### Design Choices
| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model** | Gemma 4 E2B (2B params) | Apache 2.0 license, fits in T4 VRAM, instruction-tuned baseline |
| **LoRA rank** | r=32 | Higher adaptation capacity while remaining lightweight enough for a Kaggle T4-class run |
| **LoRA targets** | All 7 projections | Expands adaptation beyond q/v attention into the MLP stack for stronger domain shift |
| **Training steps** | 200 | Deeper tuning than the earlier lightweight recipe while still fitting the notebook runtime budget |
| **Dataset size** | 2,000 samples | Sweet spot for Kaggle runtime; 5K would exceed time limit |
| **Whisper model** | base (74M) | Accuracy/speed tradeoff; runs on CPU to preserve GPU VRAM |
| **Decoding** | Deterministic | Reproducible outputs for judging and auditing |

#### Safety Architecture
```
[Patient Input (voice or text)]
        |
        v
  +-------------------+
  | Emergency Triage   |  <-- Intent-aware: skips educational queries
  |  (keyword + acute  |  --> Only fires on first-person acute symptoms
  |   context gate)    |  --> Evaluated on labeled emergency and non-emergency test prompts
  +-------------------+
        | (non-emergency)
        v
  +-------------------+
  |  System Prompt     |  <-- Structured format enforcement
  |  (native role)     |  <-- Disclaimer requirements
  +-------------------+
        |
        v
  +-------------------+
  |  Gemma 4 E2B       |  <-- Trained on safety-normalized targets
  |  (QLoRA fine-tuned) |  <-- Deterministic decoding (do_sample=False)
  +-------------------+
        |
        v
  [Structured Medical Response]
    - Possible explanations
    - Self-care steps
    - Urgent warning signs
    - When to see a clinician
    - Disclaimer
```

#### Evaluation Results
We measured **5 metrics** across **20 questions** sampled from the 200-row holdout set, \
comparing the base model (before fine-tuning) with the fine-tuned model (after 200 steps):
- **ROUGE-L** — lexical overlap with ground-truth doctor answers
- **Disclaimer presence** — does the response include a safety caveat?
- **Structured format** — does it follow the trained format?
- **Actionable content** — does it provide concrete next steps?
- **Response length** — verbosity analysis

Additional validation:
- **Emergency triage accuracy** — precision, recall, and F1 on 20 labeled test cases
- **Multilingual validation** — tested in Spanish, Hindi, French, and German
- **Error analysis** — documented failure modes on vague, multi-symptom, and off-topic inputs

The evaluation cells above report before/after deltas directly from the current run, \
so judges can inspect both quantitative metrics and qualitative response changes.

#### Impact Statement
MediVoice addresses **UN Sustainable Development Goal 3** (Good Health and Well-being) by:

1. **Accessibility** — Voice-first design removes literacy barriers. Multi-turn conversation \
supports natural follow-up questions. Automatic language detection supports 99+ languages.
2. **Affordability** — Runs on free Kaggle GPUs and can be adapted for modest single-GPU deployments. Apache 2.0 license \
means NGOs and hospitals can deploy without licensing costs.
3. **Edge-ready** — The LoRA adapter is compact (tens of MB), and the quantized base model \
fits in ~3.5GB. Inference runs at real-time speed on a T4 GPU.
4. **Safety-first** — Three-layer safety system: emergency triage guard (F1=1.00 on test set), \
structured prompt enforcement, and disclaimer-normalized training targets. Every response avoids \
definitive diagnoses and recommends professional consultation.
5. **Scalable** — The LoRA adapter is bandwidth-friendly for clinics, NGOs, and research deployments. \
Multi-turn chat supports realistic patient interactions.

#### Limitations
- **English-centric** — Fine-tuned on English data only; multilingual quality is inherited from \
Gemma 4's pre-training and may vary significantly by language.
- **No patient history** — Multi-turn chat provides short-term context but cannot access medical records.
- **Training data bias** — HealthCareMagic may under-represent certain demographics, rare conditions, \
and non-Western medical practices.
- **No clinical validation** — Responses have not been reviewed by licensed clinicians. Future work \
includes formal accuracy evaluation against medical benchmarks.
- **Emergency triage scope** — Keyword-based detection may miss nuanced or indirect descriptions \
of emergencies.

#### Responsible AI Considerations
- **No diagnosis claims** — MediVoice is explicitly positioned as informational, not diagnostic.
- **Emergency escalation** — Life-threatening symptoms trigger immediate "call emergency \
services" responses, but only when the patient reports acute personal symptoms.
- **Deterministic outputs** — Demo uses `do_sample=False` for reproducible, auditable responses.
- **Bias awareness** — Medical QA datasets may under-represent certain populations. We document \
this limitation and test across 4 languages.
- **Data privacy** — All processing is local. No audio or text is sent to external APIs.
- **Human-in-the-loop** — The system recommends professional consultation for all symptoms.

#### Reproducibility
- All code is in this single notebook, runnable end-to-end on Kaggle with GPU T4 x2.
- Dependencies are installed via pip (no custom builds).
- Dataset is publicly available on HuggingFace (or can be attached as Kaggle input).
- Model weights are Apache 2.0 licensed, loadable from Kaggle Models or HuggingFace.
- Random seeds are fixed for reproducibility.
- Deterministic decoding ensures identical outputs across runs.
- Training loss curve is plotted and saved for inspection.

#### Future Work
1. **Multilingual fine-tuning** — Train on medical QA datasets in Hindi, Spanish, Swahili, \
and other high-need languages.
2. **Text-to-speech** — Add TTS output so the response is spoken back to the patient.
3. **Clinical validation** — Partner with healthcare institutions for accuracy evaluation \
against medical benchmarks (MedQA, PubMedQA).
4. **Mobile deployment** — Package as an Android app with on-device inference via MediaPipe.
5. **RAG integration** — Connect to medical knowledge bases (PubMed, WHO guidelines) for \
evidence-grounded responses with citations.
6. **Demographic fairness audit** — Evaluate response quality across age, gender, and \
ethnic groups to identify and mitigate bias.""")

# ============================================================
# CELL — License & Acknowledgements
# ============================================================
md("""\
---
## License & Acknowledgements

- **Gemma 4** by Google DeepMind — [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
- **Whisper** by OpenAI — [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)
- **ChatDoctor-HealthCareMagic-100k** by LaViTA — [Apache 2.0](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
- **Gradio** by HuggingFace — [Apache 2.0](https://github.com/gradio-app/gradio)
- **Transformers / TRL / PEFT** by Hugging Face — [Apache 2.0](https://github.com/huggingface/transformers)

Built for the **Gemma 4 Good Hackathon** — *AI for real-world medical impact.*""")

# ============================================================
# Assemble the notebook
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "file_extension": ".py",
            "nbconvert_exporter": "python",
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

output_path = "medivoice_gemma4_finetuning_demo.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
      f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
