#!/usr/bin/env python3
"""
SimpleQA quantization experiment: Gemma 4 12B, BF16 vs FP8 vs FP4,
with baseline vs uncertainty-aware prompting.

Designed for a CUDA machine, including Google Colab G4. Checkpoints and
reports are written to OUTPUT_DIR. Re-run the script to resume.

Install dependencies automatically at startup. You may also install them
manually with:
  pip install "transformers>=5.10.1" datasets accelerate openai huggingface_hub \
      matplotlib numpy pandas scipy packaging "bitsandbytes>=0.46.1"
"""

# =============================================================================
# 0. INSTALL / IMPORTS
# =============================================================================
import os
import sys
import subprocess
from importlib import metadata

REQUIRED = [
    ("transformers", "5.10.1"),
    ("datasets", None),
    ("accelerate", None),
    ("openai", None),
    ("huggingface_hub", None),
    ("matplotlib", None),
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("packaging", None),
    ("bitsandbytes", "0.46.1"),
]


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


try:
    from packaging.version import Version
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "packaging"])
    from packaging.version import Version

to_install = []
for dist, minimum in REQUIRED:
    version = installed_version(dist)
    if version is None:
        to_install.append(dist if minimum is None else f"{dist}>={minimum}")
    elif minimum and Version(version) < Version(minimum):
        to_install.append(f"{dist}>={minimum}")

if to_install:
    print("Installing dependencies:", to_install, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    print("If the runtime requests a restart, restart it and run this script again.")

import gc
import re
import json
import time
import math
import random
import hashlib
import signal
import threading
import datetime
import warnings
from pathlib import Path
from getpass import getpass
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import transformers
from datasets import load_dataset
from huggingface_hub import login
from openai import OpenAI, RateLimitError
from transformers import AutoProcessor, AutoModelForMultimodalLM

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL_ID = "google/gemma-4-12B-it"
DATASET_ID = "OpenEvals/SimpleQA"
DATASET_SPLIT = "test"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

MODEL_DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = False

# Leave checkpoint IDs as None to quantize BASE_MODEL_ID at load time.
# Pre-quantized checkpoints are supported only if Transformers can load them.
FP8_CHECKPOINT_ID = None
FP4_CHECKPOINT_ID = None

VARIANTS = ["bf16", "fp8", "fp4"]
VARIANT_LABELS = {"bf16": "BF16", "fp8": "FP8", "fp4": "FP4"}
NOT_QUANTIZED_MODULES = [
    "lm_head", "vision_tower", "audio_tower",
    "embed_vision", "embed_audio", "multi_modal_projector",
]
BF16_REFERENCE_GIB = 11.95e9 * 2 / 2**30

VARIANT_SPECS = {
    "bf16": {
        "source": BASE_MODEL_ID,
        "method": "dense_bf16",
        "desc": "BF16 weights (reference)",
        "max_footprint_ratio": None,
    },
    "fp8": {
        "source": FP8_CHECKPOINT_ID or BASE_MODEL_ID,
        "method": "prequantized" if FP8_CHECKPOINT_ID else "fine_grained_fp8",
        "desc": (
            f"pre-quantized FP8 checkpoint {FP8_CHECKPOINT_ID}"
            if FP8_CHECKPOINT_ID
            else "on-the-fly block-wise FP8 (E4M3) via Transformers"
        ),
        "max_footprint_ratio": 0.75,
    },
    "fp4": {
        "source": FP4_CHECKPOINT_ID or BASE_MODEL_ID,
        "method": "prequantized" if FP4_CHECKPOINT_ID else "bnb_fp4",
        "desc": (
            f"pre-quantized FP4 checkpoint {FP4_CHECKPOINT_ID}"
            if FP4_CHECKPOINT_ID
            else "on-the-fly weight-only FP4 (E2M1) via bitsandbytes"
        ),
        "max_footprint_ratio": 0.60,
    },
}

CONDITIONS = ["baseline", "uncertain"]
CONDITION_LABELS = {"baseline": "Baseline", "uncertain": "Uncertainty-aware"}
SYSTEM_PROMPTS = {
    "baseline": None,
    "uncertain": (
        "Answer the question accurately and concisely. "
        "Output only the requested fact, without reasoning or explanation. "
        "If you cannot determine the answer with sufficient confidence, "
        "say exactly: I don't know. Do not guess or invent facts."
    ),
}

BATCH_SIZE = 32
MAX_NEW_TOKENS = 128
ENABLE_THINKING = False
PREFLIGHT_ONLY = False
DEBUG_SAMPLES_PER_BATCH = 3
CONTENT_FORMAT = "string"  # "auto", "string", or "parts"
STRICT_ROUNDTRIP = True
SANITY_MIN_FRACTION = 0.75
BATCH_CONSISTENCY_MIN = 0.5
MAX_CONSECUTIVE_ERROR_BATCHES = 3
MAX_PROMPT_TOKENS = 2048

JUDGE_SECTION_SIZE = 6
JUDGE_MAX_TOKENS = 12000
JUDGE_RETRIES = 4
MAX_CONSECUTIVE_JUDGE_FAILURES = 5
VALID_LABELS = {"correct", "incorrect", "not_attempted"}
LETTERS = "ABCDEFGHIJKLMNOP"

RESUME = True
MAX_QUESTIONS = 256  # Set to None for the full dataset.
N_BOOT = 2000

# Standalone default: local directory. Set OUTPUT_DIR to a persistent path
# (e.g. a mounted Drive directory) if you want checkpoints to survive resets.
OUTPUT_DIR = Path(os.environ.get("SIMPLEQA_OUTPUT_DIR", "./simpleqa_gemma4_quant"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_CHECKPOINT = OUTPUT_DIR / "gemma_outputs.json"
JUDGE_CHECKPOINT = OUTPUT_DIR / "judge_results.json"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
JUDGE_DETAIL_CSV = OUTPUT_DIR / "judge_details.csv"
LATEX_FILE = OUTPUT_DIR / "results_table.tex"

TURN_START, TURN_END = "<|turn>", "<|turn|>"
CHANNEL_START, CHANNEL_END = "<|channel>", "<|channel|>"
EMPTY_THOUGHT = f"{CHANNEL_START}thought\n{CHANNEL_END}"
MARKERS = {
    "turn start": TURN_START,
    "turn end": TURN_END,
    "channel start": CHANNEL_START,
    "channel end": CHANNEL_END,
}

# =============================================================================
# 2. PAUSE / CHECKPOINT CONTROL
# =============================================================================
class PauseRequested(BaseException):
    pass


class PreflightError(Exception):
    pass


class JudgeSectionError(Exception):
    pass


class PauseController:
    """First SIGINT requests a clean pause; second SIGINT stops immediately."""

    def __init__(self):
        self.requested = False
        self.original_handler = None
        self.installed = False

    def handler(self, signum, frame):
        if not self.requested:
            self.requested = True
            print(
                "\n[PAUSE] Interrupt received. Finishing the current step and "
                "saving checkpoints. Interrupt again for an immediate stop.",
                flush=True,
            )
        else:
            raise KeyboardInterrupt

    def install(self):
        if threading.current_thread() is threading.main_thread():
            self.original_handler = signal.signal(signal.SIGINT, self.handler)
            self.installed = True
        else:
            print("Warning: graceful pause unavailable outside the main thread.")

    def restore(self):
        if self.installed:
            signal.signal(signal.SIGINT, self.original_handler)
            self.installed = False

    def check(self):
        if self.requested:
            raise PauseRequested()


PAUSE = PauseController()


def interruptible_sleep(seconds):
    end = time.time() + seconds
    while time.time() < end:
        PAUSE.check()
        time.sleep(min(0.5, max(0.0, end - time.time())))


def get_secret(name, prompt=None):
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata
        value = (userdata.get(name) or "").strip()
        if value:
            return value
    except Exception:
        pass
    return getpass(prompt).strip() if prompt else ""


def load_json(path, default):
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    obj = json.load(handle)
                if candidate != path:
                    print(f"Recovered {path.name} from backup.")
                return obj
            except Exception as exc:
                print(f"Could not read {candidate.name}: {exc}")
    return default


def save_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        os.replace(path, bak)
    os.replace(tmp, path)


def archive_stale(path):
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = path.with_name(f"{path.stem}.stale-{stamp}{path.suffix}")
        os.replace(path, dest)
        print(f"Archived old checkpoint -> {dest.name}")


def sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def cfg_key(variant, condition):
    return f"{variant}__{condition}"


def split_cfg(cfg):
    return cfg.split("__", 1)


ALL_CONFIGS = [cfg_key(v, c) for v in VARIANTS for c in CONDITIONS]


def cfg_label(cfg):
    variant, condition = split_cfg(cfg)
    return f"{VARIANT_LABELS[variant]}\n{condition}"


# =============================================================================
# 3. DATASET
# =============================================================================
def find_column(columns, candidates):
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_records():
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    print(
        f"\nDataset: {DATASET_ID} | split={DATASET_SPLIT} | "
        f"rows={len(dataset)} | columns={dataset.column_names}"
    )
    question_col = find_column(
        dataset.column_names, ["question", "problem", "query", "prompt"]
    )
    answer_col = find_column(
        dataset.column_names,
        ["answer", "reference_answer", "target", "gold_answer"],
    )
    if question_col is None or answer_col is None:
        raise ValueError(f"Could not identify question/answer columns: {dataset.column_names}")

    total = len(dataset)
    if MAX_QUESTIONS is not None:
        dataset = dataset.select(range(min(MAX_QUESTIONS, total)))
    records = [
        {
            "id": i,
            "question": str(row[question_col]),
            "reference_answer": str(row[answer_col]),
        }
        for i, row in enumerate(dataset)
    ]
    if MAX_QUESTIONS is not None:
        print(
            f"\n*** SMOKE TEST: using {len(records)} of {total} questions. "
            "Set MAX_QUESTIONS=None for the full run. ***"
        )
    print("Selected questions:", len(records))
    print("Example:", records[0])
    return records


# =============================================================================
# 4. TEMPLATE / TOKENIZER
# =============================================================================
processor = None
TOK = None
model = None
STOP_IDS = None
PAD_ID = None
SPECIAL_STRINGS = []
_content_format = None
_bos_note_printed = False


def make_messages(question, condition, fmt=None):
    fmt = fmt or _content_format

    def wrap(text):
        return text if fmt == "string" else [{"type": "text", "text": text}]

    messages = []
    if SYSTEM_PROMPTS[condition]:
        messages.append({"role": "system", "content": wrap(SYSTEM_PROMPTS[condition])})
    messages.append({"role": "user", "content": wrap(question)})
    return messages


def template_call(messages, enable_thinking):
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(rendered, (list, tuple)):
        if len(rendered) != 1:
            raise PreflightError("Chat template returned multiple strings for one conversation.")
        rendered = rendered[0]
    if not isinstance(rendered, str):
        raise PreflightError(f"Chat template returned {type(rendered)}, expected str.")
    return rendered


def render_prompt(question, condition, enable_thinking=ENABLE_THINKING):
    global _bos_note_printed
    rendered = template_call(make_messages(question, condition), enable_thinking)
    bos = TOK.bos_token
    if bos and not rendered.startswith(bos):
        if not _bos_note_printed:
            print("[template] Template omitted BOS; prepending it.")
            _bos_note_printed = True
        rendered = bos + rendered
    return rendered


def detect_content_format():
    probe = "PROBE_QUESTION_12345?"
    outputs = {}
    for fmt in ("string", "parts"):
        try:
            outputs[fmt] = template_call(make_messages(probe, "uncertain", fmt), False)
        except Exception as exc:
            print(f"Content format {fmt!r} failed: {type(exc).__name__}: {str(exc)[:150]}")
    valid = {
        fmt: output for fmt, output in outputs.items()
        if probe in output and SYSTEM_PROMPTS["uncertain"] in output
    }
    if CONTENT_FORMAT != "auto":
        if CONTENT_FORMAT not in valid:
            raise PreflightError(f"CONTENT_FORMAT={CONTENT_FORMAT!r} does not render correctly.")
        return CONTENT_FORMAT
    if not valid:
        raise PreflightError("Neither string nor typed-parts content renders correctly.")
    if len(valid) == 2 and valid["string"] != valid["parts"]:
        raise PreflightError("String and typed-parts content render differently; set CONTENT_FORMAT.")
    chosen = "string" if "string" in valid else "parts"
    print("Selected content format:", chosen)
    return chosen


def check_marker_tokens():
    print("\n[check] Template markers:")
    for name, marker in MARKERS.items():
        ids = TOK.encode(marker, add_special_tokens=False)
        token_id = TOK.convert_tokens_to_ids(marker)
        valid = (
            len(ids) == 1 and token_id is not None
            and token_id == ids[0] and token_id != TOK.unk_token_id
        )
        print(f"  {name:14s} {marker!r:14s} -> {token_id} {'OK' if valid else 'PROBLEM'}")
        if not valid:
            raise PreflightError(f"Marker {marker!r} is not one known token.")
    print("  BOS:", repr(TOK.bos_token), TOK.bos_token_id,
          "| EOS:", repr(TOK.eos_token), TOK.eos_token_id,
          "| PAD:", repr(TOK.pad_token), TOK.pad_token_id)
    if TOK.bos_token_id is None or TOK.pad_token_id is None:
        raise PreflightError("Tokenizer must define BOS and PAD tokens.")


def check_rendered_prompt(question, condition, rendered):
    problems = []
    q = question.strip()
    system_prompt = SYSTEM_PROMPTS[condition]
    bos = TOK.bos_token
    user_marker = f"{TURN_START}user\n{q}"
    model_marker = f"{TURN_START}model\n"

    if bos and rendered.count(bos) != 1:
        problems.append(f"BOS appears {rendered.count(bos)} times; expected one")
    if user_marker not in rendered:
        problems.append("Exact user turn/question not found")
    if rendered.count(f"{TURN_START}user") != 1:
        problems.append("Expected exactly one user turn")
    if rendered.count(f"{TURN_START}model") != 1:
        problems.append("Expected exactly one model generation turn")
    expected_ends = 2 if system_prompt else 1
    if rendered.count(TURN_END) != expected_ends:
        problems.append(f"Expected {expected_ends} turn-end markers")
    if system_prompt:
        system_marker = f"{TURN_START}system\n"
        if system_marker not in rendered or system_prompt not in rendered:
            problems.append("System prompt missing from rendered template")
        elif rendered.index(system_marker) > rendered.index(user_marker):
            problems.append("System turn occurs after user turn")
    elif f"{TURN_START}system" in rendered:
        problems.append("Baseline must not contain a system turn")
    if model_marker in rendered:
        tail = rendered.rsplit(model_marker, 1)[1]
        if tail not in ("", EMPTY_THOUGHT):
            problems.append(f"Unexpected generation-prompt tail: {tail!r}")
        if rendered.index(model_marker) < rendered.index(user_marker):
            problems.append("Model turn precedes user turn")
    else:
        problems.append("Generation prompt missing")
    if "{{" in rendered or "{%" in rendered:
        problems.append("Unrendered Jinja syntax found")
    return problems


def encode_batch(texts):
    TOK.padding_side = "left"
    try:
        return processor(
            text=texts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
    except (TypeError, ValueError) as exc:
        print(f"Processor encoding fallback to tokenizer: {exc}")
        return TOK(texts, add_special_tokens=False, padding=True, return_tensors="pt")


def check_tokenization(texts, label):
    encoded = encode_batch(texts)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    print(f"  [{label}] shape={tuple(ids.shape)} padding_side={TOK.padding_side}")
    for row in range(ids.shape[0]):
        row_mask = mask[row]
        n_pad = int((row_mask == 0).sum())
        if row_mask[-1].item() != 1 or not torch.all(row_mask[n_pad:] == 1):
            raise PreflightError(f"[{label}] Padding is not a clean left prefix.")
        real = ids[row, n_pad:]
        if real[0].item() != TOK.bos_token_id:
            raise PreflightError(f"[{label}] First real token is not BOS.")
        if int((real == TOK.bos_token_id).sum()) != 1:
            raise PreflightError(f"[{label}] Expected exactly one BOS token.")
        decoded = TOK.decode(
            real, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != texts[row]:
            message = f"[{label}] encode/decode round-trip mismatch."
            if STRICT_ROUNDTRIP:
                raise PreflightError(message)
            print("  WARNING:", message)
    print(f"  [{label}] left padding, BOS, and round-trip checks passed")


def build_prompt_cache(records):
    print("\n[check] Rendering and validating prompts...")
    cache = {}
    issues_found = []
    for condition in CONDITIONS:
        for record in records:
            rendered = render_prompt(record["question"], condition)
            issues = check_rendered_prompt(record["question"], condition, rendered)
            if issues:
                issues_found.append((record["id"], condition, issues, rendered[:300]))
            cache[(condition, record["id"])] = rendered
    if issues_found:
        for item in issues_found[:5]:
            print("PROMPT PROBLEM:", item)
        raise PreflightError(f"{len(issues_found)} prompt(s) failed structural checks.")
    for condition in CONDITIONS:
        texts = [cache[(condition, r["id"])] for r in records]
        lengths = [len(ids) for ids in TOK(texts, add_special_tokens=False)["input_ids"]]
        print(
            f"  {condition}: {len(texts)} prompts | token min/mean/max="
            f"{min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}"
        )
        if max(lengths) > MAX_PROMPT_TOKENS:
            raise PreflightError(f"Prompt length {max(lengths)} exceeds {MAX_PROMPT_TOKENS}.")
    return cache


def run_static_checks(records):
    global _content_format, SPECIAL_STRINGS
    print("\n" + "=" * 70 + "\nSTATIC PREFLIGHT\n" + "=" * 70)
    check_marker_tokens()
    _content_format = detect_content_format()
    probe = min(records, key=lambda r: len(r["question"]))
    for condition in CONDITIONS:
        rendered = render_prompt(probe["question"], condition)
        print(f"\nRendered prompt ({condition}):\n{rendered!r}")
        problems = check_rendered_prompt(probe["question"], condition, rendered)
        if problems:
            raise PreflightError(f"Template problems for {condition}: {problems}")
        on = template_call(make_messages(probe["question"], condition), True)
        off = template_call(make_messages(probe["question"], condition), False)
        if on == off:
            print("Warning: template renders thinking=True and False identically.")
    mixed = [records[0], records[-1], *records[:2]]
    for condition in CONDITIONS:
        check_tokenization(
            [render_prompt(r["question"], condition) for r in mixed],
            f"static/{condition}",
        )
    cache = build_prompt_cache(records)
    SPECIAL_STRINGS = sorted(
        {s for s in set(TOK.all_special_tokens) | set(MARKERS.values()) if s},
        key=len, reverse=True,
    )
    print("Static preflight passed.")
    return cache


# =============================================================================
# 5. GENERATION
# =============================================================================
def parse_generation(token_ids):
    ids = list(token_ids)
    stop_set = set(STOP_IDS)
    stop_idx = next((i for i, token_id in enumerate(ids) if token_id in stop_set), None)
    finished = stop_idx is not None
    if finished:
        ids = ids[:stop_idx]
    ids = [token_id for token_id in ids if token_id != PAD_ID]
    raw = TOK.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    flags, answer_text = [], raw
    if CHANNEL_START in answer_text:
        if CHANNEL_END in answer_text:
            answer_text = answer_text.rsplit(CHANNEL_END, 1)[1]
            flags.append("thought_channel_stripped")
        else:
            answer_text = ""
            flags.append("unterminated_thought")
    elif CHANNEL_END in answer_text:
        answer_text = answer_text.rsplit(CHANNEL_END, 1)[1]
        flags.append("stray_channel_end")
    if TURN_START in answer_text:
        answer_text = answer_text.split(TURN_START, 1)[0]
        flags.append("extra_turn_cut")
    for special in SPECIAL_STRINGS:
        answer_text = answer_text.replace(special, "")
    answer = answer_text.strip()
    if not answer:
        flags.append("empty_answer")
    if not finished:
        flags.append("truncated")
    invalid = not answer or "unterminated_thought" in flags
    return {
        "answer": answer, "raw": raw, "finished": finished, "flags": flags,
        "n_generated_tokens": len(ids), "invalid": invalid,
    }


def generate_from_prompts(prompts):
    encoded = encode_batch(prompts)
    encoded = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in encoded.items()
    }
    width = encoded["input_ids"].shape[1]
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=STOP_IDS,
            pad_token_id=PAD_ID,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not torch.is_tensor(output):
        output = output.sequences
    if output.shape[1] <= width:
        raise RuntimeError("generate() returned no new tokens.")
    generated = output[:, width:].cpu().tolist()
    results = []
    for tokens, prompt_len in zip(generated, prompt_lengths):
        result = parse_generation(tokens)
        result["prompt_tokens"] = int(prompt_len)
        results.append(result)
    del encoded, output
    return results, elapsed


def error_result(message):
    return {
        "answer": "", "raw": message, "finished": False,
        "flags": ["generation_error"], "n_generated_tokens": 0,
        "invalid": True, "generation_error": True,
        "latency_seconds": 0.0, "prompt_tokens": 0,
    }


def generate_batch_safely(prompts):
    try:
        results, elapsed = generate_from_prompts(prompts)
        for result in results:
            result["generation_error"] = False
            result["latency_seconds"] = elapsed / len(prompts)
        return results
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        if len(prompts) > 1:
            midpoint = len(prompts) // 2
            return generate_batch_safely(prompts[:midpoint]) + generate_batch_safely(prompts[midpoint:])
        return [error_result("[GENERATION_ERROR: CUDA out of memory]")]
    except PauseRequested:
        raise
    except Exception as exc:
        if len(prompts) > 1:
            midpoint = len(prompts) // 2
            return generate_batch_safely(prompts[:midpoint]) + generate_batch_safely(prompts[midpoint:])
        print("Generation failure:", type(exc).__name__, str(exc)[:250])
        return [error_result(f"[GENERATION_ERROR: {type(exc).__name__}: {str(exc)[:200]}]")]


def show_sample(question, reference, prompt, result, full_prompt=False):
    print("    question :", repr(question[:160]))
    print("    prompt   :", repr(prompt if full_prompt else prompt[-160:]))
    print("    raw      :", repr(result["raw"][:300]))
    print("    answer   :", repr(result["answer"][:300]))
    print(
        f"    tokens={result['n_generated_tokens']} finished={result['finished']} "
        f"flags={result['flags']}"
    )
    print("    reference:", repr(reference))


# =============================================================================
# 6. MODEL LOADING / PREFLIGHT
# =============================================================================
def unload_model():
    global model
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_variant(variant):
    global model, STOP_IDS, PAD_ID
    spec = VARIANT_SPECS[variant]
    unload_model()
    print(
        f"\n{'=' * 70}\nLOADING {VARIANT_LABELS[variant]}: {spec['desc']}\n"
        f"source={spec['source']} method={spec['method']}\n{'=' * 70}",
        flush=True,
    )
    kwargs = {
        "dtype": MODEL_DTYPE,
        "device_map": {"": 0},
        "trust_remote_code": TRUST_REMOTE_CODE,
    }
    if spec["method"] == "fine_grained_fp8":
        if torch.cuda.get_device_capability(0) < (8, 9):
            raise PreflightError("On-the-fly FP8 requires CUDA compute capability >= 8.9.")
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(
            modules_to_not_convert=NOT_QUANTIZED_MODULES
        )
    elif spec["method"] == "bnb_fp4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=MODEL_DTYPE,
            bnb_4bit_use_double_quant=False,
            llm_int8_skip_modules=NOT_QUANTIZED_MODULES,
        )
    start = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(spec["source"], **kwargs)
    model.eval()
    print(f"Loaded in {time.time() - start:.1f}s | device={model.device}")
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    turn_end_id = TOK.convert_tokens_to_ids(TURN_END)
    STOP_IDS = sorted((set(eos) | {TOK.eos_token_id, turn_end_id}) - {None})
    PAD_ID = TOK.pad_token_id
    print("Stop token IDs:", STOP_IDS)
    if turn_end_id not in STOP_IDS:
        raise PreflightError("Turn-end marker is not configured as a stop token.")
    return spec


def quantization_report(variant):
    spec = VARIANT_SPECS[variant]
    print("\n[check] Quantization report")
    print("  config quantization:", getattr(model.config, "quantization_config", None))
    dtype_numel = Counter()
    for parameter in model.parameters():
        dtype_numel[str(parameter.dtype)] += parameter.numel()
    module_types = Counter(type(module).__name__ for module in model.modules())
    interesting = {
        name: count for name, count in module_types.items()
        if any(tag in name.lower() for tag in (
            "linear4bit", "linear8bit", "fp8", "fp4", "fpquant", "nvfp4"
        ))
    }
    footprint = model.get_memory_footprint() / 2**30
    ratio = footprint / BF16_REFERENCE_GIB
    print("  parameter elements by dtype:", dict(dtype_numel))
    print("  quantized module types:", interesting or "none detected")
    print(
        f"  footprint={footprint:.2f} GiB ({ratio:.2f}x reference) | "
        f"CUDA allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )
    method = spec["method"]
    if method == "dense_bf16":
        dominant = max(dtype_numel, key=dtype_numel.get)
        if dominant != "torch.bfloat16" or interesting:
            raise PreflightError(
                f"BF16 reference does not look like plain BF16 (dominant={dominant}, modules={interesting})."
            )
    elif method == "fine_grained_fp8":
        if not any("float8" in dtype for dtype in dtype_numel):
            raise PreflightError("FP8 requested, but no float8 parameters were detected.")
    elif method == "bnb_fp4":
        if not any("linear4bit" in name.lower() for name in interesting):
            raise PreflightError("FP4 requested, but no Linear4bit modules were detected.")
        quant_type = next(
            (getattr(module.weight, "quant_type", None)
             for module in model.modules() if type(module).__name__ == "Linear4bit"),
            None,
        )
        print("  bitsandbytes quant_type:", quant_type)
        if quant_type != "fp4":
            raise PreflightError(f"Expected bitsandbytes quant_type 'fp4', got {quant_type!r}.")
    max_ratio = spec["max_footprint_ratio"]
    if max_ratio is not None and ratio > max_ratio:
        raise PreflightError(f"Footprint ratio {ratio:.2f} exceeds expected maximum {max_ratio}.")
    return {
        "footprint_gib": footprint,
        "footprint_ratio": ratio,
        "dtype_numel": dict(dtype_numel),
        "quantized_modules": interesting,
    }


SANITY_QUESTIONS = [
    ("What is the capital of France?", "paris"),
    ("Who wrote the play Romeo and Juliet?", "shakespeare"),
    ("In what year did World War II end?", "1945"),
    ("What is the chemical symbol for gold?", "au"),
]
UNANSWERABLE_PROBE = (
    "What was the name of the third cat owned by the mayor of the village "
    "of Quendlington-on-Sea in 1893?"
)


def run_model_preflight(variant, records, prompt_cache):
    print(f"\n{'=' * 70}\nMODEL PREFLIGHT: {VARIANT_LABELS[variant]}\n{'=' * 70}")
    info = quantization_report(variant)
    print("\n[check] Known-answer generations")
    passed = total = 0
    for condition in CONDITIONS:
        prompts = [render_prompt(question, condition) for question, _ in SANITY_QUESTIONS]
        check_tokenization(prompts, f"sanity/{condition}")
        results, _ = generate_from_prompts(prompts)
        for (question, keyword), result in zip(SANITY_QUESTIONS, results):
            ok = (
                not result["invalid"]
                and re.search(rf"\b{re.escape(keyword)}\b", result["answer"].lower()) is not None
            )
            passed += int(ok)
            total += 1
            print(f"  [{condition}] {'PASS' if ok else 'FAIL'} | {question}")
            print("      answer:", repr(result["answer"][:120]), "| flags:", result["flags"])
    fraction = passed / total
    print(f"Sanity score: {passed}/{total} ({fraction:.0%})")
    if fraction < SANITY_MIN_FRACTION:
        raise PreflightError(f"Sanity pass rate {fraction:.0%} is below {SANITY_MIN_FRACTION:.0%}.")

    print("\n[check] Unanswerable probe (informational)")
    for condition in CONDITIONS:
        result, _ = generate_from_prompts([render_prompt(UNANSWERABLE_PROBE, condition)])
        print(f"  [{condition}] {result[0]['answer'][:160]!r}")

    print("\n[check] Batched vs single generation")
    sample = records[:min(6, len(records))]
    prompts = [prompt_cache[("uncertain", record["id"])] for record in sample]
    batched, _ = generate_from_prompts(prompts)
    matches = 0
    for prompt, batch_result in zip(prompts, batched):
        single, _ = generate_from_prompts([prompt])
        same = single[0]["answer"] == batch_result["answer"]
        matches += int(same)
        if not same:
            print("  Difference:", repr(batch_result["answer"][:80]), "vs", repr(single[0]["answer"][:80]))
    rate = matches / len(prompts)
    print(f"Identical answers: {matches}/{len(prompts)} ({rate:.0%})")
    if rate < BATCH_CONSISTENCY_MIN:
        raise PreflightError("Batched and single generation disagree; inspect padding/template behavior.")
    print(f"Model preflight passed for {VARIANT_LABELS[variant]}.")
    return info


# =============================================================================
# 7. GENERATION CHECKPOINT / RUN
# =============================================================================
def experiment_fingerprint():
    template = getattr(processor, "chat_template", None) or getattr(TOK, "chat_template", None) or ""
    return sha({
        "base_model": BASE_MODEL_ID,
        "variants": {variant: VARIANT_SPECS[variant] for variant in VARIANTS},
        "dtype": str(MODEL_DTYPE),
        "dataset": DATASET_ID,
        "split": DATASET_SPLIT,
        "prompts": SYSTEM_PROMPTS,
        "conditions": CONDITIONS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "thinking": ENABLE_THINKING,
        "chat_template": template,
        "content_format": _content_format,
        "max_questions": MAX_QUESTIONS,
    })


def init_gen_checkpoint(records, fingerprint):
    checkpoint = load_json(GEN_CHECKPOINT, {})
    if checkpoint and (
        not RESUME or checkpoint.get("experiment_hash") != fingerprint
    ):
        print("Generation checkpoint is stale or RESUME=False; starting a new run.")
        archive_stale(GEN_CHECKPOINT)
        checkpoint = {}
    if not checkpoint:
        checkpoint = {
            "experiment_hash": fingerprint,
            "base_model": BASE_MODEL_ID,
            "dataset_id": DATASET_ID,
            "variants": VARIANTS,
            "conditions": CONDITIONS,
            "records": {},
            "variant_meta": {},
        }
    checkpoint.setdefault("records", {})
    checkpoint.setdefault("variant_meta", {})
    checkpoint["environment"] = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    by_id = {record["id"]: record for record in records}
    for qid, saved in checkpoint["records"].items():
        current = by_id.get(int(qid))
        if current and saved.get("question") != current["question"]:
            raise RuntimeError(f"Checkpoint question {qid} differs from the current dataset.")
    return checkpoint


def n_done(checkpoint):
    return sum(
        1 for record in checkpoint["records"].values()
        for config in ALL_CONFIGS if config in record
    )


def audit_config(checkpoint, key):
    results = [
        record[key] for record in checkpoint["records"].values() if key in record
    ]
    if not results:
        return
    flags = Counter(flag for result in results for flag in result["flags"])
    print(
        f"  AUDIT {key}: n={len(results)} invalid={sum(r['invalid'] for r in results)} "
        f"truncated={flags.get('truncated', 0)} empty={flags.get('empty_answer', 0)} "
        f"mean_tokens={np.mean([r['n_generated_tokens'] for r in results]):.1f}"
    )
    print("        flag counts:", dict(flags))


def run_generation(records, prompt_cache, checkpoint):
    output_records = checkpoint["records"]
    total_jobs = len(records) * len(ALL_CONFIGS)
    print(f"\nGeneration jobs: {n_done(checkpoint)}/{total_jobs} already complete")

    for variant in VARIANTS:
        def pending(condition):
            key = cfg_key(variant, condition)
            return [r for r in records if key not in output_records.get(str(r["id"]), {})]

        if not any(pending(condition) for condition in CONDITIONS):
            print(f"\n[{VARIANT_LABELS[variant]}] already complete; skipping.")
            continue
        PAUSE.check()
        try:
            load_variant(variant)
            checkpoint["variant_meta"][variant] = {
                **run_model_preflight(variant, records, prompt_cache),
                "spec": VARIANT_SPECS[variant],
                "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            save_json_atomic(GEN_CHECKPOINT, checkpoint)
        except PreflightError as exc:
            print(f"\nPREFLIGHT FAILED for {VARIANT_LABELS[variant]}: {exc}")
            checkpoint.setdefault("failed_variants", {})[variant] = str(exc)[:500]
            unload_model()
            save_json_atomic(GEN_CHECKPOINT, checkpoint)
            continue
        except Exception as exc:
            print(f"\nLOAD/PREFLIGHT ERROR for {VARIANT_LABELS[variant]}: {type(exc).__name__}: {exc}")
            checkpoint.setdefault("failed_variants", {})[variant] = f"{type(exc).__name__}: {str(exc)[:400]}"
            unload_model()
            save_json_atomic(GEN_CHECKPOINT, checkpoint)
            continue

        checkpoint.get("failed_variants", {}).pop(variant, None)
        if PREFLIGHT_ONLY:
            unload_model()
            continue

        for condition in CONDITIONS:
            key = cfg_key(variant, condition)
            todo = pending(condition)
            print(f"\n--- {VARIANT_LABELS[variant]} / {condition} | pending={len(todo)} ---")
            consecutive_error_batches = 0
            first_batch = True
            for start in range(0, len(todo), BATCH_SIZE):
                PAUSE.check()
                batch = todo[start:start + BATCH_SIZE]
                prompts = [prompt_cache[(condition, record["id"])] for record in batch]
                results = generate_batch_safely(prompts)
                if len(results) != len(batch):
                    raise RuntimeError("Generated result count does not match batch size.")

                if first_batch or any(result["invalid"] for result in results):
                    shown = results[:DEBUG_SAMPLES_PER_BATCH] if first_batch else [
                        result for result in results if result["invalid"]
                    ][:DEBUG_SAMPLES_PER_BATCH]
                    print("  Debug samples:")
                    for result in shown:
                        index = results.index(result)
                        show_sample(
                            batch[index]["question"], batch[index]["reference_answer"],
                            prompts[index], result, full_prompt=(first_batch and index == 0),
                        )
                    first_batch = False

                for record, result in zip(batch, results):
                    saved = output_records.setdefault(str(record["id"]), {
                        "id": record["id"],
                        "question": record["question"],
                        "reference_answer": record["reference_answer"],
                    })
                    saved[key] = {
                        "response": result["answer"],
                        "raw_response": result["raw"],
                        "flags": result["flags"],
                        "finished": result["finished"],
                        "invalid": result["invalid"],
                        "generation_error": result.get("generation_error", False),
                        "n_generated_tokens": result["n_generated_tokens"],
                        "prompt_tokens": result["prompt_tokens"],
                        "latency_seconds": result["latency_seconds"],
                    }
                save_json_atomic(GEN_CHECKPOINT, checkpoint)
                bad = sum(result["invalid"] for result in results)
                truncated = sum("truncated" in result["flags"] for result in results)
                print(
                    f"  {VARIANT_LABELS[variant]}/{condition}: "
                    f"{min(start + len(batch), len(todo))}/{len(todo)} | "
                    f"invalid={bad} truncated={truncated} | overall={n_done(checkpoint)}/{total_jobs}",
                    flush=True,
                )
                consecutive_error_batches = (
                    consecutive_error_batches + 1
                    if all(result.get("generation_error") for result in results) else 0
                )
                if consecutive_error_batches >= MAX_CONSECUTIVE_ERROR_BATCHES:
                    raise RuntimeError("Consecutive batches failed completely; stopping to avoid bad results.")
                PAUSE.check()
            audit_config(checkpoint, key)
        unload_model()
    return checkpoint


def completed_configs(checkpoint, records):
    return [
        config for config in ALL_CONFIGS
        if all(config in checkpoint["records"].get(str(record["id"]), {}) for record in records)
    ]


# =============================================================================
# 8. NEMOTRON JUDGE
# =============================================================================
JUDGE_SYSTEM_PROMPT = r"""
You are a strict, evidence-grounded factuality evaluator for the SimpleQA
short-answer benchmark.

For each candidate, compare the question, supplied reference answer, and
candidate response. Treat the reference answer as the primary evidence.
Do not answer the question from memory, use outside facts, or override the
reference answer. Equivalent wording and unambiguous paraphrases may be correct.

Assign exactly one label:
- correct: gives the expected answer or an unambiguous equivalent. Harmless
  extra detail is allowed unless demonstrably false or materially misleading
  from the supplied evidence.
- incorrect: attempts an answer but gives a materially different, contradictory,
  or unsupported central answer.
- not_attempted: explicitly declines, says it does not know, or gives no
  substantive answer. A wrong answer plus uncertainty is incorrect.

Set unsupported_claim=true if the candidate includes a material factual claim
not supported by the supplied evidence. Do not mark harmless incidental detail
unsupported merely because the reference is not exhaustive.
Set invalid_response=true if empty, malformed, nonsensical, or not reasonably
scorable. Clear refusals are not invalid.

Give one brief rationale sentence, at most 20 words. Judge each candidate
independently; identical responses must receive identical labels.

Return exactly one JSON object with this schema:
{"results":[{"id":0,"A":{"label":"correct","unsupported_claim":false,
"invalid_response":false,"rationale":"The response gives the reference answer."}}]}

Include every input ID exactly once. For each ID, include exactly the candidate
letters provided, and all four fields for each candidate. Labels must be
correct, incorrect, or not_attempted. Flags must be JSON booleans. No markdown
or text outside the JSON object.
"""


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("No valid JSON object found in judge response.")


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"Not a boolean: {value!r}")


def init_judge_checkpoint(fingerprint):
    judge_hash = sha({
        "generation": fingerprint,
        "judge_model": OPENROUTER_MODEL,
        "judge_prompt": JUDGE_SYSTEM_PROMPT,
    })
    checkpoint = load_json(JUDGE_CHECKPOINT, {})
    if checkpoint and (not RESUME or checkpoint.get("judge_hash") != judge_hash):
        print("Judge checkpoint is stale or RESUME=False; starting a new judge run.")
        archive_stale(JUDGE_CHECKPOINT)
        checkpoint = {}
    if not checkpoint:
        checkpoint = {
            "judge_hash": judge_hash,
            "judge_model": OPENROUTER_MODEL,
            "judgments": {},
            "sections": [],
        }
    checkpoint.setdefault("judgments", {})
    checkpoint.setdefault("sections", [])
    return checkpoint


def judge_section(client, section, gen_records, section_number, section_total):
    letter_maps, items = {}, []
    for record, configs in section:
        order = random.Random(SEED * 1_000_003 + record["id"]).sample(configs, len(configs))
        mapping = {LETTERS[i]: config for i, config in enumerate(order)}
        letter_maps[record["id"]] = mapping
        items.append({
            "id": record["id"],
            "question": record["question"],
            "reference_answer": record["reference_answer"],
            "candidates": {
                letter: gen_records[str(record["id"])][config]["response"]
                for letter, config in mapping.items()
            },
        })
    user_prompt = (
        f"Evaluate section {section_number}/{section_total}. Preserve IDs exactly. "
        "Judge each candidate independently and return JSON matching the schema.\n\n"
        + json.dumps({"items": items}, ensure_ascii=False)
    )
    last_error = None
    for attempt in range(JUDGE_RETRIES):
        PAUSE.check()
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=JUDGE_MAX_TOKENS,
                extra_body={
                    "reasoning": {"enabled": False},
                    "response_format": {"type": "json_object"},
                },
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                raise ValueError(f"Judge returned empty content (finish={choice.finish_reason}).")
            parsed = extract_json(content)
            if choice.finish_reason == "length":
                raise ValueError("Judge output truncated; lower JUDGE_SECTION_SIZE.")
            results = parsed.get("results")
            if not isinstance(results, list):
                raise ValueError("'results' missing or not a list.")
            by_id = {}
            for item in results:
                if isinstance(item, dict):
                    try:
                        by_id[int(item.get("id"))] = item
                    except (TypeError, ValueError):
                        pass
            if set(by_id) != set(letter_maps):
                raise ValueError(f"Judge ID mismatch: expected {sorted(letter_maps)}, got {sorted(by_id)}")
            output = {}
            for qid, mapping in letter_maps.items():
                output[qid] = {}
                for letter, config in mapping.items():
                    verdict = by_id[qid].get(letter)
                    if not isinstance(verdict, dict) or verdict.get("label") not in VALID_LABELS:
                        raise ValueError(f"Bad/missing verdict for question {qid}, candidate {letter}.")
                    output[qid][config] = {
                        "label": verdict["label"],
                        "unsupported_claim": to_bool(verdict.get("unsupported_claim", False)),
                        "invalid_response": to_bool(verdict.get("invalid_response", False)),
                        "rationale": str(verdict.get("rationale", ""))[:500],
                    }
            meta = {
                "section": section_number,
                "ids": sorted(letter_maps),
                "model": OPENROUTER_MODEL,
                "usage": response.usage.model_dump() if response.usage else None,
            }
            return output, meta
        except PauseRequested:
            raise
        except Exception as exc:
            last_error = exc
            wait = 30 if isinstance(exc, RateLimitError) else min(2**attempt, 30)
            print(
                f"Judge section {section_number} attempt {attempt + 1} failed: "
                f"{type(exc).__name__}: {str(exc)[:200]}. Retry in {wait}s."
            )
            interruptible_sleep(wait)
    raise JudgeSectionError(f"Could not judge section {section_number}: {last_error}")


def run_judge(client, records, gen_records, configs, fingerprint):
    checkpoint = init_judge_checkpoint(fingerprint)
    judgments = checkpoint["judgments"]
    todo, changed = [], False
    for record in records:
        qid = str(record["id"])
        saved = judgments.setdefault(qid, {})
        need = []
        for config in configs:
            if config in saved:
                continue
            generated = gen_records[qid][config]
            if generated.get("invalid"):
                saved[config] = {
                    "label": "unscored",
                    "unsupported_claim": False,
                    "invalid_response": True,
                    "rationale": "Invalid generation; not sent to judge.",
                }
                changed = True
            else:
                need.append(config)
        if need:
            todo.append((record, need))
    if changed:
        save_json_atomic(JUDGE_CHECKPOINT, checkpoint)

    sections = [todo[i:i + JUDGE_SECTION_SIZE] for i in range(0, len(todo), JUDGE_SECTION_SIZE)]
    print(f"\nJudge: {len(todo)} question(s) need judging in {len(sections)} section(s).")
    failures = 0
    for index, section in enumerate(sections, start=1):
        PAUSE.check()
        try:
            result, meta = judge_section(client, section, gen_records, index, len(sections))
        except JudgeSectionError as exc:
            failures += 1
            print(f"  {exc}; will retry on next run.")
            if failures >= MAX_CONSECUTIVE_JUDGE_FAILURES:
                save_json_atomic(JUDGE_CHECKPOINT, checkpoint)
                raise RuntimeError("Too many consecutive judge failures; check API key and rate limits.")
            continue
        failures = 0
        for qid, per_config in result.items():
            judgments[str(qid)].update(per_config)
        checkpoint["sections"].append(meta)
        save_json_atomic(JUDGE_CHECKPOINT, checkpoint)
        ids = [item[0]["id"] for item in section]
        print(f"  Judged section {index}/{len(sections)} (IDs {ids[0]}-{ids[-1]})", flush=True)
    return checkpoint


# =============================================================================
# 9. METRICS / BOOTSTRAP / PLOTS / LATEX
# =============================================================================
def build_details(records, gen_records, judgments, configs):
    rows = []
    for record in records:
        qid = str(record["id"])
        for config in configs:
            variant, condition = split_cfg(config)
            generated = gen_records[qid][config]
            verdict = judgments.get(qid, {}).get(config, {})
            response = generated.get("response", "")
            invalid_generation = bool(
                generated.get("invalid") or generated.get("generation_error") or not response.strip()
            )
            label = "unscored" if invalid_generation else verdict.get("label", "unscored")
            rows.append({
                "id": record["id"],
                "config": config,
                "variant": variant,
                "condition": condition,
                "question": record["question"],
                "reference_answer": record["reference_answer"],
                "response": response,
                "raw_response": generated.get("raw_response", ""),
                "label": label,
                "unsupported_claim": bool(verdict.get("unsupported_claim", False)),
                "invalid_response": bool(verdict.get("invalid_response", False) or invalid_generation),
                "truncated": "truncated" in generated.get("flags", []),
                "flags": "|".join(generated.get("flags", [])),
                "rationale": verdict.get("rationale", ""),
                "latency_seconds": generated.get("latency_seconds", np.nan),
                "n_generated_tokens": generated.get("n_generated_tokens", np.nan),
                "response_words": len(response.split()),
                "response_chars": len(response),
            })
    return pd.DataFrame(rows)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def calculate_metrics(frame):
    scored = frame[frame["label"].isin(VALID_LABELS)]
    n_total = len(frame)
    n_scored = len(scored)
    correct = int((scored["label"] == "correct").sum())
    incorrect = int((scored["label"] == "incorrect").sum())
    abstained = int((scored["label"] == "not_attempted").sum())
    attempted = correct + incorrect
    accuracy = safe_div(correct, n_total)
    attempted_accuracy = safe_div(correct, attempted)
    fscore = (
        safe_div(2 * accuracy * attempted_accuracy, accuracy + attempted_accuracy)
        if np.isfinite(accuracy) and np.isfinite(attempted_accuracy) else np.nan
    )
    return {
        "variant": frame["variant"].iloc[0],
        "condition": frame["condition"].iloc[0],
        "N": n_total,
        "Scored": n_scored,
        "Correct": correct,
        "Incorrect": incorrect,
        "Abstained": abstained,
        "Unscored": n_total - n_scored,
        "Overall accuracy": accuracy,
        "Attempted accuracy": attempted_accuracy,
        "Hallucination rate among attempted": safe_div(incorrect, attempted),
        "Coverage": safe_div(attempted, n_total),
        "Abstention rate": safe_div(abstained, n_total),
        "SimpleQA F-score": fscore,
        "Unsupported-claim rate": safe_div(int(scored["unsupported_claim"].sum()), n_scored),
        "Invalid-response rate": safe_div(int(frame["invalid_response"].sum()), n_total),
        "Truncated rate": safe_div(int(frame["truncated"].sum()), n_total),
        "Mean response words": frame["response_words"].mean(),
        "Mean generated tokens": frame["n_generated_tokens"].mean(),
        "Mean latency (s)": frame["latency_seconds"].mean(),
    }


PERCENT_METRICS = [
    "Overall accuracy", "Attempted accuracy", "Hallucination rate among attempted",
    "Coverage", "Abstention rate", "Unsupported-claim rate",
    "Invalid-response rate", "Truncated rate",
]
CI_METRICS = [
    "Overall accuracy", "Hallucination rate among attempted", "Coverage",
    "Abstention rate", "SimpleQA F-score",
]


def metrics_from_counts(correct, incorrect, abstained, n):
    correct = np.asarray(correct, dtype=float)
    incorrect = np.asarray(incorrect, dtype=float)
    abstained = np.asarray(abstained, dtype=float)
    attempted = correct + incorrect
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = correct / n
        attempted_accuracy = np.where(attempted > 0, correct / attempted, np.nan)
        denominator = accuracy + attempted_accuracy
        fscore = np.where(
            np.isfinite(denominator) & (denominator > 0),
            2 * accuracy * attempted_accuracy / denominator,
            np.nan,
        )
        return {
            "Overall accuracy": accuracy,
            "Attempted accuracy": attempted_accuracy,
            "Hallucination rate among attempted": np.where(
                attempted > 0, incorrect / attempted, np.nan
            ),
            "Coverage": attempted / n,
            "Abstention rate": abstained / n,
            "SimpleQA F-score": fscore,
        }


def paired_bootstrap(details, configs, n_boot=N_BOOT):
    pivot = (
        details.pivot(index="id", columns="config", values="label")
        .reindex(columns=configs).fillna("unscored")
    )
    array = pivot.to_numpy()
    n = len(array)
    counts = {
        config: tuple((array[:, j] == label).astype(np.int64)
                      for label in ("correct", "incorrect", "not_attempted"))
        for j, config in enumerate(configs)
    }
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, n, size=(n_boot, n))
    point, boot = {}, {}
    for config, (c, i, a) in counts.items():
        point[config] = metrics_from_counts(c.sum(), i.sum(), a.sum(), n)
        boot[config] = metrics_from_counts(c[indices].sum(1), i[indices].sum(1), a[indices].sum(1), n)

    def interval(samples):
        finite = samples[np.isfinite(samples)]
        if not len(finite):
            return np.nan, np.nan
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))

    ci = {
        metric: {
            config: (float(point[config][metric]), *interval(boot[config][metric]))
            for config in configs
        }
        for metric in CI_METRICS
    }
    pairs = []
    for variant in VARIANTS:
        uncertain, baseline = cfg_key(variant, "uncertain"), cfg_key(variant, "baseline")
        if uncertain in configs and baseline in configs:
            pairs.append((uncertain, baseline))
    for condition in CONDITIONS:
        for j, variant_j in enumerate(VARIANTS):
            for variant_i in VARIANTS[:j]:
                a, b = cfg_key(variant_j, condition), cfg_key(variant_i, condition)
                if a in configs and b in configs:
                    pairs.append((a, b))
    difference_rows = []
    for metric in CI_METRICS:
        for a, b in pairs:
            low, high = interval(boot[a][metric] - boot[b][metric])
            difference_rows.append({
                "metric": metric,
                "comparison": f"{a.replace('__', '/')} - {b.replace('__', '/')}",
                "difference": float(point[a][metric] - point[b][metric]),
                "CI lower": low,
                "CI upper": high,
            })
    return ci, pd.DataFrame(difference_rows)


def make_plots(metrics, configs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (9, 5), "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    palette = {"baseline": "#4C72B0", "uncertain": "#DD8452"}
    colors = [palette[split_cfg(config)[1]] for config in configs]
    labels = [cfg_label(config) for config in configs]
    plot_metrics = [
        ("Overall accuracy", "Overall accuracy"),
        ("Hallucination rate among attempted", "Hallucination rate (attempted)"),
        ("Coverage", "Coverage / attempt rate"),
        ("Abstention rate", "Abstention rate"),
        ("SimpleQA F-score", "SimpleQA F-score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes = axes.flatten()
    for axis, (column, title) in zip(axes, plot_metrics):
        values = metrics.loc[configs, column].to_numpy(dtype=float)
        bars = axis.bar(range(len(configs)), values, color=colors)
        axis.set_xticks(range(len(configs)))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Rate / score")
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                label = f"{value:.3f}" if "F-score" in title else f"{value:.1%}"
                axis.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          label, ha="center", va="bottom", fontsize=8)
    axes[5].axis("off")
    fig.suptitle("SimpleQA: Gemma 4 12B quantization × system-prompt comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simpleqa_primary_metrics.png", dpi=180, bbox_inches="tight")
    plt.show()

    counts = metrics.loc[configs, ["Correct", "Incorrect", "Abstained"]].copy()
    counts.index = labels
    fig, axis = plt.subplots(figsize=(11, 5))
    counts.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title("SimpleQA Outcome Counts by Quantization and System Prompt")
    axis.set_xlabel("Condition")
    axis.set_ylabel("Number of questions")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Outcome")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simpleqa_outcome_counts.png", dpi=180, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for axis, column, title, ylabel in [
        (axes[0], "Mean response words", "Mean Response Length", "Words"),
        (axes[1], "Mean latency (s)", "Mean Generation Latency", "Seconds per question"),
    ]:
        axis.bar(range(len(configs)), metrics.loc[configs, column].to_numpy(dtype=float), color=colors)
        axis.set_xticks(range(len(configs)))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simpleqa_efficiency.png", dpi=180, bbox_inches="tight")
    plt.show()


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_latex(metrics, configs, n_questions):
    rows = [
        ("Questions", "N"), ("Correct", "Correct"), ("Incorrect", "Incorrect"),
        ("Abstained", "Abstained"), ("Overall accuracy", "Overall accuracy"),
        ("Accuracy among attempted", "Attempted accuracy"),
        ("Hallucination rate (attempted)", "Hallucination rate among attempted"),
        ("Coverage", "Coverage"), ("Abstention rate", "Abstention rate"),
        ("SimpleQA F-score", "SimpleQA F-score"),
        ("Unsupported-claim rate", "Unsupported-claim rate"),
        ("Invalid-response rate", "Invalid-response rate"),
        ("Truncated-response rate", "Truncated rate"),
        ("Mean response length (words)", "Mean response words"),
        ("Mean latency (s/question)", "Mean latency (s)"),
    ]
    groups = []
    for config in configs:
        variant, _ = split_cfg(config)
        if groups and groups[-1][0] == variant:
            groups[-1][1] += 1
        else:
            groups.append([variant, 1])

    def fmt(column, value):
        if pd.isna(value):
            return "--"
        if column in PERCENT_METRICS:
            return f"{100 * value:.2f}\\%"
        if column == "SimpleQA F-score":
            return f"{value:.3f}"
        if column in {"Mean response words", "Mean latency (s)"}:
            return f"{value:.2f}"
        return str(int(value))

    start, cmid = 2, []
    for _, count in groups:
        cmid.append(rf"\cmidrule(lr){{{start}-{start + count - 1}}}")
        start += count
    methods = "; ".join(
        f"{VARIANT_LABELS[variant]}: {latex_escape(VARIANT_SPECS[variant]['desc'])}"
        for variant, _ in groups
    )
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{Gemma 4 12B factuality on SimpleQA ({n_questions} questions) across weight "
        rf"precisions and system prompts. Correctness labels are assigned by "
        rf"{latex_escape(OPENROUTER_MODEL.split('/')[-1].split(':')[0])}.}}",
        r"\label{tab:simpleqa_quant_comparison}", r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l " + " ".join("r" * count for _, count in groups) + "}",
        r"\toprule",
        "Metric & " + " & ".join(
            rf"\multicolumn{{{count}}}{{c}}{{{VARIANT_LABELS[variant]}}}"
            for variant, count in groups
        ) + r" \\",
        "".join(cmid),
        " & " + " & ".join(
            CONDITION_LABELS[split_cfg(config)[1]] for config in configs
        ) + r" \\",
        r"\midrule",
    ]
    for display_name, column in rows:
        values = [fmt(column, metrics.loc[config, column]) for config in configs]
        lines.append(f"{latex_escape(display_name)} & " + " & ".join(values) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\vspace{1mm}", r"\begin{minipage}{0.96\linewidth}", r"\footnotesize",
        r"Hallucination rate is the fraction of attempted answers labeled incorrect. Coverage is "
        r"the fraction of all questions attempted. Overall accuracy counts correct answers over all "
        r"questions, including abstentions. The SimpleQA F-score is the harmonic mean of overall "
        r"accuracy and accuracy among attempted answers. Unsupported-claim rate and correctness "
        r"are automatic judge estimates and require audit. Quantization methods -- " + methods + ".",
        r"\end{minipage}", r"\end{table*}",
    ]
    return "\n".join(lines)


def report(records, judge_checkpoint, generation_checkpoint, configs):
    details = build_details(
        records, generation_checkpoint["records"], judge_checkpoint["judgments"], configs
    )
    details.to_csv(JUDGE_DETAIL_CSV, index=False)
    metrics = pd.DataFrame(
        [calculate_metrics(details[details["config"] == config]) for config in configs],
        index=configs,
    )
    metrics.index.name = "config"
    metrics.to_csv(METRICS_CSV)

    print("\n================ AGGREGATE RESULTS ================")
    shown = metrics.copy()
    for column in PERCENT_METRICS:
        shown[column] = shown[column].map(lambda value: f"{100*value:.2f}%" if pd.notna(value) else "N/A")
    print(shown.round(4).to_string())

    unscored = int(metrics["Unscored"].sum())
    if unscored:
        print(f"\nWARNING: {unscored} response(s) unscored. Re-run to retry judge failures.")

    print("\n================ PAIRED 95% BOOTSTRAP CIs ================")
    ci, differences = paired_bootstrap(details, configs)
    ci_rows = []
    for metric in CI_METRICS:
        print("\n" + metric)
        for config in configs:
            estimate, low, high = ci[metric][config]
            print(
                f"  {config.replace('__', '/'):20s}: "
                + (f"{estimate:.4f} [{low:.4f}, {high:.4f}]"
                   if np.isfinite(estimate) and np.isfinite(low) else "N/A")
            )
            ci_rows.append({
                "metric": metric, "config": config, "estimate": estimate,
                "ci95_lower": low, "ci95_upper": high,
            })
    pd.DataFrame(ci_rows).to_csv(OUTPUT_DIR / "confidence_intervals.csv", index=False)

    print("\n================ PAIRED DIFFERENCES ================")
    print(differences.round(4).to_string(index=False))
    differences.to_csv(OUTPUT_DIR / "paired_bootstrap_differences.csv", index=False)

    make_plots(metrics, configs)
    latex = build_latex(metrics, configs, len(records))
    LATEX_FILE.write_text(latex, encoding="utf-8")
    print("\n================ LATEX TABLE ================\n")
    print(latex)

    print("\n================ EXPERIMENT COMPLETE ================")
    print("Base model:", BASE_MODEL_ID, "| Judge:", OPENROUTER_MODEL)
    print("Variants:", {v: VARIANT_SPECS[v]["desc"] for v in VARIANTS})
    print("Questions:", len(records), "| Conditions:", ", ".join(CONDITIONS))
    print("Configs reported:", [config.replace("__", "/") for config in configs])
    failed = generation_checkpoint.get("failed_variants", {})
    if failed:
        print("\nVariants that failed loading/preflight:")
        for variant, message in failed.items():
            print(f"  {variant}: {message[:300]}")
    print("\nFiles written to:", OUTPUT_DIR.resolve())
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name:45s} {path.stat().st_size / 1024:.1f} KiB")
    print("""
INTERPRETATION NOTES
1. Lower hallucination rate is not sufficient alone; compare coverage and overall accuracy.
2. Baseline omits the system turn entirely. Thinking is disabled for every run.
3. Bootstrap CIs resample paired question IDs; they do not measure judge uncertainty.
4. The judge is an LLM. Manually audit incorrect, abstained, and unsupported-claim cases.
5. Inspect truncated rate; heavy truncation suggests MAX_NEW_TOKENS is too low.
6. FP8/FP4 methods are the methods specified in the result-table footnote.
""")


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    global processor, TOK
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. In Colab select Runtime > Change runtime type > GPU."
        )
    print(
        "GPU:", torch.cuda.get_device_name(0),
        "| compute capability:", torch.cuda.get_device_capability(0),
        "| VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    )
    print("PyTorch:", torch.__version__, "| Transformers:", transformers.__version__)
    PAUSE.install()
    try:
        records = load_records()
        hf_token = get_secret("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        client = None
        if not PREFLIGHT_ONLY:
            api_key = get_secret("OPENROUTER_API_KEY", "Enter your OpenRouter API key: ")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is required for the judge.")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                max_retries=0,
                timeout=180.0,
            )

        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE
        )
        TOK = getattr(processor, "tokenizer", processor)
        print("Processor:", type(processor).__name__, "| tokenizer:", type(TOK).__name__)

        prompt_cache = run_static_checks(records)
        fingerprint = experiment_fingerprint()
        print("Experiment fingerprint:", fingerprint)
        generation_checkpoint = init_gen_checkpoint(records, fingerprint)
        PAUSE.check()
        generation_checkpoint = run_generation(records, prompt_cache, generation_checkpoint)
        save_json_atomic(GEN_CHECKPOINT, generation_checkpoint)

        if PREFLIGHT_ONLY:
            print("\nPREFLIGHT_ONLY=True: checks finished; benchmark not run.")
            return

        configs = completed_configs(generation_checkpoint, records)
        if not configs:
            print("\nNo configuration completed generation; nothing to judge.")
            return
        missing = [config for config in ALL_CONFIGS if config not in configs]
        if missing:
            print(
                "\nWARNING: incomplete configs excluded from judging/results:",
                [config.replace("__", "/") for config in missing],
            )
        judge_checkpoint = run_judge(
            client, records, generation_checkpoint["records"], configs, fingerprint
        )
        report(records, judge_checkpoint, generation_checkpoint, configs)
    except PauseRequested:
        print("\n[PAUSED] Progress is checkpointed. Re-run this script to resume.")
    except KeyboardInterrupt:
        print("\n[STOPPED] Immediate stop. Re-run this script to resume from checkpoints.")
    finally:
        PAUSE.restore()
        unload_model()


if __name__ == "__main__":
    main()
