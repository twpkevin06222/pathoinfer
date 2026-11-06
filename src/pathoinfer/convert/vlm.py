"""Vision-language models (Qwen3-VL, SmolVLM) fine-tuned with TRL + PEFT, for vLLM.

1. ``merge_lora``: fold LoRA adapters into the base weights and save a bf16 Hugging Face
   folder (safetensors) with its processor and a VqaCard.
2. ``quantise_vlm``: an llm-compressor recipe on the **language model only**. The ignore
   list is built from the model's own module names (``ignore_list``): every child of
   ``model`` except the language model (vision encoder, merger/projector) and ``lm_head``
   stay bf16, and the list is checked to cover exactly the non-language Linear layers.
   Schemes: ``fp8`` (W8A8, dynamic activations), ``w4a16-awq``, ``w4a16-gptq``; the
   4-bit ones calibrate on pathology image + question pairs. The output is
   compressed-tensors safetensors, which vLLM loads.
3. Parity against bf16 runs in ``convert.vlm_parity`` over HTTP. pathoinfer never
   imports vllm.

Heavy imports (transformers, peft, llmcompressor) are inside the functions, so this
module imports without the ``[vlm-convert]`` extra.
"""

import base64
import hashlib
import json
import os
import random
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import yaml
from loguru import logger
from numpy.typing import NDArray

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.config import VlmQuantConfig, VqaDataConfig
from pathoinfer.convert.common import CARD_FILE
from pathoinfer.registry.loader import load_vlm
from pathoinfer.registry.model_card import VqaCard

# Where the language model lives, by attribute of ``model.model``; everything else there
# (vision tower, merger/projector) is kept in bf16.
LANGUAGE_MODELS = ("language_model", "text_model")
SCHEMES: dict[str, dict[str, Any]] = {
    "fp8": {"modifier": "QuantizationModifier", "scheme": "FP8_DYNAMIC", "calibrated": False},
    "w4a16-awq": {"modifier": "QuantizationModifier", "scheme": "W4A16_ASYM", "calibrated": True},
    "w4a16-gptq": {"modifier": "GPTQModifier", "scheme": "W4A16", "calibrated": True},
}
# AWQ smoothing pairs (smooth layer -> balance layers) of Qwen3 and Llama decoder layers,
# the text models of Qwen3-VL and SmolVLM. llm-compressor has no mappings for the VLM
# classes, so these are anchored to the language model path: the vision tower has
# similarly named layers that must not be touched.
AWQ_PAIRS = [
    ("input_layernorm", ["q_proj", "k_proj", "v_proj"]),
    ("v_proj", ["o_proj"]),
    ("post_attention_layernorm", ["gate_proj", "up_proj"]),
    ("up_proj", ["down_proj"]),
]
PRECISIONS = {"fp8": "fp8", "w4a16-awq": "int4", "w4a16-gptq": "int4"}


@dataclass(frozen=True)
class IgnoreList:
    """Which Linear layers a recipe quantises and which it keeps in bf16."""

    language_model: str  # module path of the language model, e.g. model.language_model
    patterns: list[str]  # llm-compressor ``ignore`` entries
    quantised: list[str]  # Linear module names the recipe will quantise
    kept: list[str]  # Linear module names left in bf16


def ignore_list(model: Any, extra_patterns: list[str] | None = None) -> IgnoreList:
    """Ignore everything but the language model's Linear layers, verified against the
    model's module names.

    Raises:
        ValueError: no language model found, a pattern matches no module, or the
            patterns would quantise a non-language Linear or skip a language one.
    """
    import torch

    inner = model.model
    names = [n for n in LANGUAGE_MODELS if hasattr(inner, n)]
    if len(names) != 1:
        raise ValueError(
            f"cannot find the language model among {[n for n, _ in inner.named_children()]}"
        )
    language = f"model.{names[0]}"
    others = [n for n, child in inner.named_children() if n != names[0] and _has_linear(child)]
    patterns = [f"re:^model\\.{re.escape(n)}\\." for n in others] + [
        "lm_head",
        *(extra_patterns or []),
    ]
    linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    kept = [n for n in linear if any(_matches(p, n) for p in patterns)]
    quantised = [n for n in linear if n not in kept]
    unused = [p for p in patterns if not any(_matches(p, n) for n in linear)]
    if unused:
        raise ValueError(f"ignore patterns match no Linear layer: {unused}")
    outside = [n for n in quantised if not n.startswith(language + ".")]
    if outside:
        raise ValueError(f"would quantise layers outside the language model: {outside[:5]}")
    if not quantised:
        raise ValueError("nothing left to quantise")
    return IgnoreList(language, patterns, quantised, kept)


def awq_mappings(model: Any, language_model: str) -> list[dict[str, Any]]:
    """:data:`AWQ_PAIRS` as llm-compressor mappings inside ``language_model``, each
    checked to match modules of ``model``.

    Raises:
        ValueError: a mapping matches nothing (an unsupported text model).
    """
    names = [n for n, _ in model.named_modules()]
    prefix = "re:^" + re.escape(language_model) + r"\..*"
    mappings = []
    for smooth, balance in AWQ_PAIRS:
        smooth_layer = f"{prefix}{smooth}$"
        balance_layers = [f"{prefix}{b}$" for b in balance]
        for pattern in [smooth_layer, *balance_layers]:
            if not any(_matches(pattern, n) for n in names):
                raise ValueError(f"AWQ mapping {pattern} matches no module of {language_model}")
        mappings.append({"smooth_layer": smooth_layer, "balance_layers": balance_layers})
    return mappings


def recipe(
    scheme: str, ignore: list[str], mappings: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The llm-compressor recipe for ``scheme`` on Linear layers, minus ``ignore``.
    ``w4a16-awq`` smooths with ``mappings`` (:func:`awq_mappings`) before quantising."""
    spec = SCHEMES[scheme]
    modifiers: dict[str, Any] = {}
    if scheme == "w4a16-awq":
        if not mappings:
            raise ValueError("w4a16-awq needs AWQ mappings")
        modifiers["AWQModifier"] = {"mappings": mappings, "duo_scaling": True}
    modifiers[spec["modifier"]] = {
        "targets": ["Linear"],
        "scheme": spec["scheme"],
        "ignore": ignore,
    }
    return {"quant_stage": {"quant_modifiers": modifiers}}


def recipe_yaml(
    scheme: str, ignore: list[str], mappings: list[dict[str, Any]] | None = None
) -> str:
    """:func:`recipe` as the YAML text llm-compressor takes."""
    return str(yaml.safe_dump(recipe(scheme, ignore, mappings), sort_keys=False))


@log_call
def merge_lora(
    base: str | Path, adapter: str | Path, out: Path, card: Mapping[str, Any]
) -> VqaCard:
    """Merge the PEFT adapter at ``adapter`` into ``base`` (a local folder or Hub id) and
    write a bf16 model folder with its processor and a VqaCard. ``card`` gives the card
    fields the weights cannot (name, version, input, output)."""
    peft = require("peft", extra("vlm-convert"))
    transformers = require("transformers", extra("vlm-convert"))
    import torch

    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    model = transformers.AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    merged = peft.PeftModel.from_pretrained(model, adapter).merge_and_unload()
    processor = transformers.AutoProcessor.from_pretrained(base)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    merged.save_pretrained(partial, safe_serialization=True)
    processor.save_pretrained(partial)
    vqa = _card(
        card,
        architecture=merged.config.architectures[0],
        precision="bf16",
        provenance={"parent_model": str(base), "source_metadata": {"adapter": str(adapter)}},
    )
    (partial / CARD_FILE).write_text(vqa.model_dump_json(indent=2) + "\n")
    os.replace(partial, out)
    logger.info("merged {} into {}: {}", adapter, base, out)
    return vqa


@log_call
def quantise_vlm(source: str | Path, out: Path, cfg: VlmQuantConfig) -> VqaCard:
    """Quantise the bf16 VLM folder at ``source`` (with a VqaCard) into the new folder
    ``out``, written as compressed-tensors safetensors that vLLM loads."""
    llmcompressor = require("llmcompressor", extra("vlm-convert"))
    transformers = require("transformers", extra("vlm-convert"))
    import torch

    card, folder = load_vlm(source)
    if card.precision != "bf16":
        raise ValueError(f"expected a bf16 VLM, got {card.precision}")
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    model = transformers.AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.bfloat16)
    processor = transformers.AutoProcessor.from_pretrained(folder)
    ignore = ignore_list(model, cfg.extra_ignore)
    logger.info(
        "{}: quantising {} Linear layers of {}, keeping {} in bf16",
        card.name,
        len(ignore.quantised),
        ignore.language_model,
        len(ignore.kept),
    )
    kwargs: dict[str, Any] = {}
    calibration_hash = None
    if SCHEMES[cfg.scheme]["calibrated"]:
        assert cfg.calibration is not None
        rows, calibration_hash = load_pairs(cfg.calibration)
        kwargs = {
            "dataset": _calibration_dataset(rows, processor, cfg.max_seq_len),
            "data_collator": _collate,
            "num_calibration_samples": len(rows),
            "max_seq_length": cfg.max_seq_len,
            "sequential_targets": [_decoder_layer(model, ignore.language_model)],
        }
    mappings = awq_mappings(model, ignore.language_model) if cfg.scheme == "w4a16-awq" else None
    llmcompressor.oneshot(
        model=model,
        processor=processor,
        recipe=recipe_yaml(cfg.scheme, ignore.patterns, mappings),
        **kwargs,
    )
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    model.save_pretrained(partial, save_compressed=True)
    processor.save_pretrained(partial)
    variant = write_card(
        card,
        partial,
        PRECISIONS[cfg.scheme],
        calibration_hash,
        {
            "scheme": cfg.scheme,
            "ignore": ignore.patterns,
            "quantised_layers": len(ignore.quantised),
        },
    )
    os.replace(partial, out)
    logger.info("wrote {} ({} {}, {})", out, variant.name, variant.version, variant.precision)
    return variant


def write_card(
    card: VqaCard,
    folder: Path,
    precision: str,
    calibration_hash: str | None,
    settings: dict[str, Any],
) -> VqaCard:
    """Write the quantised variant's card into ``folder``: precision, parent model,
    calibration hash and the recipe settings; parity cleared."""
    data = card.model_dump(mode="json")
    data["precision"] = precision
    data["version"] = f"{card.version}+{settings['scheme']}"
    data["provenance"] = data["provenance"] | {
        "parent_model": f"{card.name}@{card.version}",
        "calibration_set_hash": calibration_hash,
        "parity": None,
        "source_metadata": {**card.provenance.source_metadata, "quantisation": settings},
    }
    variant = VqaCard.model_validate(data)
    (folder / CARD_FILE).write_text(variant.model_dump_json(indent=2) + "\n")
    return variant


def load_pairs(cfg: VqaDataConfig) -> tuple[list[dict[str, Any]], str]:
    """The rows of ``cfg.path`` (a seeded subset of ``cfg.samples``), with ``image``
    resolved to an absolute path, and a hash of exactly which images (by content) and
    questions they are."""
    path = Path(cfg.path)
    if path.suffix == ".jsonl":
        table = pl.read_ndjson(path)
    elif path.suffix == ".csv":
        table = pl.read_csv(path)
    elif path.suffix == ".parquet":
        table = pl.read_parquet(path)
    else:
        raise ValueError(f"{path}: expected .jsonl, .csv or .parquet")
    missing = {"image", "question"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    rows = table.to_dicts()
    if cfg.samples is not None and cfg.samples < len(rows):
        rows = random.Random(cfg.seed).sample(rows, cfg.samples)
    digest = hashlib.sha256()
    for row in rows:
        row["image"] = str((path.parent / row["image"]).resolve())
        image_hash = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        digest.update(json.dumps([image_hash, row["question"]]).encode())
    return rows, digest.hexdigest()[:16]


def read_rgb(path: str | Path) -> NDArray[np.uint8]:
    """An image file as an ``(H, W, 3)`` uint8 RGB array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def image_data_url(path: str | Path) -> str:
    """The image file as a data URL, as OpenAI-compatible servers take it."""
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _card(fields: Mapping[str, Any], **overrides: Any) -> VqaCard:
    data = {"task": "vqa", **fields, **overrides}
    return VqaCard.model_validate(data)


def _calibration_dataset(rows: list[dict[str, Any]], processor: Any, max_len: int) -> Any:
    datasets = require("datasets", extra("vlm-convert"))
    encoded = []
    for row in rows:
        content = [{"type": "image"}, {"type": "text", "text": row["question"]}]
        messages = [{"role": "user", "content": content}]
        if row.get("answer"):
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": str(row["answer"])}]}
            )
        text = processor.apply_chat_template(messages, add_generation_prompt=len(messages) == 1)
        image = read_rgb(row["image"])
        inputs = processor(text=[text], images=[image], max_length=max_len, truncation=False)
        # Kept exactly as the processor returns them (one sample): processors differ in
        # which outputs carry a batch axis (Qwen flattens patches, SmolVLM does not).
        encoded.append({k: _as_list(v) for k, v in inputs.items()})
    return datasets.Dataset.from_list(encoded)


def _as_list(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    assert len(batch) == 1, "VLM calibration runs one sample at a time"
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def _decoder_layer(model: Any, language_model: str) -> str:
    """The class name of the language model's decoder layers, for sequential calibration."""
    module = model.get_submodule(language_model)
    return type(module.layers[0]).__name__


def _has_linear(module: Any) -> bool:
    import torch

    return any(isinstance(m, torch.nn.Linear) for m in module.modules())


def _matches(pattern: str, name: str) -> bool:
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return name == pattern or name.endswith("." + pattern)
