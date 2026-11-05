"""Merge LoRA adapters into safetensors and quantise VLM weights (llm-compressor). Lands after
v0.3.
"""

from pathlib import Path

from pathoinfer._extras import extra, require


def quantise_vlm(folder: Path, out: Path) -> Path:
    require("llmcompressor", extra("convert"))
    raise NotImplementedError("lands after v0.3")
