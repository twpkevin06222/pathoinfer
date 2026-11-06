"""Write a Triton model repository from pathoinfer model folders.

For each model folder (ONNX or a TensorRT plan) it writes::

    REPOSITORY/<card name>/config.pbtxt
    REPOSITORY/<card name>/1/model.onnx | model.plan
    REPOSITORY/<card name>/model_card.json        (for reference; Triton ignores it)

``config.pbtxt`` is built as Triton's own ``ModelConfig`` protobuf message
(``tritonclient.grpc.model_config_pb2``) and written in its text format, so it only
holds fields Triton knows. The input comes from the card (name, dtype, layout and tile
size, without the batch axis); for ONNX models the outputs (names, dtypes, shapes, ``-1``
for dynamic axes) come from the graph. For plans the outputs are left to Triton's
auto-complete, which reads them from the plan. Dynamic batching and instances come from
``TritonRepoConfig``. Clients name the model by the card's name, which is what
``TritonRuntime`` uses by default.
"""

import os
import shutil
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.convert.common import CARD_FILE
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import ModelCard

PLATFORMS = {
    "onnx": ("onnxruntime_onnx", "model.onnx"),
    "tensorrt": ("tensorrt_plan", "model.plan"),
}
CARD_DTYPES = {"float32": "TYPE_FP32", "float16": "TYPE_FP16", "uint8": "TYPE_UINT8"}
ONNX_DTYPES = {
    "tensor(float)": "TYPE_FP32",
    "tensor(float16)": "TYPE_FP16",
    "tensor(bfloat16)": "TYPE_BF16",
    "tensor(int64)": "TYPE_INT64",
    "tensor(int32)": "TYPE_INT32",
    "tensor(uint8)": "TYPE_UINT8",
    "tensor(bool)": "TYPE_BOOL",
}


class TritonRepoConfig(BaseModel):
    """How Triton serves each model. ``max_batch_size`` must be at least the batch size
    clients send (``runtime.batch_size``). ``kind`` ``auto`` lets Triton place instances
    (GPU when there is one); ``gpu`` or ``cpu`` forces it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_batch_size: int = Field(32, gt=0)
    preferred_batch_sizes: list[int] = [8, 16, 32]
    max_queue_delay_us: int = Field(2000, ge=0)
    instances: int = Field(1, gt=0)
    kind: str = Field("auto", pattern="^(auto|gpu|cpu)$")

    @model_validator(mode="after")
    def _preferred_within_max(self) -> "TritonRepoConfig":
        too_big = [b for b in self.preferred_batch_sizes if b > self.max_batch_size]
        if too_big:
            raise ValueError(f"preferred batch sizes {too_big} exceed max_batch_size")
        return self


@log_call
def write_repository(
    sources: list[str | Path], repository: Path, cfg: TritonRepoConfig | None = None
) -> list[Path]:
    """Add each model source to ``repository`` (created if missing); returns the model
    directories. A model already in the repository is replaced."""
    cfg = cfg or TritonRepoConfig()
    written = []
    for source in sources:
        card, folder = load_model(str(source))
        written.append(_write_model(card, folder, repository, cfg))
    return written


def model_config_text(card: ModelCard, folder: Path, cfg: TritonRepoConfig) -> str:
    """``config.pbtxt`` for ``card``'s model file in ``folder``."""
    pb = require("tritonclient.grpc.model_config_pb2", extra("triton"))
    text_format = require("google.protobuf.text_format", extra("triton"))
    if card.framework not in PLATFORMS:
        raise ValueError(f"Triton serves ONNX models and TensorRT plans, not {card.framework}")
    platform, _ = PLATFORMS[card.framework]
    config = pb.ModelConfig(name=card.name, platform=platform, max_batch_size=cfg.max_batch_size)
    size = card.input.tile_size
    dims = [3, size, size] if card.input.layout == "NCHW" else [size, size, 3]
    config.input.append(
        pb.ModelInput(
            name=card.input.name,
            data_type=pb.DataType.Value(CARD_DTYPES[card.input.dtype]),
            dims=dims,
        )
    )
    if card.framework == "onnx":
        from pathoinfer.registry.onnx_info import read_onnx

        info = read_onnx(folder / card.file)
        for name in card.output.tensor_names():
            shape = info.outputs[name][1:]  # without the batch axis
            config.output.append(
                pb.ModelOutput(
                    name=name,
                    data_type=pb.DataType.Value(ONNX_DTYPES[info.output_types[name]]),
                    dims=[s if isinstance(s, int) else -1 for s in shape],
                )
            )
    config.dynamic_batching.CopyFrom(
        pb.ModelDynamicBatching(
            preferred_batch_size=cfg.preferred_batch_sizes,
            max_queue_delay_microseconds=cfg.max_queue_delay_us,
        )
    )
    group = pb.ModelInstanceGroup(count=cfg.instances)
    if cfg.kind != "auto":
        group.kind = pb.ModelInstanceGroup.Kind.Value(f"KIND_{cfg.kind.upper()}")
    config.instance_group.append(group)
    return str(text_format.MessageToString(config))


def _write_model(card: ModelCard, folder: Path, repository: Path, cfg: TritonRepoConfig) -> Path:
    text = model_config_text(card, folder, cfg)
    _, file = PLATFORMS[card.framework]
    target = repository / card.name
    partial = repository / f".{card.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "1").mkdir(parents=True)
    shutil.copyfile(folder / card.file, partial / "1" / file)
    (partial / "config.pbtxt").write_text(text)
    (partial / CARD_FILE).write_text(card.model_dump_json(indent=2) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(partial, target)
    logger.info("wrote Triton model {} ({})", target, PLATFORMS[card.framework][0])
    return target
