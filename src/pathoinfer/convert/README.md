# pathoinfer - convert

Turns a full-precision model folder into faster variants for deployment: FP16, INT8 or
FP8 ONNX, and TensorRT plans for a specific GPU. Every step writes a **new** model folder
and never edits one in place. A variant is trusted only after `convert parity` has
compared it with its reference through the real stitchers and written it with the result
in its card.

Post-training quantisation (PTQ) only: quantisation-aware training belongs in training
(pathotrain).

## Install

```bash
pip install "pathoinfer[onnx,convert]"        # FP16, ONNX Runtime INT8, ModelOpt INT8/FP8
pip install "pathoinfer[onnx,convert,trt]"     # plus TensorRT plans (several GB)
pip install "pathoinfer[mlflow]"               # to register passed variants
```

`[convert]` lists the helpers NVIDIA ModelOpt's ONNX quantiser needs instead of
`nvidia-modelopt[onnx]`: that extra pins `onnxruntime-gpu<1.31` and `onnx~=1.21`, which
clash with the `onnxruntime-gpu>=1.31.1` the workspace uses. The listed versions are
verified with onnxruntime-gpu 1.31.1 and onnx 1.23.2 (`tests/convert`).

## Workflow

```bash
# 1. Convert (each writes a new folder).
pathoinfer convert fp16 models/tumour-seg models/tumour-seg-fp16
pathoinfer convert ptq  models/tumour-seg models/tumour-seg-int8 --config quant.yaml

# 2. Optionally build a plan for this GPU from a converted ONNX folder.
pathoinfer convert trt models/tumour-seg-fp16 models/tumour-seg-fp16-plan --batch 1 32 64

# 3. Gate: compare with the reference; on a pass write the final folder (and register it).
pathoinfer convert parity models/tumour-seg models/tumour-seg-fp16-plan \
    models/tumour-seg-fp16-blackwell --config parity.yaml \
    --register tumour-seg --alias fp16-blackwell --report parity.json
```

`convert parity` exits 0 on a pass and 1 on a fail. A fail writes no folder and
registers nothing.

Every converted card records `precision`, `version` (with a suffix such as `+fp16`,
`+int8` or `+trt-nvidia-geforce-rtx-5090-sm120`) and `provenance.parent_model`
(`name@version` of the source). Quantised cards also record
`provenance.calibration_set_hash` and the settings under
`provenance.source_metadata.quantisation`. Plans record the GPU, compute capability,
TensorRT version and profile under `provenance.source_metadata.tensorrt`.

## FP16 (`fp16.py`)

`pathoinfer convert fp16 SOURCE OUT [--fp16-io] [--keep-fp32 OP ...]`

ONNX Runtime's converter turns weights and activations into float16. Op types on its
default block list, plus `--keep-fp32`, stay float32 with Casts around them. Graph
inputs and outputs stay float32 unless you pass `--fp16-io`, in which case the card's
`input.dtype` becomes `float16`. The source must be an FP32 ONNX model.

## INT8 and FP8 (`ptq.py`)

`pathoinfer convert ptq SOURCE OUT --config quant.yaml`

```yaml
# quant.yaml
precision: int8               # int8 or fp8 (fp8 needs engine: modelopt)
engine: modelopt              # modelopt (TensorRT targets) or ort (runs in ONNX Runtime)
method: entropy               # entropy, minmax or percentile
percentile: 99.999            # for method: percentile
calibration:
  slides: calibration/*.svs   # path, glob, .txt/.csv/.parquet list, or a list of these
  tiles: 512                  # spread evenly over the slides; 500-1000 is typical
  seed: 0
keep_ops: [LayerNormalization, Softmax]   # op types kept in high precision
keep_nodes: ["*attn*", "*attention*"]     # node-name globs (lower case) kept too
keep_ends: true               # also keep the first and last Conv/MatMul/Gemm
```

Calibration tiles are read through patchology at the card's tile size and mpp (tissue
tiles, chosen at random with `seed`) and preprocessed exactly as at inference. List
slides that cover your scanners, stains and tissue types. The calibration-set hash
identifies exactly which tiles were read.

| Engine | Precisions | Kept layers run in | Use for |
|---|---|---|---|
| `modelopt` (NVIDIA ModelOpt) | INT8, FP8 | FP16 | TensorRT plans; FP8 on Hopper and Blackwell |
| `ort` (ONNX Runtime) | INT8 | FP32 | INT8 that must run in ONNX Runtime without TensorRT |

ModelOpt runs in a child process. While it quantises, it patches ONNX Runtime's
calibration classes for the rest of the process it runs in, which would break any later
ONNX Runtime quantisation. It also skips convolutions with very few channels, so a tiny
network may come out unquantised. During calibration ONNX Runtime may log that
`libnvinfer.so.10` is missing (its TensorRT provider wants TensorRT 10, `[trt]` installs
11); this is harmless, and calibration falls back to CUDA or CPU.

## TensorRT plans (`tensorrt.py`)

`pathoinfer convert trt SOURCE OUT [--batch MIN OPT MAX] [--workspace-gb 4] [--device 0]`

A plan is a serialised TensorRT engine for one GPU architecture and TensorRT version.
It is built as a strongly typed network, so its precision is the ONNX graph's: FP32,
FP16 from `convert fp16`, or INT8/FP8 from `convert ptq`. The plan therefore computes
what parity checks. One optimisation profile covers batch sizes `MIN` to `MAX`, tuned
for `OPT`. Runs must use a `batch_size` within it.

Built plans are cached in `~/.cache/pathoinfer/plans`, keyed by model hash, GPU,
compute capability, TensorRT version, precision and profile, so building the same plan
again copies the cached one. Plan folders (`framework: tensorrt`) run in
`pathoinfer run` like any other model, on the GPU they were built for.

## Parity gate (`parity.py`)

`pathoinfer convert parity REFERENCE CANDIDATE OUT --config parity.yaml [--register NAME --alias ALIAS] [--tracking-uri URI] [--report FILE]`

```yaml
# parity.yaml
slides: parity/*.svs          # slides held out from calibration
window_tiles: 6               # mini-slides: windows of 6 x 6 grid tiles around tissue
windows: 2                    # per slide
seed: 0
reference_device: auto        # the full-precision model
candidate: {device: cuda, batch_size: 16}
iou: 0.5                      # detection matching
tolerances:
  dice_min: 0.97              # segmentation: Dice per class, polygon level
  f1_min: 0.97                # detection: F1 per class
  count_diff_max: 0.03        # detection: |n_candidate - n_reference| / n_reference per class
  auc_min: 0.99               # classification: one-vs-rest AUC per class
segmentation: {output_mpp: 2.0}   # stitch settings, as in a run config
detection: {score_threshold: null}
```

Both models run on the same tiles: mini-slides, read once through patchology. Each
window's outputs go through the task's stitcher (cores, seams, polygons) for each model,
and the stitched results are compared per class, pooled over windows:

| Task | Metric | Pass when the worst class |
|---|---|---|
| Segmentation | Dice of the class's polygons, `2 \|A ∩ B\| / (\|A\| + \|B\|)` | `dice_min` >= tolerance |
| Detection | F1 of detections matched one to one by box IoU >= `iou` | `f1_min` >= tolerance |
| Detection | Relative count difference | `count_diff_max` <= tolerance |
| Classification | AUC of the candidate's core probabilities against the reference's argmax | `auc_min` >= tolerance |

On a pass, the candidate folder is copied to `OUT` and its card gains
`provenance.parity`: the reference (`name@version`), the tile-set hash, the metric
values, the tolerances and the time of the check. With `--register NAME --alias ALIAS`
the new folder is logged to MLflow, added as a version of `NAME`, and `ALIAS` (for
example `fp16-blackwell` or `int8-l4`) points at it. Load it later as
`models:/NAME@ALIAS`.

On a fail, the report lists the failing metrics, the worst class of each metric and,
when both models are ONNX, the five intermediate tensors whose values differ most from
the reference's. These are the layers to keep in high precision (`keep_ops`,
`keep_nodes`) before trying again. `--report` writes the report as JSON in both cases.

Choose tolerances for the model, not for the method. A class that covers a small share
of the slide amplifies a few flipped pixels into a large Dice drop: on a random test
network with 0.5% tumour pixels, INT8 logits within 5.5% of FP32, with 99.7% of pixels
agreeing, still gave a Dice of 0.67.

## Python

```python
from pathlib import Path

from pathoinfer.config import ParityConfig, QuantConfig, load_yaml
from pathoinfer.convert.fp16 import to_fp16
from pathoinfer.convert.parity import gate
from pathoinfer.convert.ptq import quantise
from pathoinfer.convert.tensorrt import build_plan

to_fp16("models/tumour-seg", Path("models/tumour-seg-fp16"))
quantise("models/tumour-seg", Path("models/tumour-seg-int8"), load_yaml("quant.yaml", QuantConfig))
build_plan("models/tumour-seg-fp16", Path("models/tumour-seg-fp16-plan"), max_batch=64)
report = gate(
    "models/tumour-seg",
    "models/tumour-seg-fp16-plan",
    load_yaml("parity.yaml", ParityConfig),
    Path("models/tumour-seg-fp16-blackwell"),
    register=("tumour-seg", "fp16-blackwell"),
)
print(report.summary())
```

## Testing

`tests/convert/test_convert.py` (needs `onnx`; runs in CI):

- the fixture model converts to FP16 and passes parity, and the passed card records it;
- a deliberately broken conversion (last layer negated) fails parity, names `logits` as
  the worst layer, writes nothing and registers nothing;
- ONNX Runtime INT8 writes Q/DQ nodes and a reproducible calibration hash;
- parity metrics: AUC, one-to-one detection matching, and F1, count difference and AUC
  through the real detection and heatmap stitchers;
- the CLI, and MLflow registration under an alias (with full MLflow).

`tests/convert/test_nvidia.py` (needs ModelOpt; plans also need TensorRT and a GPU):
ModelOpt INT8 on a 32-channel network; FP16 plans build, come from the cache the second
time and pass parity; ModelOpt INT8 and FP8 plans run through parity; and ONNX Runtime
quantisation still works after ModelOpt in the same process.
