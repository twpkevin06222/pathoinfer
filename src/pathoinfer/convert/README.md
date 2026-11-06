# pathoinfer - convert

Turns a full-precision model folder into faster variants for deployment: FP16, INT8 or
FP8 ONNX and TensorRT plans for tile models, and FP8 or 4-bit vision-language models
(VLMs) for vLLM. Every step writes a **new** model folder
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
pip install "pathoinfer[vlm-convert,vllm-client]"  # VLMs: llm-compressor, PEFT, httpx
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

## Vision-language models (`vlm.py`, `vlm_parity.py`)

For Qwen3-VL and SmolVLM fine-tuned with TRL + PEFT. pathoinfer never imports vllm: it
writes folders that `vllm serve` loads and talks to vLLM servers over HTTP. Run vLLM in
its own environment; it pins its own torch and transformers. When to choose `fp8`,
`w4a16-awq` or `w4a16-gptq`, and measurements on Qwen3-VL-2B, are in
[docs/vlm_quantisation.md](../../../docs/vlm_quantisation.md).

```bash
pathoinfer convert vlm-merge Qwen/Qwen3-VL-2B-Instruct runs/lora-final models/pvqa-bf16 \
    --card card.yaml
pathoinfer convert vlm models/pvqa-bf16 models/pvqa-fp8 --config vlm.yaml
# with `vllm serve models/pvqa-bf16 --port 8000` and `vllm serve models/pvqa-fp8 --port 8001` running:
pathoinfer convert vlm-parity models/pvqa-bf16 models/pvqa-fp8 models/pvqa-fp8-blackwell \
    --config vlm_parity.yaml \
    --reference-url http://localhost:8000 --candidate-url http://localhost:8001 \
    [--register pvqa --alias fp8-blackwell] [--tracking-uri URI] [--report parity.json]
```

### Merge (`vlm-merge`)

`merge_lora(base, adapter, out, card)` loads the base model (a local folder or Hugging
Face Hub id) in bf16, applies the PEFT adapter, merges it into the weights and saves a
safetensors folder with the base model's processor. `--card` gives the card fields the
weights cannot:

```yaml
# card.yaml
name: pvqa
version: "1"
input: {tile_size: 448, mpp: 0.5}   # how slide crops were made for fine-tuning
output: {grounding: false}           # true if answers contain boxes
```

The folder gets a `VqaCard` (task `vqa`, framework `hf`, file `config.json`) with
`architecture` from the model config, `precision: bf16` and the base model as
`parent_model`. `registry.loader.load_vlm` reads it; `load_model` refuses it, because
tile code paths cannot run a VLM.

### Quantise (`vlm`)

```yaml
# vlm.yaml (VlmQuantConfig)
scheme: w4a16-gptq                   # fp8, w4a16-awq or w4a16-gptq
calibration:                         # VqaDataConfig; needed for the w4a16 schemes
  path: data/calibration/pairs.jsonl # .jsonl, .csv or .parquet: image, question (, answer)
  samples: 256                       # seeded random subset; null for every row
  seed: 0
max_seq_len: 2048
extra_ignore: []                     # more module patterns to keep in bf16
```

`quantise_vlm(source, out, cfg)` runs one llm-compressor recipe on the **language model
only**:

| Scheme | Recipe | Calibration | Card `precision` |
|---|---|---|---|
| `fp8` | `QuantizationModifier`, `FP8_DYNAMIC` (W8A8, per-token dynamic activations) | none | `fp8` |
| `w4a16-awq` | `AWQModifier` (smoothing) then `QuantizationModifier`, `W4A16_ASYM` | image + question pairs | `int4` |
| `w4a16-gptq` | `GPTQModifier`, `W4A16` | image + question pairs | `int4` |

- **Ignore list** (`ignore_list`). Built from the model's module names: every child of
  `model` except the language model (`model.language_model` in Qwen3-VL,
  `model.text_model` in SmolVLM), plus `lm_head`. It is checked before quantising: each
  pattern must match a Linear layer, and every quantised layer must sit inside the
  language model. Qwen3-VL keeps `model.visual` (blocks, `merger`,
  `deepstack_merger_list`); SmolVLM keeps `model.vision_model` and `model.connector`.
- **AWQ mappings** (`awq_mappings`). llm-compressor 0.15 has none for the VLM classes, so
  pathoinfer uses the Qwen3/Llama decoder pairs (`input_layernorm` -> q/k/v,
  `v_proj` -> `o_proj`, `post_attention_layernorm` -> gate/up, `up_proj` -> `down_proj`),
  anchored inside the language model so the similarly named vision layers are left
  alone. Each must match a module, or the run stops.
- **Calibration** (`load_pairs`). Rows are pathology crops made the way the model saw
  them in fine-tuning, with questions (and answers, which are appended as the assistant
  turn). Images are read with OpenCV. Each pair goes through the model's processor and
  chat template, one sample at a time, and the decoder layers are calibrated in sequence.
  The calibration-set hash covers each image's content and question, so renaming a file
  does not change it, but changing its pixels does.

The output folder holds compressed-tensors safetensors (`float-quantized` for FP8,
`pack-quantized` for W4A16), the processor and a card with `precision`, version suffix
(`+fp8`, `+w4a16-awq`, `+w4a16-gptq`), `parent_model`, `calibration_set_hash` (null for
FP8) and `source_metadata.quantisation` (scheme, ignore patterns, number of quantised
layers).

### Parity (`vlm-parity`)

```yaml
# vlm_parity.yaml (VlmParityConfig)
eval:
  path: data/pathvqa_eval/pairs.jsonl  # image, question, answer (, box in image pixels)
  samples: null
tolerances:
  accuracy_drop_max: 0.02
  iou_drop_max: 0.05
max_tokens: 64
concurrency: 8                          # questions in flight per server
timeout_s: 120
```

`gate(reference, candidate, cfg, out, reference_url, candidate_url)` asks both servers
every eval question greedily (temperature 0) through `runtime.remote.vllm.VllmClient`,
an OpenAI-compatible client that sends each crop as a data URL. It scores:

- **accuracy**: exact match with the true answer after `normalise` (boxes in `[...]`
  dropped, lower case, punctuation and a/an/the removed), as PathVQA scores answers;
- **IoU**, for grounding cards on rows with a `box`: `parse_box` takes the first box
  that `runtime.vlm.parse_boxes` reads (Qwen-VL's `{"bbox_2d": [...]}` JSON, else a
  bracketed box of four numbers) in the card's `box_format` and `box_units`
  (`per_mille`, `normalised` or `pixel`), as image pixels clipped to the image.

Metrics are `accuracy_reference`, `accuracy`, `accuracy_drop` and, for grounding,
`iou_reference`, `iou`, `iou_drop`. The candidate passes when each drop is within its
tolerance. A pass copies it to the new folder `out` with `provenance.parity` (reference,
eval-set hash, metrics, tolerances, time) and, with `--register NAME --alias ALIAS`,
registers it in MLflow. A fail lists up to five questions the reference answered right
and the candidate wrong, exits 1, and writes and registers nothing.

`check(reference, candidate, cfg, ask_reference, ask_candidate)` takes any two
`(image, question) -> answer` functions, so parity can also run against other servers.

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

`tests/convert/test_vlm.py` (needs transformers; recipes also llm-compressor, merging
PEFT): on tiny random Qwen3-VL and SmolVLM models, the ignore list keeps exactly the
vision encoder, projector and `lm_head`; AWQ mappings stay inside the language model;
the three recipes build and llm-compressor accepts them; a LoRA merge equals PEFT's own
merge and writes a valid card; quantised cards record precision, parent and calibration
hash; calibration pairs are sampled and hashed by content; answers normalise, boxes parse
in every unit; parity passes, fails on accuracy and on box drift, and names the
regressed questions; and the vLLM client sends the image and greedy settings.

`tests/convert/test_vlm_full.py` (markers `gpu` and `slow`): quantises a real VLM with
each scheme and gates it through two vLLM servers. Set `PATHOINFER_VLM_MODEL` (a bf16
folder with a VqaCard), `PATHOINFER_VLM_PAIRS` and, for parity,
`PATHOINFER_VLM_CANDIDATE`, `PATHOINFER_VLLM_REFERENCE_URL` and
`PATHOINFER_VLLM_CANDIDATE_URL`; otherwise the tests skip.

`tests/convert/test_nvidia.py` (needs ModelOpt; plans also need TensorRT and a GPU):
ModelOpt INT8 on a 32-channel network; FP16 plans build, come from the cache the second
time and pass parity; ModelOpt INT8 and FP8 plans run through parity; and ONNX Runtime
quantisation still works after ModelOpt in the same process.
