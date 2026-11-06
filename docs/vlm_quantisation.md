# Quantising vision-language models

How pathoinfer turns a vision-language model (VLM) fine-tuned with TRL + PEFT, such as
Qwen3-VL or SmolVLM, into a smaller model that vLLM serves, and how to choose a scheme.
pathoinfer never imports vllm: it writes folders vLLM loads and talks to vLLM servers
over HTTP.

Install `pathoinfer[vlm-convert]` (llm-compressor and PEFT) and, for parity,
`pathoinfer[vllm-client]` (httpx). Run vLLM itself in its own environment.

## Workflow

```bash
# 1. Fold the LoRA adapters into the base weights: a bf16 folder with a VqaCard.
pathoinfer convert vlm-merge Qwen/Qwen3-VL-2B-Instruct runs/lora-final models/pvqa-bf16 \
    --card card.yaml

# 2. Quantise the language model.
pathoinfer convert vlm models/pvqa-bf16 models/pvqa-fp8 --config vlm.yaml

# 3. Serve both and gate the quantised one against bf16 on a fixed eval set.
vllm serve models/pvqa-bf16 --port 8000 &
vllm serve models/pvqa-fp8 --port 8001 &
pathoinfer convert vlm-parity models/pvqa-bf16 models/pvqa-fp8 models/pvqa-fp8-blackwell \
    --config vlm_parity.yaml \
    --reference-url http://localhost:8000 --candidate-url http://localhost:8001 \
    --register pvqa --alias fp8-blackwell
```

```yaml
# card.yaml: what the weights cannot say
name: pvqa
version: "1"
input: {tile_size: 448, mpp: 0.5}   # how slide crops were made for fine-tuning
output: {grounding: false}           # true if answers contain boxes
```

```yaml
# vlm.yaml
scheme: w4a16-awq                    # fp8, w4a16-awq or w4a16-gptq
calibration:                         # needed for the w4a16 schemes
  path: data/calibration/pairs.jsonl # image, question (, answer) per row
  samples: 256
max_seq_len: 2048
```

```yaml
# vlm_parity.yaml
eval:
  path: data/pathvqa_eval/pairs.jsonl  # image, question, answer (, box) per row
  samples: null                         # every row
tolerances:
  accuracy_drop_max: 0.02               # PathVQA accuracy may drop by 2 points
  iou_drop_max: 0.05                    # grounding models: mean box IoU
```

Image + question tables (`.jsonl`, `.csv` or `.parquet`) have an `image` column (a path
relative to the table) and `question`, plus `answer` and optionally `box`
(`[x0, y0, x1, y1]` in image pixels) for evaluation. Use pathology crops made the way the
model saw them in fine-tuning (the card's `tile_size` and `mpp`), not generic images.

## What gets quantised

Only the **language model**. The vision encoder, the merger or projector that maps image
features into the language model, and `lm_head` stay bf16:

- they are a small share of the weights (Qwen3-VL-2B: language model 1.72 B parameters,
  81%; vision encoder and mergers 0.41 B, 19%), so quantising them saves little;
- they are the most sensitive part for pathology: the vision features carry the
  morphology, and errors there reach every answer.

The ignore list is built from the model's own module names, not written by hand. Every
child of `model` except the language model (`model.language_model` in Qwen3-VL,
`model.text_model` in SmolVLM) is ignored, plus `lm_head`, and the result is checked:
each pattern must match a module, and every quantised Linear layer must sit inside the
language model. For Qwen3-VL that keeps `model.visual` (blocks, `merger`,
`deepstack_merger_list`); for SmolVLM `model.vision_model` and `model.connector`.

The card records the scheme, the ignore patterns, the number of quantised layers, the
parent model and, for calibrated schemes, a hash of exactly which images (by content)
and questions were used.

## Choosing a scheme

| Scheme | Weights | Activations | Calibration | GPUs (vLLM) | Pick it when |
|---|---|---|---|---|---|
| `fp8` | FP8 | FP8, per token, dynamic | none | Hopper, Blackwell (Ada too); older GPUs run it weight-only | First choice on Hopper and Blackwell |
| `w4a16-awq` | INT4 | bf16 | a few hundred pairs | Ampere and newer | Memory-bound serving: small batches, long contexts, a smaller GPU |
| `w4a16-gptq` | INT4 | bf16 | a few hundred pairs | Ampere and newer | 4-bit when AWQ fails parity, or the text model has no AWQ mappings |

**Start with FP8 on Hopper or Blackwell.** Both weights and activations are FP8, so
matrix multiplies run on FP8 tensor cores and large batches get faster, not only
smaller. Weights shrink to about half of bf16. Activation scales are computed per token
at run time, so there is no calibration set to get wrong. It is usually the closest to
bf16, and it is the fastest to produce (seconds for a 2B model).

**Use AWQ when memory, not compute, limits you.** With small batches (one user, one
slide at a time) decoding is limited by reading weights from memory, and 4-bit weights
are about a quarter of bf16. AWQ rescales channels by how strongly activations use them
before rounding, which protects the weights that matter. It needs calibration data and
mappings of which layers to rescale together; pathoinfer supplies the Qwen3/Llama
mappings, anchored inside the language model.

**Use GPTQ as the 4-bit fallback.** It corrects rounding error layer by layer from the
calibration data and needs no architecture-specific mappings, so it works on text models
AWQ does not cover. It is slower to run and more sensitive to the calibration set; on
some models it is a little more accurate than AWQ, on others less. Try it when AWQ fails
parity.

Whatever the scheme, the parity gate decides. A scheme that keeps accuracy on generic
benchmarks can still lose it on pathology questions.

## Parity

`convert vlm-parity` asks both servers every question of the eval set greedily
(temperature 0) and compares:

- **accuracy**: exact match with the reference answer after normalising (boxes in `[...]`
  dropped, lower case, punctuation and the articles a/an/the removed), as PathVQA scores
  answers;
- **box IoU**, for grounding models (`output.grounding`) on rows with a `box`: the first
  box in each answer, read in the card's `box_format` and `box_units` (Qwen-VL answers
  in per-mille coordinates, `{"bbox_2d": [x0, y0, x1, y1]}`).

The candidate passes when its accuracy is at most `accuracy_drop_max` below the bf16
model's and its mean IoU at most `iou_drop_max` below. A pass copies it to a new folder
whose card records `provenance.parity` (metrics, tolerances, eval-set hash) and, with
`--register NAME --alias ALIAS`, registers it in MLflow. A fail lists up to five
questions the bf16 model answered right and the candidate wrong, and writes and
registers nothing.

## Measured here

Qwen3-VL-2B-Instruct (bf16 weights 4.26 GB), one RTX 5090, 64 calibration crops from a
real slide (448 px at 0.5 µm/px), llm-compressor 0.15.0:

| Scheme | Time | Weights on disk | Format |
|---|---|---|---|
| bf16 (source) | | 4.26 GB | |
| `fp8` | 5 s | 2.85 GB | compressed-tensors `float-quantized` |
| `w4a16-gptq` | 24 s | 2.16 GB | compressed-tensors `pack-quantized` |
| `w4a16-awq` | 41 s | 2.17 GB | compressed-tensors `pack-quantized` |

Each run quantised 196 Linear layers (28 decoder layers x 7 projections); the vision
tower (24 blocks), the merger, the three deepstack mergers and `lm_head` stayed bf16.
Weights shrink less than 2x or 4x because the vision tower and embeddings stay bf16.

Each folder reloaded in transformers and answered the same four crops greedily. All
four agreed on the three closed questions (yes / hematoxylin / yes); on the open one
("Describe the tissue in one word") bf16 said "cancer", FP8 and AWQ "Cancer." (a match
after normalising) and GPTQ "Carcinoma" (a miss). That is the kind of drift the parity
gate measures over a real eval set; four questions on an untuned model say nothing about
accuracy. vLLM was not installed here, so serving these folders was not tested.
