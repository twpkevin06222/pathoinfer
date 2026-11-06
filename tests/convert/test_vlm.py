"""VLM conversion on tiny random Qwen3-VL and SmolVLM (Idefics3) models; full quantisation
runs on a real model are marked gpu and slow (test_vlm_full.py)."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import pytest
import yaml

from pathoinfer.config import VlmParityConfig, VlmQuantConfig, VqaDataConfig
from pathoinfer.convert import vlm, vlm_parity
from pathoinfer.registry.loader import CardError, load_model, load_vlm
from pathoinfer.registry.model_card import VqaCard, VqaOutput

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
CARD = {
    "name": "path-vqa",
    "version": "1",
    "input": {"tile_size": 448, "mpp": 0.5},
    "output": {"grounding": True},
}


def tiny_qwen3vl() -> Any:
    config = transformers.Qwen3VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 16,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 64,
        },
    )
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(config)


def tiny_smolvlm() -> Any:
    config = transformers.Idefics3Config(
        text_config={
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 256,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 64,
            "patch_size": 16,
        },
        scale_factor=2,
    )
    return transformers.Idefics3ForConditionalGeneration(config)


@pytest.mark.parametrize(
    ("build", "language", "patterns"),
    [
        (
            tiny_qwen3vl,
            "model.language_model",
            ["re:^model\\.visual\\.", "lm_head"],
        ),
        (
            tiny_smolvlm,
            "model.text_model",
            ["re:^model\\.vision_model\\.", "re:^model\\.connector\\.", "lm_head"],
        ),
    ],
)
def test_ignore_list_keeps_vision_and_projector_in_bf16(
    build: Any, language: str, patterns: list[str]
) -> None:
    model = build()
    ignore = vlm.ignore_list(model)
    assert ignore.language_model == language and ignore.patterns == patterns
    linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    assert sorted(ignore.quantised + ignore.kept) == sorted(linear)
    assert all(n.startswith(language + ".") for n in ignore.quantised)
    assert not any(n.startswith(language + ".") for n in ignore.kept if n != "lm_head")
    assert "lm_head" in ignore.kept


def test_qwen3vl_mergers_are_kept() -> None:
    kept = vlm.ignore_list(tiny_qwen3vl()).kept
    assert "model.visual.merger.linear_fc1" in kept
    assert "model.visual.deepstack_merger_list.0.linear_fc2" in kept


def test_ignore_list_rejects_patterns_that_match_nothing() -> None:
    with pytest.raises(ValueError, match="match no Linear"):
        vlm.ignore_list(tiny_qwen3vl(), ["re:^model\\.audio\\."])


@pytest.mark.parametrize(
    ("scheme", "modifier", "preset"),
    [
        ("fp8", "QuantizationModifier", "FP8_DYNAMIC"),
        ("w4a16-awq", "QuantizationModifier", "W4A16_ASYM"),
        ("w4a16-gptq", "GPTQModifier", "W4A16"),
    ],
)
def test_recipes(scheme: str, modifier: str, preset: str) -> None:
    model = tiny_qwen3vl()
    ignore = vlm.ignore_list(model)
    mappings = vlm.awq_mappings(model, ignore.language_model)
    parsed = yaml.safe_load(vlm.recipe_yaml(scheme, ignore.patterns, mappings))
    modifiers = parsed["quant_stage"]["quant_modifiers"]
    assert modifiers[modifier] == {
        "targets": ["Linear"],
        "scheme": preset,
        "ignore": ignore.patterns,
    }
    if scheme == "w4a16-awq":  # AWQ smooths first, then quantises
        assert list(modifiers) == ["AWQModifier", "QuantizationModifier"]
        assert modifiers["AWQModifier"]["mappings"] == mappings
    else:
        assert list(modifiers) == [modifier]


@pytest.mark.parametrize("build", [tiny_qwen3vl, tiny_smolvlm])
def test_awq_mappings_stay_inside_the_language_model(build: Any) -> None:
    model = build()
    language = vlm.ignore_list(model).language_model
    mappings = vlm.awq_mappings(model, language)
    names = [n for n, _ in model.named_modules()]
    for mapping in mappings:
        for pattern in [mapping["smooth_layer"], *mapping["balance_layers"]]:
            hits = [n for n in names if vlm._matches(pattern, n)]
            assert hits and all(n.startswith(language + ".") for n in hits), pattern
    with pytest.raises(ValueError, match="AWQ mapping"):
        vlm.awq_mappings(model, "model.visual" if build is tiny_qwen3vl else "model.vision_model")
    with pytest.raises(ValueError, match="needs AWQ mappings"):
        vlm.recipe("w4a16-awq", [])


@pytest.mark.parametrize("scheme", ["fp8", "w4a16-awq", "w4a16-gptq"])
def test_llmcompressor_accepts_the_recipes(scheme: str) -> None:
    recipe_module = pytest.importorskip("llmcompressor.recipe")
    model = tiny_qwen3vl()
    ignore = vlm.ignore_list(model)
    text = vlm.recipe_yaml(scheme, ignore.patterns, vlm.awq_mappings(model, ignore.language_model))
    modifiers = recipe_module.Recipe.create_instance(text).modifiers
    quantise = modifiers[-1]
    assert type(quantise).__name__ == vlm.SCHEMES[scheme]["modifier"]
    assert list(quantise.ignore) == ignore.patterns
    if scheme == "w4a16-awq":
        assert type(modifiers[0]).__name__ == "AWQModifier" and len(modifiers[0].mappings) == 4


def test_4bit_schemes_need_calibration() -> None:
    with pytest.raises(ValueError, match="calibration"):
        VlmQuantConfig(scheme="w4a16-gptq")
    assert VlmQuantConfig(scheme="fp8").calibration is None


class Processor:
    """Stands in for the processor a real base model folder has."""

    def save_pretrained(self, folder: Path) -> None:
        (Path(folder) / "processor_config.json").write_text("{}")


def test_merge_lora_folds_adapters_and_writes_a_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peft = pytest.importorskip("peft")
    base = tiny_qwen3vl().to(torch.bfloat16)
    base.save_pretrained(tmp_path / "base")
    lora = peft.get_peft_model(
        base,
        peft.LoraConfig(target_modules=["q_proj", "v_proj"], r=4, init_lora_weights=False),
    )
    lora.save_pretrained(tmp_path / "adapter")
    expected = lora.merge_and_unload().state_dict()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    card = vlm.merge_lora(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged", CARD)
    assert card.architecture == "Qwen3VLForConditionalGeneration" and card.precision == "bf16"
    assert card.provenance.parent_model == str(tmp_path / "base")
    from safetensors.torch import load_file

    weights: dict[str, Any] = {}
    for shard in (tmp_path / "merged").glob("*.safetensors"):
        weights |= load_file(shard)
    assert not any("lora" in k for k in weights)
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(weights[name], expected[name])
    original = transformers.Qwen3VLForConditionalGeneration.from_pretrained(tmp_path / "base")
    assert not torch.equal(weights[name], original.state_dict()[name].to(torch.bfloat16))
    loaded, folder = load_vlm(tmp_path / "merged")
    assert loaded == card and (folder / "processor_config.json").is_file()
    with pytest.raises(CardError, match="load_vlm"):
        load_model(tmp_path / "merged")
    with pytest.raises(FileExistsError):
        vlm.merge_lora(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged", CARD)


def bf16_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "bf16"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    card = VqaCard.model_validate(
        {"task": "vqa", "architecture": "Qwen3VLForConditionalGeneration", **CARD}
    )
    (folder / "model_card.json").write_text(card.model_dump_json())
    return folder


def test_quantised_card_records_precision_parent_and_calibration(tmp_path: Path) -> None:
    card, _ = load_vlm(bf16_folder(tmp_path))
    out = tmp_path / "q"
    out.mkdir()
    (out / "config.json").write_text("{}")
    variant = vlm.write_card(card, out, "int4", "abc123", {"scheme": "w4a16-gptq", "ignore": []})
    assert variant.precision == "int4" and variant.version == "1+w4a16-gptq"
    assert variant.provenance.parent_model == "path-vqa@1"
    assert variant.provenance.calibration_set_hash == "abc123"
    assert variant.provenance.source_metadata["quantisation"]["scheme"] == "w4a16-gptq"
    assert load_vlm(out)[0] == variant


def pairs(tmp_path: Path, n: int = 6, suffix: str = ".jsonl") -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        name = f"crop_{i}.png"
        cv2.imwrite(str(tmp_path / name), rng.integers(0, 255, (64, 80, 3), np.uint8))
        rows.append(
            {
                "image": name,
                "question": f"is there tumour in crop {i}?",
                "answer": "yes" if i % 2 else "no",
                "box": [10.0, 10.0, 40.0, 30.0],
            }
        )
    path = tmp_path / f"pairs{suffix}"
    table = pl.DataFrame(rows)
    {".jsonl": table.write_ndjson, ".parquet": table.write_parquet}[suffix](path)
    return path


@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
def test_calibration_pairs_are_sampled_and_hashed_by_content(tmp_path: Path, suffix: str) -> None:
    path = pairs(tmp_path, suffix=suffix)
    rows, digest = vlm.load_pairs(VqaDataConfig(path=path, samples=4, seed=1))
    assert len(rows) == 4 and all(Path(r["image"]).is_absolute() for r in rows)
    assert vlm.load_pairs(VqaDataConfig(path=path, samples=4, seed=1))[1] == digest
    assert vlm.load_pairs(VqaDataConfig(path=path, samples=4, seed=2))[1] != digest
    image = Path(rows[0]["image"])
    cv2.imwrite(str(image), np.zeros((64, 80, 3), np.uint8))  # same name, different pixels
    assert vlm.load_pairs(VqaDataConfig(path=path, samples=4, seed=1))[1] != digest


def test_pairs_need_image_and_question_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"image": "x.png"}) + "\n")
    with pytest.raises(ValueError, match="question"):
        vlm.load_pairs(VqaDataConfig(path=path))


def test_answers_are_compared_normalised() -> None:
    assert vlm_parity.normalise("Yes.") == vlm_parity.normalise("yes")
    assert vlm_parity.normalise("The Liver ") == "liver"
    assert vlm_parity.normalise("no") != vlm_parity.normalise("yes")


def test_boxes_are_parsed_in_the_cards_units() -> None:
    size = (200, 100)
    per_mille = VqaOutput(grounding=True)
    assert vlm_parity.parse_box('{"bbox_2d": [100, 200, 500, 600]}', per_mille, size) == [
        20.0,
        20.0,
        100.0,
        60.0,
    ]
    pixel = VqaOutput(grounding=True, box_units="pixel", box_format="cxcywh")
    assert vlm_parity.parse_box("box 50 50 20 10", pixel, size) == [40.0, 45.0, 60.0, 55.0]
    assert vlm_parity.parse_box("no box here", per_mille, size) is None
    assert vlm_parity.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert vlm_parity.iou(None, [0, 0, 10, 10]) == 0.0


def candidate_folder(tmp_path: Path, reference: Path) -> Path:
    card, _ = load_vlm(reference)
    out = tmp_path / "fp8"
    out.mkdir()
    (out / "config.json").write_text("{}")
    vlm.write_card(card, out, "fp8", None, {"scheme": "fp8", "ignore": []})
    return out


def oracle(path: Path, box: str = "[125, 156.25, 500, 468.75]") -> Any:
    """Answers every question right, with a per-mille box equal to the true one
    (``[10, 10, 40, 30]`` px on an 80 x 64 image)."""
    truth = {r["question"]: r["answer"] for r in pl.read_ndjson(path).to_dicts()}
    return lambda image, question: f"{truth[question].capitalize()}. {box}"


def test_parity_passes_within_tolerance_and_writes_the_record(tmp_path: Path) -> None:
    path = pairs(tmp_path)
    cfg = VlmParityConfig(eval=VqaDataConfig(path=path, samples=None))
    reference = bf16_folder(tmp_path)
    candidate = candidate_folder(tmp_path, reference)
    report = vlm_parity.check(reference, candidate, cfg, oracle(path), oracle(path))
    assert report.passed and report.metrics["accuracy"] == 1.0
    assert report.metrics["iou"] == pytest.approx(1.0)
    card = vlm_parity.write_passed(reference, candidate, report, tmp_path / "final")
    assert card.provenance.parity is not None
    assert card.provenance.parity.tile_set_hash == report.eval_hash
    assert card.provenance.parity.reference == "path-vqa@1"


def test_parity_fails_and_names_the_regressed_questions(tmp_path: Path) -> None:
    path = pairs(tmp_path)
    cfg = VlmParityConfig(eval=VqaDataConfig(path=path, samples=None))
    right = oracle(path)

    def wrong_on_tumour(image: str, question: str) -> str:
        return "no" if "crop 1" in question else right(image, question)

    reference = bf16_folder(tmp_path)
    report = vlm_parity.check(
        reference, candidate_folder(tmp_path, reference), cfg, right, wrong_on_tumour
    )
    assert not report.passed
    assert report.metrics["accuracy_drop"] == pytest.approx(1 / 6)
    assert [r["question"] for r in report.regressions] == ["is there tumour in crop 1?"]
    assert "FAIL" in report.summary()


def test_parity_fails_on_box_drift(tmp_path: Path) -> None:
    path = pairs(tmp_path)
    cfg = VlmParityConfig(eval=VqaDataConfig(path=path, samples=None))
    reference = bf16_folder(tmp_path)
    report = vlm_parity.check(
        reference,
        candidate_folder(tmp_path, reference),
        cfg,
        oracle(path),
        oracle(path, box="[250, 312.5, 625, 625]"),  # moved by 10 px right and down
    )
    assert not report.passed and report.failures == [
        f"iou_drop {report.metrics['iou_drop']:.4f} > 0.05"
    ]


def test_vllm_client_sends_the_image_and_greedy_settings(tmp_path: Path) -> None:
    httpx = pytest.importorskip("httpx")
    from pathoinfer.runtime.remote.vllm import VllmClient

    path = pairs(tmp_path, n=1)
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "path-vqa"}]})
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]})

    client = VllmClient("http://server", transport=httpx.MockTransport(handler))
    assert client.ask(tmp_path / "crop_0.png", "tumour?") == "yes"
    assert seen["model"] == "path-vqa" and seen["temperature"] == 0
    content = seen["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "tumour?"}
    assert path.is_file()
