"""Parity of a quantised VLM against its bf16 reference, through two vLLM servers.

Both models answer the same fixed eval set (``VqaDataConfig``: image, question, answer,
optional box) greedily. Metrics, as in PathVQA:

- accuracy: normalised exact match with ``answer`` (boxes in ``[...]`` dropped, lower
  case, punctuation and the articles a/an/the removed, spaces collapsed), so
  "Yes. [10, 20, 30, 40]" matches "yes";
- for grounding models (``output.grounding``) and rows with a ``box``: mean IoU of the
  first box in each answer with ``box``, converted from the card's box units.

The candidate passes when its accuracy is at most ``accuracy_drop_max`` below the
reference's and its mean IoU at most ``iou_drop_max`` below. A pass writes the candidate
as a new folder with ``provenance.parity`` and may register it in MLflow under an alias;
a fail lists the questions the reference answered right and the candidate wrong, and
writes nothing.
"""

import json
import os
import re
import shutil
import string
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from pathoinfer._logging import log_call
from pathoinfer.config import VlmParityConfig
from pathoinfer.convert.common import CARD_FILE
from pathoinfer.convert.parity import register_model
from pathoinfer.convert.vlm import load_pairs, read_rgb
from pathoinfer.registry.loader import load_vlm
from pathoinfer.registry.model_card import Parity, VqaCard, VqaOutput
from pathoinfer.runtime.vlm import Crop, parse_boxes

Ask = Callable[[str, str], str]  # (image path, question) -> answer
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")


@dataclass(frozen=True)
class VlmParityReport:
    passed: bool
    metrics: dict[str, float]
    tolerances: dict[str, float]
    failures: list[str]
    eval_hash: str
    samples: int
    regressions: list[dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"vlm parity {'PASS' if self.passed else 'FAIL'} on {self.samples} questions"]
        lines += [f"  {k} {v:.4f}" for k, v in self.metrics.items()]
        lines += [f"  failed: {f}" for f in self.failures]
        lines += [
            f"  regressed: {r['question']!r}: reference {r['reference']!r}, "
            f"candidate {r['candidate']!r}, truth {r['answer']!r}"
            for r in self.regressions
        ]
        return "\n".join(lines)


def normalise(answer: str) -> str:
    """The answer text for exact match: boxes (``[...]``) dropped, lower case, no
    punctuation or articles, single spaces."""
    text = _BRACKETS.sub(" ", str(answer)).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(_ARTICLES.sub(" ", text).split())


def parse_box(answer: str, output: VqaOutput, size: tuple[int, int]) -> list[float] | None:
    """The first box in ``answer`` as ``xyxy`` pixels of the image of ``size`` (width,
    height), read as ``runtime.vlm.parse_boxes`` does."""
    grounding = parse_boxes(answer, output, Crop(0, 0, size[0], size[1], size))
    return [float(v) for v in grounding.boxes[0]] if len(grounding) else None


def iou(a: list[float] | None, b: list[float]) -> float:
    if a is None:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def score(
    card: VqaCard, rows: list[dict[str, Any]], answers: list[str]
) -> tuple[float, float | None]:
    """Accuracy over ``rows`` and, for grounding models, mean IoU over rows with a box."""
    correct = [
        normalise(a) == normalise(str(r["answer"])) for r, a in zip(rows, answers, strict=True)
    ]
    boxed = [(r, a) for r, a in zip(rows, answers, strict=True) if r.get("box")]
    if not (card.output.grounding and boxed):
        return sum(correct) / len(correct), None
    ious = []
    for row, answer in boxed:
        height, width = read_rgb(row["image"]).shape[:2]
        box = parse_box(answer, card.output, (width, height))
        ious.append(iou(box, [float(v) for v in row["box"]]))
    return sum(correct) / len(correct), sum(ious) / len(ious)


def check(
    reference: str | Path,
    candidate: str | Path,
    cfg: VlmParityConfig,
    ask_reference: Ask,
    ask_candidate: Ask,
) -> VlmParityReport:
    """Ask both models every eval question and judge the candidate."""
    ref_card, _ = load_vlm(reference)
    cand_card, _ = load_vlm(candidate)
    if (ref_card.architecture, ref_card.output) != (cand_card.architecture, cand_card.output):
        raise ValueError("reference and candidate differ in architecture or output")
    rows, eval_hash = load_pairs(cfg.eval)
    if any(r.get("answer") is None for r in rows):
        raise ValueError(f"{cfg.eval.path}: every eval row needs an answer")
    ref_answers = _answers(ask_reference, rows, cfg.concurrency)
    cand_answers = _answers(ask_candidate, rows, cfg.concurrency)
    ref_acc, ref_iou = score(ref_card, rows, ref_answers)
    cand_acc, cand_iou = score(cand_card, rows, cand_answers)
    metrics = {
        "accuracy_reference": ref_acc,
        "accuracy": cand_acc,
        "accuracy_drop": ref_acc - cand_acc,
    }
    tolerances = {"accuracy_drop": cfg.tolerances.accuracy_drop_max}
    if ref_iou is not None and cand_iou is not None:
        metrics |= {"iou_reference": ref_iou, "iou": cand_iou, "iou_drop": ref_iou - cand_iou}
        tolerances["iou_drop"] = cfg.tolerances.iou_drop_max
    failures = [
        f"{name} {metrics[name]:.4f} > {bound:g}"
        for name, bound in tolerances.items()
        if metrics[name] > bound
    ]
    regressions = [
        {"question": r["question"], "answer": str(r["answer"]), "reference": a, "candidate": b}
        for r, a, b in zip(rows, ref_answers, cand_answers, strict=True)
        if normalise(a) == normalise(str(r["answer"])) != normalise(b)
    ][:5]
    report = VlmParityReport(
        not failures,
        metrics,
        tolerances,
        failures,
        eval_hash,
        len(rows),
        regressions if failures else [],
    )
    logger.info("{}", report.summary())
    return report


@log_call
def gate(
    reference: str | Path,
    candidate: str | Path,
    cfg: VlmParityConfig,
    out: Path,
    reference_url: str,
    candidate_url: str,
    register: tuple[str, str] | None = None,
    tracking_uri: str | None = None,
) -> VlmParityReport:
    """``check`` against the two servers; on a pass write ``out`` with the parity in its
    card (and register it), on a fail write nothing."""
    from pathoinfer.runtime.remote.vllm import VllmClient

    clients = [
        VllmClient(url, timeout_s=cfg.timeout_s, concurrency=cfg.concurrency)
        for url in (reference_url, candidate_url)
    ]
    try:
        report = check(
            reference,
            candidate,
            cfg,
            lambda image, question: clients[0].ask(image, question, cfg.max_tokens),
            lambda image, question: clients[1].ask(image, question, cfg.max_tokens),
        )
    finally:
        for client in clients:
            client.close()
    if report.passed:
        write_passed(reference, candidate, report, out)
        if register is not None:
            register_model(out, *register, tracking_uri=tracking_uri)
    return report


def write_passed(
    reference: str | Path, candidate: str | Path, report: VlmParityReport, out: Path
) -> VqaCard:
    """Copy the candidate folder to the new folder ``out`` with ``provenance.parity``."""
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    ref_card, _ = load_vlm(reference)
    cand_card, cand_dir = load_vlm(candidate)
    data = cand_card.model_dump(mode="json")
    data["provenance"]["parity"] = Parity(
        reference=f"{ref_card.name}@{ref_card.version}",
        tile_set_hash=report.eval_hash,
        metrics=report.metrics,
        tolerances=report.tolerances,
        checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
    ).model_dump(mode="json")
    card = VqaCard.model_validate(data)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(cand_dir, partial)
    (partial / CARD_FILE).write_text(card.model_dump_json(indent=2) + "\n")
    os.replace(partial, out)
    logger.info("vlm parity passed: wrote {}", out)
    return card


def _answers(ask: Ask, rows: list[dict[str, Any]], concurrency: int) -> list[str]:
    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda r: ask(r["image"], r["question"]), rows))


def report_json(report: VlmParityReport) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(report), indent=2) + "\n"
