"""make_runtime(card, config) -> Runtime: picks the runtime for the card's format and the
config. Lands in v0.1.
"""

from pathoinfer.config import InferenceConfig
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def make_runtime(card: ModelCard, config: InferenceConfig) -> Runtime:
    raise NotImplementedError("lands in v0.1")
