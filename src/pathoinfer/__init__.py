"""pathoinfer: whole slide image inference for pathotrain models, built on patchology."""

# INFO by default; export PATHOINFER_LOG_LEVEL=DEBUG (or ALL) for more, see patchology.log.
from pathoinfer._logging import add_file_sink, set_log_level

__all__ = ["add_file_sink", "set_log_level"]
