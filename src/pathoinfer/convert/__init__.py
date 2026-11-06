"""Offline conversion of model folders into faster variants, each a new model folder:
``fp16`` (ONNX FP32 -> FP16), ``ptq`` (INT8/FP8 static PTQ with ModelOpt or ONNX Runtime),
``tensorrt`` (plans per GPU) and ``parity`` (the gate a variant must pass before use).
``vlm`` (VLM weights) lands after v0.3."""
