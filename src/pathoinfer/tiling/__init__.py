"""Tiling on top of patchology: tissue mask, overlapping grid at the model's mpp, TileSpecs
with cores (``grid``), a stable slide id (``slide_id``) and a torch DataLoader that reads
tiles in worker processes (``dataset``, needs torch). Slides are read through patchology
only; pathoinfer has no reader backends of its own."""
