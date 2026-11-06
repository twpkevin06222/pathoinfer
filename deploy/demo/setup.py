"""Demo setup, run when the pathoinfer image is built: a synthetic slide and a Triton
repository for the demo model (copied into the demo Triton image).

    python setup.py DATA_DIR MODEL_REPOSITORY [MODEL_FOLDER]
"""

import sys
from pathlib import Path

from pathoinfer.serving.triton_repo import TritonRepoConfig, write_repository
from pathoinfer.tiling.synthetic import write_pyramid

if __name__ == "__main__":
    data, repository = Path(sys.argv[1]), Path(sys.argv[2])
    model = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parents[1] / "model"
    (data / "slides").mkdir(parents=True, exist_ok=True)
    write_pyramid(data / "slides" / "synthetic.tiff")
    write_repository([model], repository, TritonRepoConfig(max_batch_size=32))
    print(f"wrote {data / 'slides' / 'synthetic.tiff'} and {repository}", flush=True)
