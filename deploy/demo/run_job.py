"""Demo client (the `demo` service): submit a job, follow it, check its outputs.

    python run_job.py API_URL SLIDE [MODEL_FOLDER]

Exits 0 when the job finishes with a GeoJSON holding at least one polygon.
Standard library only, so it also runs outside the image.
"""

import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path


def call(url: str, body: dict[str, object] | None = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=30) as r:
        payload = r.read()
    return json.loads(payload) if r.headers.get_content_type() == "application/json" else payload


if __name__ == "__main__":
    api, slide = sys.argv[1].rstrip("/"), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).parents[1] / "model")
    config = {"task": "segmentation", "model": model, "runtime": {"batch_size": 16}}
    job = call(f"{api}/jobs", {"slide": slide, "config": config})
    print(f"submitted job {job['id']}", flush=True)
    while job["status"] in ("queued", "running"):
        time.sleep(1)
        job = call(f"{api}/jobs/{job['id']}")
        print(f"  {job['status']}: {job['tiles_done']}/{job['tiles_total']} tiles", flush=True)
    if job["status"] != "done":
        sys.exit(f"job {job['status']}: {job['error']}")
    outputs = call(f"{api}/jobs/{job['id']}/outputs")
    for output in outputs:
        print(f"  {output['name']} ({output['bytes']} bytes)", flush=True)
    geojson = next(o for o in outputs if o["name"].endswith(".geojson.gz"))
    features = json.loads(gzip.decompress(call(api + geojson["url"])))["features"]
    if not features:
        sys.exit("the GeoJSON has no features")
    print(f"done: {len(features)} polygons, e.g. {features[0]['properties']['classification']}")
