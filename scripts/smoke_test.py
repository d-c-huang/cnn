"""Check the production frontend and both trained inference models over HTTP."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=60)
    arguments = parser.parse_args()
    base = arguments.base_url.rstrip("/") + "/"
    if urlparse(base).scheme not in {"http", "https"}:
        parser.error("--base-url must use HTTP or HTTPS")

    def request(path: str, body=None):
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req = Request(urljoin(base, path.lstrip("/")), data=payload, headers={"Content-Type": "application/json"} if payload is not None else {})
        with urlopen(req, timeout=arguments.timeout) as response:
            if response.status != 200:
                raise ValueError(f"{path}: unexpected HTTP {response.status}")
            return response.read(), response.headers.get("Content-Type", "")

    def json_request(path: str, body=None):
        data, content_type = request(path, body)
        if "application/json" not in content_type:
            raise ValueError(f"{path}: expected JSON")
        return json.loads(data)

    try:
        health = json_request("/api/health")
        if health.get("status") != "ok":
            raise ValueError("Health endpoint did not report ok")
        html, content_type = request("/")
        source = html.decode("utf-8")
        if "text/html" not in content_type or 'id="root"' not in source:
            raise ValueError("Root did not serve the built frontend")
        assets = re.findall(r'(?:src|href)=["\']([^"\']*/assets/[^"\']+)["\']', source)
        if not assets:
            raise ValueError("Built frontend assets were not referenced")
        for asset in assets:
            data, _ = request(asset)
            if not data:
                raise ValueError(f"Empty asset: {asset}")
        try:
            request("/api/does-not-exist")
            raise ValueError("An unknown API path must return 404, not frontend HTML")
        except HTTPError as exc:
            if exc.code != 404:
                raise
        catalog = json_request("/api/inference/models")
        defaults = catalog.get("defaults", {})
        model_ids = [defaults.get("mlp"), defaults.get("cnn")]
        if not all(model_ids) or len(set(model_ids)) != 2:
            raise ValueError("Both trained MLP and CNN models must be available")
        architectures = {row["id"]: row["architecture"] for row in catalog.get("models", [])}
        if [architectures.get(identifier) for identifier in model_ids] != ["mlp", "cnn"]:
            raise ValueError("Default model IDs do not identify one MLP and one CNN")
        example = json_request("/api/inference/examples?label=7")
        if example.get("source") != "mnist-test" or example.get("label") != 7:
            raise ValueError("Example endpoint did not return the requested real MNIST test sample")
        pixels = example["pixels"]
        if len(pixels) != 28 or any(len(row) != 28 for row in pixels):
            raise ValueError("The test example must be 28×28")
        result = json_request("/api/inference/predict", {"model_ids": model_ids, "pixels": pixels, "mode": "mnist"})
        preprocessing = result.get("preprocessing", {})
        if preprocessing.get("mode") != "mnist" or preprocessing.get("processed_pixels") != pixels:
            raise ValueError("MNIST mode must preserve the original 28×28 test pixels")
        predictions = result.get("predictions", [])
        if {row.get("model_id") for row in predictions} != set(model_ids):
            raise ValueError("Prediction response did not include both selected models")
        for row in predictions:
            scores = row["probabilities"]
            if len(scores) != 10 or not all(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1 for value in scores):
                raise ValueError("Invalid class probability vector")
            if not math.isclose(sum(scores), 1.0, abs_tol=1e-5):
                raise ValueError("Class probabilities do not sum to one")
            if row["predicted_label"] != max(range(10), key=lambda label: scores[label]):
                raise ValueError("Prediction disagrees with argmax")
        print(json.dumps({"status": "ok", "base_url": base.rstrip("/"), "assets": len(assets), "predictions": [{"architecture": row["architecture"], "label": row["predicted_label"], "score": max(row["probabilities"])} for row in predictions]}, ensure_ascii=False))
        return 0
    except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Production smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
