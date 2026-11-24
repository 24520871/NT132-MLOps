"""Tell the running API to re-check the `production` alias (POST /admin/reload).

CLI:  python -m mlops.serving.reload_client        (env API_URL, default http://api:8000; optional ADMIN_TOKEN)
Used as the last Airflow task after a promote; the API's own polling would catch up anyway, this is just faster.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def request_reload(base_url: str, token: str | None = None, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/admin/reload", method="POST")
    if token:
        req.add_header("X-Admin-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> None:
    url = os.getenv("API_URL", "http://api:8000")
    try:
        print(request_reload(url, os.getenv("ADMIN_TOKEN")))
    except (urllib.error.URLError, OSError) as exc:  # HTTPError is a URLError; covers 401/503 and unreachable API
        print(f"Reload request to {url} failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
