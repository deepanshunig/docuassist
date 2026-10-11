"""Upload one sample PDF through the running backend."""
import argparse
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8011")
    arguments = parser.parse_args()
    with arguments.pdf.open("rb") as source:
        response = httpx.post(
            arguments.api_url.rstrip("/") + "/documents",
            files={"file": (arguments.pdf.name, source, "application/pdf")}, timeout=300,
            trust_env=False,
        )
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()
