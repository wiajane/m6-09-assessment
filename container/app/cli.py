"""Standardised CLI for the Cat Detection container.

Usage inside the container:
    python /app/app/cli.py info
    python /app/app/cli.py predict
"""

import csv
import json
import os
import sys
from pathlib import Path

from app.detector import CatDetector

INPUT_DIR  = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"
STUDENT_JSON = Path("/app/STUDENT.json")
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}


def cmd_info():
    """Print STUDENT.json to stdout and exit 0."""
    print(STUDENT_JSON.read_text())


def cmd_predict():
    """Run the detector over /data/input/ and write /data/output/predictions.csv."""
    if not INPUT_DIR.exists():
        print(f"[error] Input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all image paths, preserving relative sub-paths
    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print("[warn] No images found in", INPUT_DIR, file=sys.stderr)

    detector = CatDetector()

    CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)

        for img_path in image_paths:
            # Use forward-slash relative path as the image identifier
            rel = img_path.relative_to(INPUT_DIR).as_posix()

            try:
                detections = detector.predict(str(img_path))
            except Exception as exc:  # don't let one bad image kill the whole run
                print(f"[warn] Failed on {rel}: {exc}", file=sys.stderr)
                detections = []

            if detections:
                for det in detections:
                    writer.writerow([
                        rel,
                        det["xmin"],
                        det["ymin"],
                        det["xmax"],
                        det["ymax"],
                        det["confidence"],
                        det["class"],
                    ])
            else:
                # Write the empty-detection row so every image is represented
                writer.writerow([rel, "", "", "", "", "", ""])

    print(f"Done. Wrote {len(image_paths)} image(s) -> {OUTPUT_CSV}")


def main():
    if len(sys.argv) < 2:
        print("Usage: cli.py <info|predict>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "info":
        cmd_info()
    elif cmd == "predict":
        cmd_predict()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
