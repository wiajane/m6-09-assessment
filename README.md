![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Final Assessment | Cat Detection v2 — Improve, Export to ONNX, Containerise & Compete

> ⚠️ **Before submitting:** fill the Week-1 baseline row in the table below from your Week-1 run, then `docker login && docker push wiajane/cat-detector:final`.

## Image for leaderboard

```bash
docker pull wiajane/cat-detector:final
```

Image: `wiajane/cat-detector:final`
Student: Jala Suleymanova

## Overview

Final assessment of Unit 6, in three parts:

1. **Improve** the Week-1 YOLO26 cat detector with Week-2 techniques (larger backbone, stronger augmentation, regularisation).
2. **Convert** the improved model to **ONNX** using YOLO26's NMS-free end-to-end export.
3. **Containerise** the inference behind a fixed, standardised CLI, scored on mAP@0.5 against the instructor's holdout.

Dataset (same as Week 1): [Cat Detection Dataset on Google Drive](https://drive.google.com/drive/folders/1qeGvkaK7UkNMYoESQHxGbV4DRH8EgEb0?usp=drive_link)

## This submission

| Notebook | Role |
|---|---|
| `m6-04-assessment.ipynb` | Part A — improve & train: recap, 3 Week-2 techniques, training, test eval, comparison table |
| `m6-09-assessment.ipynb` | ONNX export + sanity check, container build, `info`/`predict` test, CSV verification |

### Result (v2 — best, test split)

| Run | Backbone | Tricks | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|---|
| Week-1 baseline | yolo26n | none | _fill from Week-1_ | _fill_ | _fill_ | _fill_ |
| **v2 — best** | yolo26s | mosaic + HSV + rotation + weight_decay, 35 ep | **0.9104** | **0.6778** | **0.9149** | **0.8497** |

### Week-2 techniques applied

1. **Larger backbone (transfer learning):** YOLO26n → YOLO26s (~2.4M → ~9.5M params).
2. **Stronger augmentation:** `mosaic`, `hsv_h/s/v`, `degrees`, `translate`, `scale`, `fliplr`.
3. **Regularisation + early stopping:** `weight_decay=0.0005`, `patience=8`, `lr0=0.01`, `momentum=0.937`.

## Container

```
container/
  Dockerfile
  STUDENT.json
  requirements.txt
  app/
    __init__.py
    cli.py
    detector.py
  models/
    best.onnx
```

### Build

```bash
docker build -t wiajane/cat-detector:final container
```

### Standardised CLI

`info` — prints `STUDENT.json` to stdout:

```bash
docker run --rm wiajane/cat-detector:final info
```

`predict` — runs ONNX inference over `/data/input/`, writes `/data/output/predictions.csv`:

```bash
docker run --rm \
  -v /absolute/path/to/holdout:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  wiajane/cat-detector:final predict
```

CSV schema (UTF-8, header, forward-slash relative paths, absolute-pixel box coords, one empty row for images with no detection):

```
image_path,xmin,ymin,xmax,ymax,confidence,class
```

### Publish

```bash
docker login
docker push wiajane/cat-detector:final
```

## Reproduce locally

```bash
pip install ultralytics onnx onnxslim onnxruntime numpy pillow opencv-python-headless
# train + evaluate
jupyter nbconvert --to notebook --execute --inplace m6-04-assessment.ipynb
# export ONNX + build & test the container
jupyter nbconvert --to notebook --execute --inplace m6-09-assessment.ipynb
```

Everything generated lands in isolated folders: `jane_work/` (clean data, splits, `data.yaml`), `runs/` (training run + `best.pt` + `best.onnx`), and `container/` (container source + model).

## Definition of done

- [x] ≥ 3 Week-2 techniques applied; comparison table vs Week-1 baseline (fill baseline row).
- [x] Best model exported to ONNX; ONNX-vs-PyTorch sanity check shown.
- [x] `STUDENT.json` with real first/last names (Jale Suleymanova).
- [x] Container builds cleanly from `Dockerfile`.
- [x] `info` prints the JSON student record.
- [x] `predict` writes `/data/output/predictions.csv` with the exact schema.
- [x] CSV verified on a small local folder.
- [ ] Image pushed to a public registry (`docker push wiajane/cat-detector:final`).
