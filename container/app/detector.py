import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# The model lives next to the app directory inside the container
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "best.onnx")


class CatDetector:
    def __init__(self, onnx_path=DEFAULT_MODEL, imgsz=640, conf=0.25):
        self.session = ort.InferenceSession(
            os.path.abspath(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.imgsz = imgsz
        self.conf = conf
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = ("cat",)  # single-class model

    #  Internal helpers

    def _letterbox(self, img: Image.Image):
        """Fit img inside a square of self.imgsz using grey padding.
        Returns (input_array, scale, (pad_x, pad_y)).
        """
        target = self.imgsz
        w, h = img.size
        scale = target / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (target, target), (114, 114, 114))
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        arr = (
            np.array(canvas, dtype=np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
        return arr, scale, (pad_x, pad_y)

    #  Public API

    def predict(self, image_path: str) -> list:
        """Run inference on one image.

        Returns a list of dicts:
            {xmin, ymin, xmax, ymax, confidence, class}
        Coordinates are in original-image pixels (floats).
        Returns an empty list when no detections exceed self.conf.
        """
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, (pad_x, pad_y) = self._letterbox(img)
        raw = self.session.run(None, {self.input_name: x})[0]  # (1, 300, 6)
        raw = raw[0]  # (300, 6)  ->  [x1, y1, x2, y2, score, class_id]

        results = []
        for x1, y1, x2, y2, score, cls in raw:
            if float(score) < self.conf:
                continue
            # Map from letterboxed-input space back to original pixels
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale
            # Clip to image bounds
            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))
            results.append({
                "xmin": round(x1, 2),
                "ymin": round(y1, 2),
                "xmax": round(x2, 2),
                "ymax": round(y2, 2),
                "confidence": round(float(score), 4),
                "class": self.class_names[int(cls)],
            })

        return results
