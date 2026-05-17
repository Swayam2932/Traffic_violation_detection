"""
solution.py — Traffic Violation Detection System
AID 728 Course Project

Models:
  yolov8l.pt            — vehicle + person detection  (COCO pretrained)
  helmet_classifier.pt  — EfficientNet-B0, binary: helmet=0 / no_helmet=1
  lp_detector.pt        — YOLOv8 (fine-tuned LP detector or YOLOv8n fallback)
  paddleocr/            — PaddleOCR EN models (offline)

Helmet detection strategy:
  • Head crop = top 28% of person bbox (head only, not torso).
  • Three crop scales (22 / 28 / 35%) are classified independently.
  • Each crop is also horizontally flipped (test-time augmentation).
  • The 6 predictions are averaged → more stable than a single forward pass.
  • Threshold = 0.42  (P(helmet) < 0.42 → violation).

Rider assignment strategy:
  • GLOBAL EXCLUSIVE: each detected person is assigned to exactly ONE vehicle
    (the one with the highest overlap score). Prevents riders on adjacent bikes
    from being counted on the wrong motorcycle.
  • Geometric sanity checks: person centre-x must lie within the vehicle's
    horizontal span; person cannot be entirely above or below the vehicle.
  • Physical cap: at most 3 riders per motorcycle.
"""

import os
import cv2
import re
import numpy as np
import torch
import torchvision.transforms as T
import timm
from PIL import Image
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

PERSON_CLASS         = 0
MOTORCYCLE_CLASSES   = {3}

YOLO_CONF            = 0.30
RIDER_OVERLAP_THRESH = 0.25      # min person→vehicle overlap to assign
MAX_RIDERS_PER_BIKE  = 3         # physical maximum on one motorcycle

# ── Helmet detection ─────────────────────────────────────────────
HELMET_THRESHOLD     = 0.42      # P(helmet) < this → violation
                                  # (lower = more sensitive to no-helmet)
HEAD_CROP_FRACS      = [0.22, 0.28, 0.35]   # fraction of person-bbox height
                                              # used for head crops (multi-scale)
HEAD_CROP_MIN_PX     = 32        # skip crops smaller than this (bad YOLO box)
HELMET_CLASS_IDX     = 0         # class 0 = helmet (alphabetical order)
IMG_SIZE             = 224

# ── License plate ────────────────────────────────────────────────
LP_CONF              = 0.10
LP_DETECTOR_MIN_MB   = 20.0      # below → treat lp_detector.pt as generic fallback

INFER_TFM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

_INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")

_INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UP","UK","WB","AN","CH","DD","DL","JK","LA","LD","PY",
}


# ═══════════════════════════════════════════════════════════════════
#  RIDER-ASSIGNMENT HELPERS
# ═══════════════════════════════════════════════════════════════════

def _person_in_vehicle_overlap(pbox, vbox):
    """Fraction of PERSON bbox area that lies inside VEHICLE bbox."""
    xa = max(pbox[0], vbox[0]);  xb = min(pbox[2], vbox[2])
    ya = max(pbox[1], vbox[1]);  yb = min(pbox[3], vbox[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0:
        return 0.0
    area_p = max(1.0, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
    return inter / area_p


def _rider_score(pbox, vbox, img_h):
    """
    Returns an assignment score for (person, vehicle).
    Returns 0 if geometric sanity checks fail.

    Checks:
      1. Overlap >= RIDER_OVERLAP_THRESH
      2. Person centre-x within vehicle x-span (±20% tolerance)
      3. Person is not entirely above or below the vehicle
    """
    ov = _person_in_vehicle_overlap(pbox, vbox)
    if ov < RIDER_OVERLAP_THRESH:
        return 0.0

    # Horizontal alignment
    p_cx      = (pbox[0] + pbox[2]) / 2.0
    v_w       = max(1, vbox[2] - vbox[0])
    tolerance = v_w * 0.20
    if p_cx < vbox[0] - tolerance or p_cx > vbox[2] + tolerance:
        return 0.0

    # Vertical sanity: person bottom above vehicle top → impossible rider
    if pbox[3] < vbox[1]:
        return 0.0
    # Person top below vehicle bottom → walking behind bike, not riding
    if pbox[1] > vbox[3]:
        return 0.0

    return ov


def _assign_persons_to_vehicles(persons, vehicles, img_h):
    """
    GLOBAL EXCLUSIVE ASSIGNMENT.
    Each person is assigned to exactly ONE vehicle — the one with the
    highest _rider_score.  Persons with no qualifying vehicle are ignored.

    Returns: dict  vehicle_idx -> list[person_idx]
    """
    scored = []
    for p_idx, pbox in enumerate(persons):
        for v_idx, vbox in enumerate(vehicles):
            sc = _rider_score(pbox, vbox, img_h)
            if sc > 0:
                scored.append((sc, p_idx, v_idx))

    scored.sort(key=lambda x: -x[0])   # best match first
    assigned_persons  = set()
    vehicle_to_riders = {i: [] for i in range(len(vehicles))}

    for sc, p_idx, v_idx in scored:
        if p_idx in assigned_persons:
            continue
        assigned_persons.add(p_idx)
        vehicle_to_riders[v_idx].append(p_idx)

    return vehicle_to_riders


# ═══════════════════════════════════════════════════════════════════
#  HELMET CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def _head_crops_for_person(img: np.ndarray, pbox: list) -> list:
    """
    Generate multiple head-region crops from a person bbox at three scales.
    Returns only crops that meet the minimum pixel-size requirement.

    Why three scales?
      YOLO person bboxes vary in how tightly they fit the person.
      22% of bbox height captures the head for a loose bbox (full-body).
      35% captures the head for a tight bbox (waist-up shot).
      Classifying all three and averaging is more robust.
    """
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = [int(c) for c in pbox]
    person_h = y2 - y1
    person_w = x2 - x1

    crops = []
    for frac in HEAD_CROP_FRACS:
        head_h  = max(HEAD_CROP_MIN_PX, int(person_h * frac))
        # Small horizontal pad to capture full head width even if pbox is narrow
        pad_x   = int(person_w * 0.10)
        hx1     = max(0, x1 - pad_x)
        hx2     = min(w_img, x2 + pad_x)
        hy2     = min(h_img, y1 + head_h)
        crop    = img[y1:hy2, hx1:hx2]
        if (crop is not None
                and crop.size > 0
                and crop.shape[0] >= HEAD_CROP_MIN_PX
                and crop.shape[1] >= HEAD_CROP_MIN_PX):
            crops.append(crop)

    return crops


def _crops_to_tensors(crops: list) -> list:
    """
    Convert crops to classifier tensors.
    Each crop also generates a horizontally-flipped version (TTA).
    """
    tensors = []
    for crop in crops:
        try:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(INFER_TFM(pil))
            # Test-time augmentation: horizontal flip
            tensors.append(INFER_TFM(pil.transpose(Image.FLIP_LEFT_RIGHT)))
        except Exception:
            pass
    return tensors


# ═══════════════════════════════════════════════════════════════════
#  PLATE OCR HELPERS
# ═══════════════════════════════════════════════════════════════════

def _clean_plate(raw: str) -> str:
    """Position-aware char correction for Indian plates (LL NN L(L)(L) NNNN)."""
    if not raw:
        return "UNKNOWN"
        
    text = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
    text = text.replace("POLICE", "").replace("IND", "")
    
    L = r"[A-Z0683]"       
    N = r"[0-9OIQZASBGTC]"   
    
    pattern = f"({L}{{2}})({N}{{2}})({L}{{1,3}})({N}{{4}})"
    match = re.search(pattern, text)
    if match:
        state, dist, series, number = match.groups()
        state = state.translate(str.maketrans("0683", "OGBJ"))
        dist = dist.translate(str.maketrans("OIQZASBGTC", "0102458610"))
        series = series.translate(str.maketrans("0683", "OGBJ"))
        number = number.translate(str.maketrans("OIQZASBGTC", "0102458610"))
        return state + dist + series + number
        
    if len(text) < 4:
        return "UNKNOWN"
    if len(text) >= 8:
        state  = text[0:2].translate(str.maketrans("0683", "OGBJ"))
        dist   = text[2:4].translate(str.maketrans("OIQZASBGTC", "0102458610"))
        series = text[4:6].translate(str.maketrans("0683", "OGBJ"))
        number = text[6:].translate(str.maketrans("OIQZASBGTC", "0102458610"))
        text   = state + dist + series + number
    else:
        text = re.sub(r"(?<=[0-9])[OQ](?=[0-9])", "0", text)
        
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    return text[:13] if len(text) >= 4 else "UNKNOWN"


def _plate_quality(text: str) -> float:
    if not text or text == "UNKNOWN":
        return 0.0
    score = len(text) * 0.1
    if _INDIAN_PLATE_RE.match(text):
        score += 2.0
    if len(text) >= 2 and text[:2] in _INDIAN_STATE_CODES:
        score += 1.0
    return score


def _plate_variants(crop: np.ndarray) -> list:
    """5 preprocessing variants for best OCR coverage."""
    if crop is None or crop.size == 0:
        return []
    scale = max(1.0, 300 / max(crop.shape[1], 1))
    big   = cv2.resize(crop, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu  = cv2.threshold(denoised, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4)

    def g2b(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    return [big, g2b(enhanced), g2b(otsu),
            g2b(cv2.bitwise_not(otsu)), g2b(adaptive)]


def _model_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 ** 2)
    except Exception:
        return 0.0


def _read_image(image_path: str) -> np.ndarray:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        pil = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
#  OCR WRAPPER
# ═══════════════════════════════════════════════════════════════════

class _OCRWrapper:
    """Handles both old (<2.8) and new (>=2.8) PaddleOCR APIs."""

    def __init__(self, model_dir: str):
        from paddleocr import PaddleOCR
        self._new_api = False
        try:
            self._ocr = PaddleOCR(lang="en", use_textline_orientation=True)
            self._new_api = True
        except TypeError:
            paddle_dir = os.path.join(model_dir, "paddleocr")
            kwargs = dict(use_textline_orientation=True, lang="en", use_gpu=False)
            det = os.path.join(paddle_dir, "whl/det/en/en_PP-OCRv3_det_infer")
            rec = os.path.join(paddle_dir, "whl/rec/en/en_PP-OCRv4_rec_infer")
            if os.path.isdir(det): kwargs["det_model_dir"] = det
            if os.path.isdir(rec): kwargs["rec_model_dir"] = rec
            self._ocr = PaddleOCR(**kwargs)

    def run(self, img_bgr: np.ndarray) -> list:
        results = []
        try:
            if self._new_api:
                out = self._ocr.predict(img_bgr)
                if out:
                    for block in out:
                        if not isinstance(block, dict):
                            continue
                        texts  = block.get("rec_texts",  None) or block.get("rec_text",  [])
                        scores = block.get("rec_scores", None) or block.get("rec_score", [])
                        polys  = block.get("dt_polys", []) or block.get("rec_polys", []) or block.get("rec_boxes", [])
                        
                        if not isinstance(texts, list):
                            texts, scores = [texts], [scores]
                            polys = [polys] if polys is not None else []
                            
                        items = []
                        for i, (t, s) in enumerate(zip(texts, scores)):
                            if not t: continue
                            y_center = 0
                            if i < len(polys) and polys[i] is not None and len(polys[i]) > 0:
                                poly = np.array(polys[i])
                                if len(poly.shape) == 2 and poly.shape[1] == 2:
                                    y_center = np.mean(poly[:, 1])
                                elif len(poly) >= 2:
                                    y_center = poly[1] if isinstance(poly[1], (int, float, np.number)) else 0
                            items.append((y_center, str(t), float(s or 0)))
                            
                        # Sort top to bottom
                        items.sort(key=lambda x: x[0])
                        for _, t, s in items:
                            results.append((t, s))
            else:
                out = self._ocr.ocr(img_bgr)
                if out and out[0]:
                    items = []
                    for line in out[0]:
                        if line and len(line) >= 2:
                            poly = line[0]
                            txt, conf = line[1]
                            y_center = np.mean([pt[1] for pt in poly]) if poly else 0
                            items.append((y_center, str(txt), float(conf or 0)))
                            
                    items.sort(key=lambda x: x[0])
                    for _, t, s in items:
                        results.append((t, s))
        except Exception:
            pass
        return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════

class TrafficViolationDetector:

    def __init__(self, model_dir: str = "./models"):
        # 1. YOLOv8l
        self.yolo = YOLO(os.path.join(model_dir, "yolov8l.pt"))

        # 2. EfficientNet-B0 helmet classifier
        helmet_det_path = os.path.join(model_dir, "helmet_detector.pt")
        if os.path.exists(helmet_det_path) and _model_mb(helmet_det_path) >= 3:
            self.helmet_yolo = YOLO(helmet_det_path)
            # Discover which class index = "no_helmet" / "without_helmet"
            names_lower = {v.lower(): k for k, v in self.helmet_yolo.names.items()}
            self._no_helmet_cls = {
                names_lower.get("no_helmet"),
                names_lower.get("without_helmet"),
                names_lower.get("no-helmet"),
                names_lower.get("without helmet"),
            } - {None}
            self._use_helmet_yolo = True
            print(f"[INFO] Helmet: YOLO detector  classes={self.helmet_yolo.names}")
        else:
            self._use_helmet_yolo = False
            print("[INFO] Helmet: EfficientNet-B0 classifier (fallback)")

        # 3. LP detector
        lp_path      = os.path.join(model_dir, "lp_detector.pt")
        self.lp_yolo = YOLO(lp_path)
        # Generic YOLOv8n has 80 classes. Fine-tuned LP detectors usually have 1 (license_plate).
        self._lp_real = len(self.lp_yolo.names) < 10
        print(f"[INFO] LP detector: "
              f"{'fine-tuned' if self._lp_real else 'generic YOLOv8n fallback'} "
              f"({_model_mb(lp_path):.1f} MB)")

        # 4. OCR
        self._ocr = _OCRWrapper(model_dir)

    # ────────────────────────────────────────────────────────────
    #  STEP 1 — Detect vehicles and assign riders (exclusive)
    # ────────────────────────────────────────────────────────────

    def _detect_vehicles(self, image_path: str, img: np.ndarray) -> list:
        h, w = img.shape[:2]

        yolo_res = self.yolo(
            image_path, conf=YOLO_CONF, verbose=False,
            classes=list(MOTORCYCLE_CLASSES) + [PERSON_CLASS],
        )[0]

        vehicles, persons = [], []
        for box in yolo_res.boxes:
            cls  = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            if cls in MOTORCYCLE_CLASSES:
                vehicles.append(xyxy)
            elif cls == PERSON_CLASS:
                persons.append(xyxy)

        vehicle_to_riders = _assign_persons_to_vehicles(persons, vehicles, h)

        detections = []
        for v_idx, vbox in enumerate(vehicles):
            rider_idxs = vehicle_to_riders.get(v_idx, [])

            # Enforce physical cap (keep highest-overlap riders if over limit)
            if len(rider_idxs) > MAX_RIDERS_PER_BIKE:
                rider_idxs = sorted(
                    rider_idxs,
                    key=lambda p: _rider_score(persons[p], vbox, h),
                    reverse=True,
                )[:MAX_RIDERS_PER_BIKE]

            rider_bboxes = [persons[i] for i in rider_idxs]

            # ── Build MULTI-SCALE head crops for each rider ──────────
            # Each element in rider_head_crops is a list of crops
            # (one per scale) for that particular rider.
            rider_head_crops = []
            for i in rider_idxs:
                crops = _head_crops_for_person(img, persons[i])
                rider_head_crops.append(crops)

            detections.append({
                "vehicle_bbox"    : vbox,
                "num_riders"      : len(rider_bboxes),
                "rider_head_crops": rider_head_crops,  # list[list[ndarray]]
                "rider_bboxes"    : rider_bboxes,
            })

        return detections

    # ────────────────────────────────────────────────────────────
    #  STEP 2 — Helmet classification  (multi-scale + TTA)
    # ────────────────────────────────────────────────────────────

    def _count_helmet_violations(self, rider_head_crops, rider_bboxes, img):
        if self._use_helmet_yolo:
            violations = 0
            for pbox in rider_bboxes:
                x1,y1,x2,y2 = [int(c) for c in pbox]
                
                # The YOLOv8 helmet detector needs the full person (or upper body) context.
                # A tight head crop causes it to hallucinate helmets.
                pad = 15
                h_img, w_img = img.shape[:2]
                px1 = max(0, x1 - pad)
                py1 = max(0, y1 - pad)
                px2 = min(w_img, x2 + pad)
                py2 = min(h_img, y2 + pad)
                
                crop = img[py1:py2, px1:px2]
                if crop is None or crop.size == 0:
                    continue
                    
                res = self.helmet_yolo(crop, conf=0.25, verbose=False)[0]
                # Violation if any "without_helmet" box is detected within the person crop
                classes_detected = {int(b.cls[0]) for b in res.boxes}
                if classes_detected & self._no_helmet_cls:
                    violations += 1
            return violations
        else:
            # fallback to old EfficientNet path
            return self._count_helmet_violations_classifier(rider_head_crops)

    # ────────────────────────────────────────────────────────────
    #  STEP 3 — License plate OCR
    # ────────────────────────────────────────────────────────────

    def _candidate_plate_crops(self, img: np.ndarray, vbox: list) -> list:
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in vbox]
        veh_h = y2 - y1
        veh_w = x2 - x1

        crops = []
        
        # 1. Use the full vehicle crop for the fine-tuned YOLO LP detector to retain context
        if self._lp_real:
            try:
                pad_v = max(10, int(veh_h * 0.05))
                pad_h = max(10, int(veh_w * 0.05))
                vx1 = max(0, x1 - pad_h)
                vy1 = max(0, y1 - pad_v)
                vx2 = min(w_img, x2 + pad_h)
                vy2 = min(h_img, y2 + pad_v)
                
                veh_crop = img[vy1:vy2, vx1:vx2]
                
                lp_res = self.lp_yolo(veh_crop, conf=LP_CONF, verbose=False)[0]
                for box in lp_res.boxes:
                    bx1, by1, bx2, by2 = [int(c) for c in box.xyxy[0].tolist()]
                    pad = 5
                    crop = veh_crop[
                        max(0, by1-pad):min(veh_crop.shape[0], by2+pad),
                        max(0, bx1-pad):min(veh_crop.shape[1], bx2+pad)]
                    if crop.size > 0:
                        crops.append(crop)
            except Exception:
                pass

        # 2. Fallback: Naive bottom-half crops for blind OCR
        sx1 = max(0, x1 - int(veh_w * 0.03))
        sx2 = min(w_img, x2 + int(veh_w * 0.03))
        sy1 = y1 + int(veh_h * 0.40)
        sy2 = min(h_img, y2 + int(veh_h * 0.05))
        search = img[sy1:sy2, sx1:sx2]

        if search is not None and search.size > 0:
            sh = search.shape[0]
            crops.append(search[int(sh * 0.60):, :])
            crops.append(search[int(sh * 0.30):, :])
            crops.append(search)
            
        return [c for c in crops if c is not None and c.size > 0]

    def _read_plate(self, img: np.ndarray, vbox: list) -> str:
        try:
            crops = self._candidate_plate_crops(img, vbox)
            if not crops:
                return "UNKNOWN"
            best_text, best_score = "UNKNOWN", 0.0
            for crop in crops:
                for variant in _plate_variants(crop):
                    ocr_out = self._ocr.run(variant)
                    if not ocr_out:
                        continue
                    combined_txt  = "".join(t for t, _ in ocr_out)
                    combined_conf = sum(c for _, c in ocr_out) / len(ocr_out)
                    for txt, conf in [(combined_txt, combined_conf)] + ocr_out:
                        cleaned = _clean_plate(txt)
                        if cleaned == "UNKNOWN" or len(cleaned) < 4:
                            continue
                        score = conf + _plate_quality(cleaned)
                        if score > best_score:
                            best_score, best_text = score, cleaned
            return best_text
        except Exception:
            return "UNKNOWN"

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Input : path to image (JPG / PNG / JPEG / WEBP)
        Output: {
                  "violations": [
                    {
                      "num_riders"       : int,
                      "helmet_violations": int,
                      "license_plate"    : str
                    }, ...
                  ]
                }
        """
        try:
            img = _read_image(image_path)
            if img is None:
                print(f"[WARN] Could not read: {image_path}")
                return {"violations": []}

            violations = []
            for det in self._detect_vehicles(image_path, img):
                num_riders        = det["num_riders"]
                helmet_violations = self._count_helmet_violations(
                    det["rider_head_crops"], det["rider_bboxes"], img)

                if num_riders > 2 or helmet_violations > 0:
                    violations.append({
                        "num_riders"       : num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate"    : self._read_plate(
                            img, det["vehicle_bbox"]),
                        # kept for evaluate_image.py drawing
                        "vehicle_bbox"     : det["vehicle_bbox"],
                        "rider_bboxes"     : det["rider_bboxes"],
                    })

            return {"violations": violations}

        except Exception as e:
            print(f"[ERROR] predict(): {e}")
            return {"violations": []}