"""
02_download_models.py — Download base model weights
AID 728 Course Project

Downloads:
  1. YOLOv8l            — main vehicle + person detector
  2. lp_detector.pt     — license plate detector
                          Priority: keremberke YOLOv8m → AZIIIIIIIIZ → YOLOv8n
  3. PaddleOCR EN       — OCR engine (new and old API compatible)

All three steps are individually skipped if the model already exists and is valid.

If lp_detector.pt is the old 6.2 MB YOLOv8n fallback, delete it first:
    rm ./models/lp_detector.pt
Then re-run this script.

Run AFTER 01_collect_data.py.
Next step: python 03_train_helmet.py
"""

import os
import shutil
import pathlib

MODEL_DIR  = "./models"
PADDLE_DIR = os.path.join(MODEL_DIR, "paddleocr")
YOLO_PATH  = os.path.join(MODEL_DIR, "yolov8l.pt")
LP_PATH    = os.path.join(MODEL_DIR, "lp_detector.pt")
HELMET_PATH = os.path.join(MODEL_DIR, "helmet_classifier.pt")


# Real LP detectors (yasirfaizahmed, Koushim) are ~6 MB YOLOv8n fine-tuned.
# The COCO YOLOv8n fallback is also ~6 MB BUT has 80 classes, not 1.
# So we validate by class count, not just size.
LP_MIN_MB = 3.0


# ─────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────

def _mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(path)
            for f in fs
        ) / (1024 ** 2)
    return 0.0


def _is_valid_yolo(path, min_mb=5.0, max_classes=None):
    """True if file exists, meets minimum size, loads, and (optionally) has <= max_classes."""
    if not os.path.isfile(path):
        return False
    if _mb(path) < min_mb:
        print(f"  ⚠  {os.path.basename(path)} is only {_mb(path):.1f} MB "
              f"(expected >= {min_mb} MB) — treating as invalid.")
        return False
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        if max_classes is not None and len(m.names) > max_classes:
            print(f"  ⚠  {os.path.basename(path)} has {len(m.names)} classes "
                  f"(expected <= {max_classes}) — likely a generic COCO model.")
            return False
        return True
    except Exception as e:
        print(f"  ⚠  {os.path.basename(path)} failed to load: {e}")
        return False


def _remove_if_exists(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


# ─────────────────────────────────────────────────────────────────────
# 1. YOLOv8l  — skip if already valid
# ─────────────────────────────────────────────────────────────────────

def download_yolov8l():
    print("\n══ ① YOLOv8l ════════════════════════════════════")
    os.makedirs(MODEL_DIR, exist_ok=True)

    if _is_valid_yolo(YOLO_PATH, min_mb=50):
        print(f"  Already exists and valid ({_mb(YOLO_PATH):.1f} MB) — skipping.")
        return

    from ultralytics import YOLO
    print("  Downloading YOLOv8l (~87 MB)...")
    _ = YOLO("yolov8l.pt")
    if os.path.exists("yolov8l.pt"):
        shutil.move("yolov8l.pt", YOLO_PATH)

    m = YOLO(YOLO_PATH)
    print(f"  ✓ {YOLO_PATH}  ({_mb(YOLO_PATH):.1f} MB)")
    print(f"  ✓ Verified classes: person={m.names[0]}, motorcycle={m.names[3]}")


# ─────────────────────────────────────────────────────────────────────
# 2. LP detector
#
#    Priority:
#      A) Already have a valid lp_detector.pt (LP-specific, <=5 classes)
#      B) yasirfaizahmed/license-plate-object-detection     -> best quality
#      C) Koushim/yolov8-license-plate-detection            -> secondary
#      D) YOLOv8n generic                                   -> last resort
#
#    Validation checks BOTH file integrity AND class count to ensure
#    we don't accept a generic 80-class COCO model.
# ─────────────────────────────────────────────────────────────────────

def _try_hf_download(repo_id, filename, dest_path, min_mb, max_classes=5):
    """
    Download filename from a HuggingFace repo to dest_path.
    Validates size, loadability, and class count.
    Returns True on success, False on any failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        dl = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODEL_DIR,
        )
        # hf_hub_download may return the path directly or place the file as
        # MODEL_DIR/filename — handle both cases.
        candidates = [dl, os.path.join(MODEL_DIR, filename)]
        src = next((c for c in candidates if os.path.isfile(c)), None)
        if src is None:
            print(f"  ✗ Downloaded file not found at expected locations.")
            return False

        size = _mb(src)
        if size < min_mb:
            print(f"  ✗ Downloaded file too small ({size:.1f} MB < {min_mb} MB) "
                  f"— likely corrupt or an error page.")
            _remove_if_exists(src)
            return False

        # Move into final position
        if os.path.abspath(src) != os.path.abspath(dest_path):
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(src, dest_path)

        # Verify it loads AND has LP-specific classes (not 80 COCO classes)
        from ultralytics import YOLO
        m = YOLO(dest_path)
        if len(m.names) > max_classes:
            print(f"  ✗ Model has {len(m.names)} classes (expected <= {max_classes}) "
                  f"— this is a generic COCO model, not an LP detector.")
            _remove_if_exists(dest_path)
            return False
        print(f"    Classes: {m.names}")
        return True

    except Exception as e:
        print(f"  ✗ Exception: {e}")
        # Clean up any partial file
        _remove_if_exists(
            dest_path,
            os.path.join(MODEL_DIR, filename),
        )
        return False


def download_lp_detector():
    print("\n══ ② License Plate Detector ════════════════════")

    # ── A. Already have a valid LP-specific model ───────────────────
    if _is_valid_yolo(LP_PATH, min_mb=LP_MIN_MB, max_classes=5):
        print(f"  Already exists and valid ({_mb(LP_PATH):.1f} MB) — skipping.")
        print("  Tip: delete ./models/lp_detector.pt to force a re-download.")
        return

    # Remove any existing invalid/corrupt/COCO file before trying fresh downloads
    if os.path.exists(LP_PATH):
        print(f"  Removing invalid lp_detector.pt ({_mb(LP_PATH):.1f} MB) ...")
        os.remove(LP_PATH)

    # ── B. yasirfaizahmed — best detection quality (verified working) ──
    print("  [B] Trying yasirfaizahmed/license-plate-object-detection (~6 MB) ...")
    if _try_hf_download(
        repo_id="yasirfaizahmed/license-plate-object-detection",
        filename="best.pt",
        dest_path=LP_PATH,
        min_mb=LP_MIN_MB,
    ):
        print(f"  ✓ yasirfaizahmed LP detector ({_mb(LP_PATH):.1f} MB)")
        return
    print("  [B] yasirfaizahmed download failed — trying next option.")

    # ── C. Koushim LP detector (fallback, also verified working) ─────
    print("  [C] Trying Koushim/yolov8-license-plate-detection (~6 MB) ...")
    if _try_hf_download(
        repo_id="Koushim/yolov8-license-plate-detection",
        filename="best.pt",
        dest_path=LP_PATH,
        min_mb=LP_MIN_MB,
    ):
        print(f"  ✓ Koushim LP detector ({_mb(LP_PATH):.1f} MB)")
        return
    print("  [C] Koushim download failed — trying next option.")

    # ── D. Generic YOLOv8n — last resort ────────────────────────────
    from ultralytics import YOLO
    print("  [D] Falling back to generic YOLOv8n (no plate-specific training).")
    _ = YOLO("yolov8n.pt")
    local = "yolov8n.pt"
    if os.path.exists(local):
        shutil.move(local, LP_PATH)
    print(f"  ✓ YOLOv8n fallback saved ({_mb(LP_PATH):.1f} MB)")
    print("  ⚠  YOLOv8n does NOT detect license plates specifically.")
    print("     The multi-strategy OCR in solution.py will still attempt")
    print("     to read plates directly from the raw vehicle crop.")


# ─────────────────────────────────────────────────────────────────────
# 3. PaddleOCR EN  — skip if already cached
# ─────────────────────────────────────────────────────────────────────

def download_paddleocr():
    print("\n══ ③ PaddleOCR EN ═══════════════════════════════")

    if os.path.exists(PADDLE_DIR) and _mb(PADDLE_DIR) > 5:
        print(f"  Already cached ({_mb(PADDLE_DIR):.1f} MB) — skipping.")
        return

    os.makedirs(PADDLE_DIR, exist_ok=True)

    from paddleocr import PaddleOCR
    import numpy as np

    print("  Initialising PaddleOCR (this triggers model download) ...")

    ocr = None
    try:
        ocr = PaddleOCR(lang="en", use_textline_orientation=True)
        print("  ✓ Using modern PaddleOCR API")
    except TypeError:
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
            print("  ✓ Using legacy PaddleOCR API")
        except Exception as e:
            print(f"  ❌ PaddleOCR init failed: {e}")
            return
    except Exception as e:
        print(f"  ❌ PaddleOCR init failed: {e}")
        return

    # Force model download with a dummy image
    dummy = np.ones((64, 256, 3), dtype=np.uint8) * 255
    try:
        if hasattr(ocr, "predict"):
            ocr.predict(dummy)
        else:
            ocr.ocr(dummy)
        print("  ✓ OCR models downloaded")
    except Exception as e:
        print(f"  ⚠  Dummy OCR inference warning (non-fatal): {e}")

    # Copy cache to ./models/paddleocr/ for offline use during evaluation
    possible_caches = [
        pathlib.Path.home() / ".paddlex",
        pathlib.Path.home() / ".paddleocr",
        pathlib.Path.home() / ".paddlehub",
        pathlib.Path("/root/.paddlex"),
        pathlib.Path("/root/.paddleocr"),
    ]

    copied = False
    for cache in possible_caches:
        if not cache.exists():
            continue
        try:
            contents = list(cache.iterdir())
        except Exception:
            continue
        if not contents:
            continue
        print(f"  Found cache at: {cache} ({_mb(str(cache)):.1f} MB)")
        try:
            shutil.copytree(str(cache), PADDLE_DIR, dirs_exist_ok=True)
            print(f"  ✓ Copied to {PADDLE_DIR}")
            copied = True
            break
        except Exception as e:
            print(f"  ⚠  Copy failed: {e}")

    if not copied:
        print("  ⚠  Could not locate PaddleOCR cache — models will auto-download on first use.")
    else:
        print(f"  ✓ PaddleOCR ready ({_mb(PADDLE_DIR):.1f} MB)")


# ─────────────────────────────────────────────────────────────────────
# 4. Size budget check
# ─────────────────────────────────────────────────────────────────────

def check_budget():
    print("\n══ Model Budget ═════════════════════════════════")

    items = {
        "yolov8l.pt"           : YOLO_PATH,
        "lp_detector.pt"       : LP_PATH,
        "helmet_classifier.pt" : HELMET_PATH,
        "paddleocr/"           : PADDLE_DIR,
    }

    total = 0.0
    for label, path in items.items():
        size = _mb(path)
        total += size
        status = f"{size:.1f} MB" if os.path.exists(path) else "MISSING ❌"
        print(f"  {label:<30} {status}")

    print(f"  {'─' * 38}")
    print(f"  TOTAL                          {total:.1f} MB / 250 MB limit")

    if total > 250:
        print("\n  ⚠  OVER LIMIT. Options:")
        print("     1. Remove non-English PaddleOCR packs from paddleocr/")
        print("     2. If keremberke (~50 MB) pushed you over, swap back to")
        print("        AZIIIIIIIIZ (~36 MB) by deleting lp_detector.pt and")
        print("        commenting out step [B] in download_lp_detector().")
    else:
        print(f"\n  ✅ {250 - total:.1f} MB remaining")

    # Warn specifically if lp_detector is still the generic COCO fallback
    if os.path.exists(LP_PATH):
        try:
            from ultralytics import YOLO
            m = YOLO(LP_PATH)
            if len(m.names) > 5:
                print(f"\n  ⚠  lp_detector.pt has {len(m.names)} classes — this is a generic")
                print("     COCO model, NOT a real plate detector.")
                print("     Fix it by running:")
                print("       rm ./models/lp_detector.pt")
                print("       python3 02_download_models.py")
            else:
                print(f"\n  ✓ lp_detector.pt classes: {m.names}")
        except Exception:
            pass

def download_helmet_detector():
    """
    Download a YOLOv8n trained specifically for helmet/no-helmet detection.
    This replaces the EfficientNet-B0 classifier approach entirely.
    ~6 MB — easily fits in budget.
    """
    print("\n══ ④ Helmet Detector (YOLOv8n) ═════════════════")
    dest = os.path.join(MODEL_DIR, "helmet_detector.pt")

    if _is_valid_yolo(dest, min_mb=3):
        print(f"  Already exists ({_mb(dest):.1f} MB) — skipping.")
        return

    # Primary: iam-tsr/yolov8n-helmet-detection
    if _try_hf_download("iam-tsr/yolov8n-helmet-detection",
                         "best.pt", dest, min_mb=3):
        print(f"  ✓ Helmet detector downloaded ({_mb(dest):.1f} MB)")
        # Print class names so we know what indices to use
        from ultralytics import YOLO
        m = YOLO(dest)
        print(f"  ✓ Classes: {m.names}")
        return

    print("  ⚠  Download failed — helmet_classifier.pt fallback will be used.")
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    download_yolov8l()
    download_lp_detector()
    download_helmet_detector()
    download_paddleocr()
    check_budget()

    print("\n✅ Model download step complete.")
    print("   Next step: python 03_train_helmet.py")