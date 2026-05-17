"""
evaluate_image.py — Single-image traffic violation evaluator.

Usage:
    python3 evaluate_image.py <image_path>
    python3 evaluate_image.py <image_path> --debug    # extra OCR details
"""

import argparse
import sys
import os
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single image for traffic violations."
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--debug", action="store_true",
                        help="Show extra detection details")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    print("Loading models...", end=" ", flush=True)

    from solution import TrafficViolationDetector, _read_image

    detector = TrafficViolationDetector(model_dir="./models")
    print("OK")

    print(f"\n{'='*55}")
    print(f"  Image: {os.path.basename(args.image_path)}")
    print(f"{'='*55}")

    result = detector.predict(args.image_path)
    violations = result.get("violations", [])

    if not violations:
        print("\n  1. Violation Detected  :  No")
        print("  2. No-Helmet Count     :  0")
        print("  3. License Plate       :  N/A")
        print()
        return

    total_no_helmet = sum(v.get("helmet_violations", 0) for v in violations)

    print(f"\n  1. Violation Detected  :  YES  ({len(violations)} vehicle(s))")
    print(f"  2. Total No-Helmet     :  {total_no_helmet}")

    for i, v in enumerate(violations):
        no_helmet = v.get("helmet_violations", 0)
        riders = v.get("num_riders", 0)
        plate = v.get("license_plate", "UNKNOWN")

        reasons = []
        if no_helmet > 0:
            reasons.append(f"{no_helmet} without helmet")
        if riders > 2:
            reasons.append(f"overloading ({riders} riders)")

        print(f"\n  --- Vehicle {i+1} ---")
        print(f"  Violation  : {' + '.join(reasons)}")
        print(f"  Riders     : {riders}")
        print(f"  No Helmet  : {no_helmet}")
        print(f"  3. Plate   : {plate}")

        if args.debug:
            vbox = v.get("vehicle_bbox")
            rbboxes = v.get("rider_bboxes", [])
            if vbox:
                print(f"  Vehicle Box: [{int(vbox[0])},{int(vbox[1])},{int(vbox[2])},{int(vbox[3])}]")
            print(f"  Rider Boxes: {len(rbboxes)}")

    # ── Draw annotated image ──
    img = _read_image(args.image_path)
    if img is not None:
        for i, v in enumerate(violations):
            vbox = v.get("vehicle_bbox")
            plate = v.get("license_plate", "UNKNOWN")
            no_helmet = v.get("helmet_violations", 0)

            if vbox:
                x1, y1, x2, y2 = [int(c) for c in vbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Plate:{plate}"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for rbox in v.get("rider_bboxes", []):
                rx1, ry1, rx2, ry2 = [int(c) for c in rbox]
                color = (0, 0, 255) if no_helmet > 0 else (0, 255, 0)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)

        base = os.path.splitext(os.path.basename(args.image_path))[0]
        out_name = f"result_{base}.jpg"
        cv2.imwrite(out_name, img)
        print(f"\n  [Saved] {out_name}")

    print()


if __name__ == "__main__":
    main()
