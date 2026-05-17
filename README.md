# 🚦 Traffic Violation Detection System

An advanced, AI-powered computer vision pipeline designed to automatically detect traffic violations—specifically focusing on **two-wheeler overloading** and **riding without a helmet**—and subsequently extract the **license plate number** of the violating vehicles using robust Optical Character Recognition (OCR).

## 🌟 Key Features

1. **Vehicle & Rider Detection:** 
   - Utilizes `YOLOv8l` to accurately detect motorcycles and people in high-resolution traffic imagery.
   - Features a custom geometrical assignment algorithm that maps detected persons to the specific motorcycle they are riding based on bounding box intersection and vertical alignment.

2. **Helmet Compliance Checking:**
   - Employs a fine-tuned YOLO model (`helmet_detector.pt`) specifically trained to classify riders as "With Helmet" or "Without Helmet".
   - Uses context-aware padding around the rider's upper body to ensure the model retains spatial awareness, significantly reducing false positives compared to tight head-crops.

3. **Overloading Detection:**
   - Automatically flags motorcycles carrying more than the legal limit of 2 riders.

4. **Robust License Plate Extraction:**
   - **Detection:** Uses a lightweight, fine-tuned `YOLOv8n` model (`lp_detector.pt`) to locate license plates. It dynamically searches the entire vehicle bounding box to account for diverse plate placements.
   - **OCR:** Integrates **PaddleOCR** for highly accurate text recognition. It supports multi-line plates by sorting detected text blocks by their Y-coordinates, ensuring top-to-bottom reading order.
   - **Format-Aware Cleaning:** Implements strict Regular Expression matching tailored for **Indian License Plates** (e.g., `MH 02 DT 4596`). It aggressively filters out noisy text (like 'POLICE', 'IND', or 'BOSS') and uses positional awareness to correct common OCR misreads (e.g., coercing 'S' to '5' in digit blocks, but leaving it as 'S' in state codes).

## 🛠️ Technology Stack

- **Computer Vision:** Ultralytics YOLOv8 (Object Detection & Classification)
- **OCR:** PaddleOCR (Text detection and recognition)
- **Data Processing:** OpenCV, NumPy, Pillow, Regex
- **Frameworks:** PyTorch

## 📂 Project Structure

- `solution.py`: The core `TrafficViolationDetector` pipeline that orchestrates YOLO detection, rider assignment, helmet validation, and OCR.
- `evaluate_image.py`: Command-line script to test the pipeline on individual images and generate annotated output images (`result_*.jpg`).
- `models.py`: Utility script to download and structure the required model weights (YOLO and PaddleOCR models).
- `models/`: Directory housing all downloaded weights and inference models.
- `requirements.txt`: Python dependencies required to run the pipeline.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.12+ installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Running the Detector
You can run the evaluation script on any traffic image. The script will output the violation details to the console and save a bounding-box annotated image in the root directory.

```bash
python3 evaluate_image.py "path/to/your/image.jpg"
```

**Example Output:**
```text
=======================================================
  Image: sample_traffic.jpg
=======================================================

  1. Violation Detected  :  YES  (1 vehicle(s))
  2. Total No-Helmet     :  1

  --- Vehicle 1 ---
  Violation  : 1 without helmet
  Riders     : 2
  No Helmet  : 1
  3. Plate   : MH02DT4596
```

## 🧠 System Architecture Breakdown

1. **Step 1: Scene Parsing:** The image is fed into the base YOLOv8l model. Motorcycles and persons are detected.
2. **Step 2: Rider Assignment:** Bounding boxes are analyzed. Persons heavily overlapping with a motorcycle are assigned as riders of that specific vehicle.
3. **Step 3: Violation Checking:** 
   - If riders > 2, an overloading violation is flagged.
   - Each rider is cropped (with 15px padding) and evaluated by the helmet detector. If "Without Helmet" is detected, a helmet violation is flagged.
4. **Step 4: Plate OCR (If Violated):** If any violation is found on a vehicle, the vehicle's full crop is sent to the fine-tuned License Plate YOLO model. Found plates are passed to PaddleOCR. The raw text is sorted vertically, cleaned using Indian Plate regex rules, and scored for confidence and format validity.
