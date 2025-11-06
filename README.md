# Car Plate Recognition using YOLOv8 🚗💡
![predictions](predictions.jpg)

Welcome to the **Car Plate Recognition** project using **YOLOv8**!  
This repository provides a complete pipeline — from dataset preparation and model training to inference and plate text reading — for automatic car plate recognition.

![video_predictions](output02.avi)

---

## 🚀 Overview

This project extends the basic YOLOv8 object detection workflow to detect and read car plates.  
You’ll learn how to prepare a custom dataset, train a YOLOv8 model, and use OCR (EasyOCR) to read license plates from images and videos.

📦 **Dataset:**  
You can find the dataset used in this project on Kaggle:  
👉 [Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

💻 **Kaggle Notebook:**  
A full tutorial notebook is also available here:  
👉 [Training Object Detection Model using YOLOv8](https://www.kaggle.com/code/elahehgolrokh/training-object-detection-model-using-yolo8)

---

## 🧩 Development Steps

1. **Train a baseline plate detection model** using YOLOv8 on the Kaggle dataset.  
2. **Add the plate reading phase** using EasyOCR to recognize plate numbers.  
3. **Improve the detection model** through fine-tuning on [this Kaggle dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset).  
4. **To Do:** Enhance prediction **consistency across video frames** for smoother tracking.

---

## ⚙️ Installation

Tested on **Ubuntu 20.04** with **Python 3.9.12**.

### 1️⃣ Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

---

## 🗂️ Data Preparation

To use YOLOv8 for car plate detection, organize your dataset as follows:

1. In your dataset root directory (e.g., `data/`), create two folders: `images/` and `labels/`.  
2. Images can be in `.jpg` or `.png` format.  
3. Create a YAML configuration file specifying dataset paths.  
4. Optionally split your dataset into `train`, `validation`, and `test` subsets inside both `images` and `labels` folders.  
5. Label files must be in `.txt` format, with each line containing:  
   ```
   class_id x_center y_center width height
   ```

### Example directory structure

```
data
├── images
│   ├── train
│   ├── validation
│   └── test
└── labels
    ├── train
    ├── validation
    └── test
```

### Original dataset format (before conversion)
If your dataset starts in this format:
```
.
├── annotations
│   ├── Cars0.xml
│   ├── Cars100.xml
│   ├── ...
└── images
    ├── Cars0.png
    ├── Cars100.png
    ├── ...
```
The data preparation phase in this repository automatically converts it to YOLO format.

---

## 🏋️‍♂️ Training the YOLOv8 Model

To train the car plate detection model, simply run:

```
python main.py
```

You can customize the following arguments:

| Flag | Description |
|------|--------------|
| `-rpr`, `--remove_prev_runs` | Remove previous YOLO run directories before training |
| `-p`, `--prepare` | Run data preparation before training |
| `-t`, `--train` | Enable model training |
| `-e`, `--export` | Export the trained YOLOv8 model |

---

## 🧠 Inference with Trained Models

To perform inference on images using a trained YOLOv8 model:

```
python inference.py --model_path 'path/to/model' --image_path 'path/to/test_image' --output_name 'output.png'
```

You can customize the following arguments:

| Flag | Description |
|------|--------------|
| `-mp`, `--model_path` | Path to the trained YOLOv8 model |
| `-ip`, `--image_path` | Path to an image or directory of images |
| `-vp`, `--video_path` | Path to a test video file |
| `-on`, `--output_name` | Name for the saved output (image or video) |

📍 **Default model path:**  
`runs/detect/train/weights/best.pt`  

🖼️ **Output:**  
Prediction results (images or videos) are saved in the `runs/` directory.

> **Note:**  
> You may need to adjust `datasets_dir`, `weights_dir`, or `runs_dir` in `.config/Ultralytics/settings.yaml` depending on your project’s root directory.

### Run inference on videos
```
python inference.py --model_path 'path/to/model' --video_path 'path/to/test_video' --output_name 'output.avi'
```

---

## 🔍 Plate Reading (OCR)

![plate_reading](plate_reading.jpg)

To recognize plate numbers from detections, use the OCR option:

```
python inference.py --model_path 'path/to/model' --image_path 'path/to/test_image' -rf
```

This enables the **EasyOCR** module to extract plate text from the detected bounding boxes.

---

## 🧾 License
This project is open-source and distributed under the MIT License.  
Feel free to use, modify, and share it for research or personal projects.

---

## 🙌 Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Kaggle Dataset: Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

---

**Author:** Elaheh Golrokh  
📧 For questions or collaboration: [GitHub Profile](https://github.com/elahehgolrokh)
