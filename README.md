# Face Mask Detection using YOLOv8

A simple real-time face mask detection application built with YOLOv8 and OpenCV.

## Project Structure

```
.
├── detect_mask.py
├── train_mask_yolo.ipynb
├── best.pt
├── requirements.txt
└── slides/
```

## Installation

Clone this repository and install the required packages.

```bash
pip install -r requirements.txt
```

## Training

Model training is provided in `train_mask_yolo.ipynb`.

The notebook is designed to run on Google Colab using a T4 GPU. Dataset download is handled through Roboflow using your personal API key.

After training, download the generated `best.pt` file and place it in the same directory as `detect_mask.py`.

## Running

```bash
python detect_mask.py --model best.pt
```

Optional arguments:

```bash
--source 1
--conf 0.35
```

Keyboard shortcuts:

- **Q** : Exit
- **S** : Save current frame

## Technologies

- Python
- YOLOv8
- OpenCV
- Ultralytics
- Google Colab
- Roboflow

## Model

This project uses the pretrained **YOLOv8 Nano** model and fine-tunes it on a face mask dataset. The nano version was selected because it provides a good balance between inference speed and detection performance, making it suitable for real-time applications on standard laptops.

## Evaluation

Model performance is evaluated using Precision, Recall, and mAP@50. During deployment, the confidence threshold can be adjusted depending on the desired trade-off between detection sensitivity and false positives.

## Future Improvements

Some possible improvements include:

- Object tracking (ByteTrack)
- RTSP/CCTV support
- Detection logging
- Web dashboard
- Edge deployment