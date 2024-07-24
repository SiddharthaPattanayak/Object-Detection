import cv2
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(frame):
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image)
    return predictions[0]

def get_predominant_color(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    predominant_color = palette[0].astype(int)
    return tuple(predominant_color)

def save_cropped_objects(frame, predictions, output_folder, frame_num):
    for i, (box, score, label) in enumerate(zip(predictions['boxes'], predictions['scores'], predictions['labels'])):
        if score > 0.5:
            xmin, ymin, xmax, ymax = box.int().tolist()
            cropped_image = frame[ymin:ymax, xmin:xmax]
            color = get_predominant_color(cropped_image)
            color_name = f"{color[0]}_{color[1]}_{color[2]}"
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            label_folder = os.path.join(output_folder, label_name)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            filename = f"{label_folder}/frame{frame_num}_obj{i}_{color_name}.jpg"
            cv2.imwrite(filename, cropped_image)

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_num = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_objects(frame)
        save_cropped_objects(frame, predictions, output_folder, frame_num)

        frame_num += 1

        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = r"C:\Users\LENOVO\Desktop\Video Fotage\Shopping,_People,_Commerce,_Mall,_Many,_Crowd,_Walking___Free_Stock_video_footage___YouTube(720p).mp4"
output_folder = r"C:\Users\LENOVO\Desktop\Object_Image_Save_Folder"
process_video(video_path, output_folder)
