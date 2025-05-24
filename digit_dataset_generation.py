import cv2
import numpy as np
import os
import random

OUTPUT_DIR = "dataset_digits"
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_PLAIN
]
IMG_SIZE = 28
NUM_SAMPLES_PER_CLASS = 1000
TRAIN_RATIO = 0.9

def add_noise(img):
    noise = np.random.randint(0, 50, img.shape, dtype='uint8')
    return cv2.add(img, noise)

def generate_digit_image(digit, font):
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype='uint8') * 255
    font_scale = random.uniform(0.8, 1.2)
    thickness = random.randint(1, 2)
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    org = ((IMG_SIZE - text_size[0]) // 2, (IMG_SIZE + text_size[1]) // 2)
    cv2.putText(img, str(digit), org, font, font_scale, (0,), thickness, cv2.LINE_AA)
    
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)
    
    img = add_noise(img)
    
    return img

def save_img(img, label, index, split):
    folder = os.path.join(OUTPUT_DIR, split, str(label))
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, f"{index}.png"), img)

for digit in range(1, 10):
    for i in range(NUM_SAMPLES_PER_CLASS):
        font = random.choice(FONTS)
        img = generate_digit_image(digit, font)
        split = "train" if i < NUM_SAMPLES_PER_CLASS * TRAIN_RATIO else "val"
        save_img(img, digit, i, split)
