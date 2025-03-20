import cv2
import csv
import os
import random
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from typing import List

N_FACES_THRESHOLD = 20

IMG_HEIGHT = 100
IMG_WIDTH = 70

X_TOP_LEFT_IDX = 2
Y_TOP_LEFT_IDX = 3
X_BOT_RIGHT_IDX = 6
Y_BOT_RIGHT_IDX = 7


FRs = {
    "Eigen": cv2.face.EigenFaceRecognizer_create(),
    "Fisher": cv2.face.FisherFaceRecognizer_create(),
    "LBPH": cv2.face.LBPHFaceRecognizer_create(),
}


def read_img_paths(directory):
    filename_list = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".jpg")
        ]
    )

    return filename_list

def read_ROIs(path):
    rois = loadmat(path)
    return rois["SubDir_Data"]

def read_labels(csv_path):
    labels = {}
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for idx, row in enumerate(reader, start=1):
            labels[idx] = int(row[0])

    return labels



def image_preprocess(path, top_left, bottom_right):

    out_img: np.ndarray = None
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(img_gray.shape[1], bottom_right[0]), min(
        img_gray.shape[0], bottom_right[1]
    )
    img_cropped = img_gray[y1:y2, x1:x2]
    img_resized = cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT))

    return img_resized


if __name__ == "__main__":

    random.seed(10)

    csv_path = "/home/nastia/agh/biometrics/caltech/caltech_labels.csv"
    labels = read_labels(csv_path)

    mat_path = "/home/nastia/agh/biometrics/caltech/ImageData.mat"
    rois = read_ROIs(mat_path)

    filenames = read_img_paths("/home/nastia/agh/biometrics/caltech")


    train_img: List[np.ndarray] = []
    train_lbl: List[int] = []
    test_img: List[np.ndarray] = []
    test_lbl: List[int] = []

    for img_path in filenames:
        img_index = int(os.path.basename(img_path).split("_")[1].split(".")[0])

        if img_index not in labels:
            continue

        label = labels[img_index]
        n_faces = sum(1 for l in labels.values() if l == label)

        roi = rois[:, img_index - 1]
        top_left = (int(roi[X_TOP_LEFT_IDX]), int(roi[Y_TOP_LEFT_IDX]))
        bottom_right = (int(roi[X_BOT_RIGHT_IDX]), int(roi[Y_BOT_RIGHT_IDX]))

        if n_faces >= N_FACES_THRESHOLD:
            img_input = image_preprocess(img_path, top_left, bottom_right)

            if random.random() <= 0.75:
                train_img.append(img_input)
                train_lbl.append(label)
            else:
                test_img.append(img_input)
                test_lbl.append(label)

    for method_name, method in FRs.items():
        method.train(train_img, np.array(train_lbl))
        correct_n = 0
        for i in tqdm(range(len(test_lbl))):
            predicted_label, _ = method.predict(test_img[i])
            if predicted_label == test_lbl[i]:
                correct_n += 1
        accuracy = correct_n / float(len(test_lbl))
        print("{} accuracy = {:.2f}".format(method_name, correct_n / float(len(test_lbl))))