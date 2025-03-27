import cv2
import csv
import os
import random
import numpy as np
import torch
from scipy.io import loadmat
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import List

# Constants for image cropping
N_FACES_THRESHOLD = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

X_TOP_LEFT_IDX = 2
Y_TOP_LEFT_IDX = 3
X_BOT_RIGHT_IDX = 6
Y_BOT_RIGHT_IDX = 7

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
    img = cv2.imread(path)

    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(img.shape[1], bottom_right[0]), min(img.shape[0], bottom_right[1])
    img_cropped = img[y1:y2, x1:x2]
    img_resized = cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT))

    roi_float = img_resized / 255.0
    roi_tensor = torch.from_numpy(roi_float).permute(2, 0, 1).float()

    return roi_tensor

def collate_fn(x):
    return x[0]

if __name__ == "__main__":
    random.seed(10)

    csv_path = "/home/lsriw/vudvud/biometrics/caltech/caltech_labels.csv"
    labels = read_labels(csv_path)

    mat_path = "/home/lsriw/vudvud/biometrics/caltech/ImageData.mat"
    rois = read_ROIs(mat_path)

    filenames = read_img_paths("/home/lsriw/vudvud/biometrics/caltech")

    train_img: List[torch.Tensor] = []  
    train_lbl: List[int] = []
    test_img: List[torch.Tensor] = [] 
    test_lbl: List[int] = []

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device='cpu'
    )
    
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

    aligned_train = torch.stack(train_img)
    aligned_test = torch.stack(test_img)

    embeddings_train = resnet(aligned_train).detach().cpu().numpy()
    embeddings_test = resnet(aligned_test).detach().cpu().numpy()

    clf = cv2.ml.SVM_create()
    clf.setKernel(cv2.ml.SVM_LINEAR)
    clf.setType(cv2.ml.SVM_C_SVC)
    clf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    clf.train(embeddings_train, cv2.ml.ROW_SAMPLE, np.array(train_lbl))

    _, test_pred = clf.predict(embeddings_test)
    accuracy = accuracy_score(test_lbl, test_pred)

    print("SVM classifier accuracy on test set: {:.2f}%".format(accuracy * 100))

    cm = confusion_matrix(test_lbl, test_pred)
    print("Confusion Matrix:")
    print(cm)
