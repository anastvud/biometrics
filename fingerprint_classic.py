import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict


def read_img_paths(directory: str) -> List[str]:
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")
    ]


def img_preprocessing(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_features(
    image: np.ndarray, alg_type: str
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    assert alg_type == "sift" or alg_type == "orb"
    detector = cv2.SIFT_create() if alg_type == "sift" else cv2.ORB_create()

    keypoints: List[np.ndarray] = []
    descriptors: List[np.ndarray] = []
    keypoints, descriptors = detector.detectAndCompute(image, None)

    return keypoints, descriptors


def match_fingerprints(
    query_features: Dict[str, Dict[str, List[np.ndarray]]],
    reference_features: Dict[str, Dict[str, np.ndarray]],
    alg_type: str,
    threshold=0.75,
) -> Tuple[List[str], List[str]]:
    assert alg_type == "sift" or alg_type == "orb"
    matcher = (
        cv2.BFMatcher(cv2.NORM_L2)
        if alg_type == "sift"
        else cv2.BFMatcher(cv2.NORM_HAMMING)
    )
    y_pred: List[str] = list()
    y_test: List[str] = list()

    for query_id, samples in query_features.items():
        for sample in samples[alg_type]:
            best_match_id, best_match_score = None, 0

            for ref_id, ref_feats in reference_features.items():
                ref_descriptors = ref_feats[alg_type]
                if ref_descriptors is None or sample is None:
                    continue

                matches = matcher.knnMatch(sample, ref_descriptors, k=2)
                good_matches = (
                    [m for m, n in matches if m.distance < threshold * n.distance]
                    if len(matches[0]) > 1
                    else []
                )

                if len(good_matches) > best_match_score:
                    best_match_score = len(good_matches)
                    best_match_id = ref_id

            y_test.append(query_id)
            y_pred.append(best_match_id if best_match_id else "")

    return y_pred, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    # Read filepaths
    filepaths = read_img_paths(args.path)

    # Get features
    features_base = dict()
    features2match = dict()
    for path in tqdm(filepaths):
        gray = img_preprocessing(path)
        kp_sift, dsc_sift = get_features(gray, "sift")
        kp_orb, dsc_orb = get_features(gray, "orb")

        filename = path[-9:-4] # get 109_6
        id_finger, id_sample = filename.split("_") # split into 109 and 6

        if id_sample == "1":
            features_base[id_finger] = {
                "sift": dsc_sift,
                "orb": dsc_orb,
            }
        else:
            if id_finger not in features2match:
                features2match[id_finger] = {"sift": [], "orb": []}
            features2match[id_finger]["sift"].append(dsc_sift)
            features2match[id_finger]["orb"].append(dsc_orb)

    # Match features
    preds, gt = match_fingerprints(features2match, features_base, "sift")
    print("--- SIFT ---")
    print(classification_report(gt, preds))

    preds, gt = match_fingerprints(features2match, features_base, "orb")
    print("--- ORB ---")
    print(classification_report(gt, preds))
