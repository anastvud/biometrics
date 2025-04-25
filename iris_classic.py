import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d

from typing import Tuple

def remove_glare(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(H[150:])
    # plt.show()
    idx = np.argmax(H[150:]) + 151
    binary = cv2.threshold(image, idx, 250, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Binary", binary)
    # cv2.waitKey(0)

    st3 = np.ones((3, 3), dtype="uint8")
    st7 = np.ones((7, 7), dtype="uint8")

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, st3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, st3, iterations=2)

    im_floodfill = binary.copy()

    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, st7, iterations=1)
    _, _, stats, cents = cv2.connectedComponentsWithStats(im_out)
    cx, cy = 0, 0
    for st, cent in zip(stats, cents):
        if 1500 < st[4] < 3000:
            if 0.9 < st[2] / st[3] < 1.1:
                cx, cy = cent.astype(int)
                r = st[2] // 2
                cv2.circle(image, (cx, cy), r, (125, 125, 125), thickness=2)

    image = np.where(im_out, 64, image)
    image = cv2.medianBlur(image, 5)

    return image, cx, cy

def exploiding_circle(image: np.ndarray, cx: int, cy: int,
                      radius_init: int = 75, radius_step: int = 5,
                      radius_max: int = 200, seed_step: int = 10,
                      angle_step: int = 5, use_half_circle: bool = False) -> np.ndarray:
    h, w = image.shape

    def detect_circle(seed_cx, seed_cy, r_start, r_end, r_step, part='full'):
        max_diff = 0
        best_cx, best_cy, best_radius = 0, 0, 0
        seeds = [(seed_cx + dx, seed_cy + dy) for dx in [-seed_step, 0, seed_step]
                                           for dy in [-seed_step, 0, seed_step]]

        for seed_x, seed_y in seeds:
            brightness_by_radius = []
            for r in range(r_start, r_end, r_step):
                total_brightness = 0.0
                count = 0
                for angle in range(0, 360, angle_step):
                    # Use half-circle angles if requested
                    if part == 'left' and not (90 <= angle <= 270):
                        continue
                    if part == 'right' and not (angle <= 90 or angle >= 270):
                        continue

                    theta = np.deg2rad(angle)
                    x = int(seed_x + r * np.cos(theta))
                    y = int(seed_y + r * np.sin(theta))
                    if 0 <= x < w and 0 <= y < h:
                        total_brightness += float(image[y, x])
                        count += 1
                if count > 0:
                    brightness_by_radius.append(total_brightness / count)
                else:
                    brightness_by_radius.append(0)

            brightness_diff = np.abs(np.diff(brightness_by_radius))
            if len(brightness_diff) > 0:
                max_local_diff = np.max(brightness_diff)
                if max_local_diff > max_diff:
                    max_diff = max_local_diff
                    best_idx = np.argmax(brightness_diff)
                    best_radius = r_start + best_idx * r_step
                    best_cx, best_cy = seed_x, seed_y

        return best_cx, best_cy, best_radius

    # --- Stage 1: Pupil ---
    pupil_cx, pupil_cy, pupil_radius = detect_circle(
        cx, cy, radius_init, radius_max, radius_step, part='full'
    )
    cv2.circle(image, (pupil_cx, pupil_cy), pupil_radius, (255, 255, 255), 2)

    # --- Stage 2: Iris ---
    iris_cx, iris_cy, iris_radius = detect_circle(
        pupil_cx, pupil_cy, pupil_radius + 100, pupil_radius + 200, radius_step,
        part='left' if use_half_circle else 'full'
    )
    cv2.circle(image, (iris_cx, iris_cy), iris_radius, (180, 180, 180), 2)
    # cv2.imshow("Iris", image)
    # cv2.waitKey(0)
    return image

def gabor_filters(image: np.ndarray, pupil_cx: int, pupil_cy: int,
                  pupil_radius: int, iris_radius: int) -> str:
    # Prepare Gabor kernels
    kernels = []
    sigma = 2
    for theta in range(8):
        angle = theta / 8.0 * np.pi
        kernel = gabor_kernel(frequency=0.15, theta=angle, sigma_x=sigma, sigma_y=sigma)
        kernels.append(kernel)

    # Sampling points - polar coordinate system
    locations = []
    radius_steps = np.linspace(pupil_radius, iris_radius, 9)[1:]  # 8 steps
    angle_steps = np.linspace(-45, 45, 9)  # degrees, 8 steps

    for r in radius_steps:
        for angle_deg in angle_steps:
            angle_rad = np.deg2rad(angle_deg)
            x_left = int(pupil_cx - r * np.cos(angle_rad))
            y_left = int(pupil_cy + r * np.sin(angle_rad))
            x_right = int(pupil_cx + r * np.cos(angle_rad))
            y_right = int(pupil_cy + r * np.sin(angle_rad))

            if 0 <= x_left < image.shape[1] and 0 <= y_left < image.shape[0]:
                locations.append((x_left, y_left))
            if 0 <= x_right < image.shape[1] and 0 <= y_right < image.shape[0]:
                locations.append((x_right, y_right))

    # Extract patches and apply filters
    code = ""
    patch_size = 21
    half_patch = patch_size // 2
    padded = cv2.copyMakeBorder(image, half_patch, half_patch, half_patch, half_patch,
                                 cv2.BORDER_REFLECT)

    for (x, y) in locations:
        patch = padded[y:y + patch_size, x:x + patch_size]
        for kernel in kernels:
            real = convolve2d(patch, np.real(kernel), mode='valid')
            imag = convolve2d(patch, np.imag(kernel), mode='valid')
            real_sum = np.sum(real)
            imag_sum = np.sum(imag)
            code += '1' if real_sum > 0 else '0'
            code += '1' if imag_sum > 0 else '0'

    return code

def compare_iris_codes(code1: str, code2: str) -> float:
    differing_bits = sum(c1 != c2 for c1, c2 in zip(code1, code2))
    hamming_distance = differing_bits / len(code1)
    return hamming_distance

def get_iris_code(img):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        circle = exploiding_circle(img_no_glare, x, y)

        # Gabor filters
        code = gabor_filters(img_no_glare, x, y, 75, 150)

        return code

def match_iris_codes(path, codes) -> bool:
    img = cv2.imread(path)
    img_code = get_iris_code(img)

    name = path.split("/")[-1].split(".")[0]

    print(f"----------------------------------------------------------------")
    print(f"Comparing iris codes with {name}")
    for filename, code in codes.items():
        hamming_distance = compare_iris_codes(code, img_code)
        print(f"Hamming distance between {filename} and {name}: {hamming_distance:.2f}")
        if hamming_distance < 0.18:
            print(f"Match found with {filename} and {name}")




def main(data_path: str) -> None:
    # Get files from data path
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]

    codes = {}
    for filename in filename_list:
        # Read image
        img = cv2.imread(os.path.join(data_path, filename))

        code = get_iris_code(img)
        codes[filename.partition('.')[0]] = code

    # Match iris codes
    match_iris_codes("./iris_database_test/irisA_3.png", codes)
    match_iris_codes("./iris_database_test/irisB_3.png", codes)
    match_iris_codes("./iris_database_test/irisF.png", codes)




if __name__ == "__main__":
    data_path = "./iris_database_train"
    main(data_path)
