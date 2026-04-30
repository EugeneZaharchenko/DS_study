import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


BASE_DIR = os.path.dirname(__file__)


# ---------- Utils ----------

def load_image(filename):
    path = os.path.join(BASE_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Image not found: {path}")

    return image


def show(image, title=""):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# ---------- Preprocessing ----------

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed


# ---------- Contours ----------

def find_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# ---------- Rectangle detection ----------

def detect_rectangles(image, contours):
    output = image.copy()
    count = 0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
            count += 1

    return output, count


# ---------- Circle detection (contour-based) ----------

def detect_circles_contours(image, contours):
    output = image.copy()
    count = 0

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if 0.8 < circularity < 1.2:
            cv2.drawContours(output, [c], -1, (255, 0, 0), 3)
            count += 1

    return output, count


# ---------- Circle detection (Hough - better) ----------

def detect_circles_hough(image):
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=200
    )

    count = 0

    if circles is not None:
        circles = circles[0].astype("int")

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (255, 0, 0), 3)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            count += 1

    return output, count


# ---------- Main pipeline ----------

def process_image(filename):
    image = load_image(filename)
    binary = preprocess(image)
    contours = find_contours(binary)

    rect_img, rect_count = detect_rectangles(image, contours)
    circ_img, circ_count = detect_circles_contours(image, contours)

    print(f"{filename}:")
    print(f"  Rectangles: {rect_count}")
    print(f"  Circles (contour): {circ_count}")

    show(rect_img, "Rectangles")
    show(circ_img, "Circles (contour)")

    # Optional: better circle detection
    hough_img, hough_count = detect_circles_hough(image)
    print(f"  Circles (Hough): {hough_count}")
    show(hough_img, "Circles (Hough)")


# ---------- Entry ----------

if __name__ == "__main__":
    process_image("Image_1.jpg")
    process_image("Image_2.jpg")
    process_image("Image_3.jpg")
    process_image("test_circles.jpg")
