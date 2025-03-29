import cv2 as cv
import numpy as np
import time

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
color_tolerance = 1.0
use_hsv = True
show_mask = True
box_size = 50
box_tolerance = 0.2
neighbour_filter = 2
big_box = False

def doloci_barvo_koze(slika: np.ndarray, levo_zgoraj: tuple[int, int], desno_spodaj: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    global color_tolerance, use_hsv
    roi = slika[levo_zgoraj[1]:desno_spodaj[1], levo_zgoraj[0]:desno_spodaj[0]]
    if use_hsv:
        roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mean = np.mean(roi, axis=(0, 1))
    std = np.std(roi, axis=(0, 1))
    lowerBound = mean - color_tolerance * std
    upperBound = mean + color_tolerance * std
    lowerBound = np.clip(lowerBound, 0, 255)
    upperBound = np.clip(upperBound, 0, 255)
    return lowerBound, upperBound

def zmanjsaj_sliko(slika: np.ndarray, sirina: int, visina: int) -> np.ndarray:
    return cv.resize(slika,(sirina,visina))

def obdelaj_sliko_s_skatlami(slika: np.ndarray, sirina_skatle: int, visina_skatle: int, barva_koze: tuple[np.ndarray, np.ndarray]) -> list[list[int]]:
    global use_hsv
    numRows = slika.shape[0] // visina_skatle
    numCols = slika.shape[1] // sirina_skatle
    boxes = []
    for i in range(numRows):
        row = []
        for j in range(numCols):
            x0, y0 = j * sirina_skatle, i * visina_skatle
            x1, y1 = x0 + sirina_skatle, y0 + visina_skatle
            box = slika[y0:y1, x0:x1]
            if use_hsv:
                box = cv.cvtColor(box, cv.COLOR_BGR2HSV)
            numSkinPixels = prestej_piksle_z_barvo_koze(box, barva_koze)
            row.append(numSkinPixels)
        boxes.append(row)
    return boxes

def prestej_piksle_z_barvo_koze(slika: np.ndarray, barva_koze: tuple[np.ndarray, np.ndarray]) -> int:
    lowerBound, upperBound = barva_koze
    mask = cv.inRange(slika, lowerBound, upperBound)
    return cv.countNonZero(mask)

def chooseSkinArea(cap: cv.VideoCapture) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            return None
    ret, frame = cap.read()
    if not ret:
        return None
    frame = zmanjsaj_sliko(frame, WINDOW_WIDTH, WINDOW_HEIGHT)
    roi = cv.selectROI(frame, showCrosshair=False)
    cv.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        return None
    x0, y0, w, h = roi
    x1, y1 = x0 + w, y0 + h
    return (frame, (x0, y0), (x1, y1))

def filterBoxes(boxes: list[list[int]]) -> list[list[int]]:
    global box_tolerance, neighbour_filter
    filteredBoxes = [[False for _ in row] for row in boxes]
    numRows = len(boxes)
    numCols = len(boxes[0]) if numRows > 0 else 0

    for i, row in enumerate(boxes):
        for j, numSkinPixels in enumerate(row):
            if numSkinPixels > box_tolerance * box_size * box_size:
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < numRows and 0 <= nj < numCols and boxes[ni][nj] > box_tolerance * box_size * box_size:
                            neighbors += 1
                if neighbors >= neighbour_filter:
                    filteredBoxes[i][j] = True
    return filteredBoxes

def createControlWindow() -> None:
    global color_tolerance, use_hsv, show_mask, box_size, box_tolerance, neighbour_filter, big_box
    cv.namedWindow('Controls', 0)
    cv.createTrackbar('Color Tolerance', 'Controls', int(color_tolerance*10), 50, lambda x: None)
    cv.createTrackbar('Use HSV', 'Controls', 1 if use_hsv else 0, 1, lambda x: None)
    cv.createTrackbar('Show Mask', 'Controls', 1 if show_mask else 0, 1, lambda x: None)
    cv.createTrackbar('Box Size', 'Controls', box_size, 100, lambda x: None)
    cv.createTrackbar('Box Tolerance', 'Controls', int(box_tolerance*100), 100, lambda x: None)
    cv.createTrackbar('Neighbour Filter', 'Controls', neighbour_filter, 4, lambda x: None)
    cv.createTrackbar('Big Box', 'Controls', 1 if big_box else 0, 1, lambda x: None)