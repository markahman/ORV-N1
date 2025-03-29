import cv2 as cv
import numpy as np
import time

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
color_tolerance = 1.0
use_hsv = True

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