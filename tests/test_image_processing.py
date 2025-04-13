import unittest
import cv2 as cv
import numpy as np
from main import doloci_barvo_koze, zmanjsaj_sliko, obdelaj_sliko_s_skatlami, prestej_piksle_z_barvo_koze, filterBoxes, findBoundingBoxes

class TestImageProcessing(unittest.TestCase):

    def test_doloci_barvo_koze(self):
        sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        lower, upper = doloci_barvo_koze(sample_image, (20, 20), (80, 80))
        self.assertTrue(np.all(lower >= 0) and np.all(lower <= 255))
        self.assertTrue(np.all(upper >= 0) and np.all(upper <= 255))

    def test_zmanjsaj_sliko(self):
        sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized_image = zmanjsaj_sliko(sample_image, 50, 50)
        self.assertEqual(resized_image.shape, (50, 50, 3))

    def test_obdelaj_sliko_s_skatlami(self):
        sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        lower, upper = doloci_barvo_koze(sample_image, (20, 20), (80, 80))
        boxes = obdelaj_sliko_s_skatlami(sample_image, 25, 25, (lower, upper))
        self.assertEqual(len(boxes), 4)
        self.assertEqual(len(boxes[0]), 4)

    def test_prestej_piksle_z_barvo_koze(self):
        sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        lower, upper = doloci_barvo_koze(sample_image, (20, 20), (80, 80))
        count = prestej_piksle_z_barvo_koze(sample_image, (lower, upper))
        self.assertEqual(count, 10000)

    def test_filterBoxes(self):
        boxes = [[0, 0], [0, 0]]
        filtered_boxes = filterBoxes(boxes)
        self.assertEqual(filtered_boxes, [[False, False], [False, False]])

    def test_findBoundingBoxes(self):
        boxes = [[True, True, False], [True, True, False], [False, False, False]]
        bounding_boxes = findBoundingBoxes(boxes)
        self.assertEqual(len(bounding_boxes), 1)

if __name__ == '__main__':
    unittest.main()