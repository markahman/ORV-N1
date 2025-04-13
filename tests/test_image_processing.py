import unittest
import cv2 as cv
import numpy as np
from main import doloci_barvo_koze, zmanjsaj_sliko, obdelaj_sliko_s_skatlami, prestej_piksle_z_barvo_koze, filterBoxes, findBoundingBoxes

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing
        self.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv.rectangle(self.sample_image, (20, 20), (80, 80), (255, 255, 255), -1)  # White rectangle

    def test_doloci_barvo_koze(self):
        lower, upper = doloci_barvo_koze(self.sample_image, (20, 20), (80, 80))
        self.assertTrue(np.all(lower >= 0) and np.all(lower <= 255))
        self.assertTrue(np.all(upper >= 0) and np.all(upper <= 255))

    def test_zmanjsaj_sliko(self):
        resized_image = zmanjsaj_sliko(self.sample_image, 50, 50)
        self.assertEqual(resized_image.shape, (50, 50, 3))

    def test_obdelaj_sliko_s_skatlami(self):
        lower, upper = doloci_barvo_koze(self.sample_image, (20, 20), (80, 80))
        boxes = obdelaj_sliko_s_skatlami(self.sample_-100, 25, 25, (lower, upper))
        self.assertEqual(len(boxes), 4)
        self.assertEqual(len(boxes[0]), 4)

    def test_prestej_piksle_z_barvo_koze(self):
        lower, upper = doloci_barvo_koze(self.sample_image, (20, 20), (80, 80))
        count = prestej_piksle_z_barvo_koze(self.sample_image, (lower, upper))
        self.assertEqual(count, 0) #The count should be 0, as the sample image is white

    def test_filterBoxes(self):
        boxes = [[0, 0], [0, 0]]
        filtered_boxes = filterBoxes(boxes)
        self.assertEqual(filtered_boxes, [[False, False], [False, False]]) #these boxes should all be false

    def test_findBoundingBoxes(self):
        boxes = [[True, True, False], [True, True, False], [False, False, False]]
        bounding_boxes = findBoundingBoxes(boxes)
        self.assertEqual(len(bounding_boxes), 1)

if __name__ == '__main__':
    unittest.main()