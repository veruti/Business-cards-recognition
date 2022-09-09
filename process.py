import argparse

import cv2 as cv
import numpy as np
import pytesseract
from pytesseract import Output
from src.text_detector.openvio_text_detector_3 import OpenVinoTextDetector3


def main(**kwargs):
    detector = OpenVinoTextDetector3()
    image: np.ndarray = cv.imread(kwargs["image_path"])
    detector.predict(image=image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        help="Location of image (relative or absolute)",
        default="./dataset/Stanford-business-cards/027.jpg",
    )

    args = parser.parse_args()
    main(**args.__dict__)
