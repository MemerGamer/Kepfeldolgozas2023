#!/bin/env python3

import cv2 as cv
import numpy as np


def main():
    im1 = cv.imread("../kepek/3.JPG")
    im2 = cv.imread("../kepek/5.JPG")
    cv.waitKey(0)
    im3 = im2.copy()

    for q in np.arange(0.0, 1.1, 0.02):
        cv.addWeighted(im1, 1.0 - q, im2, q, 0, im3)
        cv.imshow("Film", im3)
        cv.waitKey(100)
        if q < 0.67 and q > 0.65:
            cv.imwrite("keverek.bmp", im3)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
