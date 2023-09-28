#!/bin/env python3

import cv2 as cv


def main():
    print("Intro")
    im = cv.imread("eper.jpg", 1)

    if im is None:
        print("Nem sikerült megnyitni a képet.")
        return -1

    cv.imshow("Ez itt egy alma", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
