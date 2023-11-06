#!/bin/env python
import cv2
import numpy as np


def main():
    print("Lab 9: Hough Circle Detection")

    # Load the input image
    img = cv2.imread("../kepek/hod.jpg", cv2.IMREAD_COLOR)

    # Set the radius of the circles
    R = 89

    # Split the image into color channels
    b, g, r = cv2.split(img)

    # Use the red channel for circle detection (you can change this as needed)
    imP = r

    # Run a Hough Circle Transform
    circles = cv2.HoughCircles(
        imP,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=100,
        param2=30,
        minRadius=R - 10,
        maxRadius=R + 10,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(img, center, radius, (0, 0, 255), 2)

    cv2.imshow("Detected Circles", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
