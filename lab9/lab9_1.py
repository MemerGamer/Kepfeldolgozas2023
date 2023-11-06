#!/bin/env python
import cv2
import numpy as np


def main():
    print("Lab 9: Hough Circle Detection")

    # Load the input image as a color image
    img = cv2.imread("../kepek/hod.jpg", cv2.IMREAD_COLOR)

    # Define the radius of the circles to be detected
    R = 89

    # Split the image into color channels
    b, g, r = cv2.split(img)

    # Choose the red channel for further processing
    imP = r

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(imP, 100, 200)

    # Create a black and white image for Hough Circle Detection
    imHough = np.zeros_like(imP)

    # Create a 2-D accumulator array for Hough Circle Transform
    hough_accumulator = np.zeros(imP.shape, dtype=np.uint16)

    # Iterate over the edge-detected image
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:
                # For each edge point, consider potential circles
                for theta in range(360):
                    a = int(x - R * np.cos(np.radians(theta)))
                    b = int(y - R * np.sin(np.radians(theta)))
                    if 0 <= a < imP.shape[1] and 0 <= b < imP.shape[0]:
                        hough_accumulator[b, a] += 1

    # Find the most-voted circle center in the Hough accumulator
    max_votes = np.max(hough_accumulator)
    center_y, center_x = np.where(hough_accumulator == max_votes)

    # Draw the detected circle on the original image
    for i in range(len(center_x)):
        cv2.circle(img, (center_x[i], center_y[i]), R, (0, 0, 255), 2)

    # Show the image with detected circles
    cv2.imshow("Detected Circles", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
