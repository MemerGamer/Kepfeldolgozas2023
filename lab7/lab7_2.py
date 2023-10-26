#!/usr/bin/python3
import cv2
import numpy as np


def hit_or_miss_skeletonization(image_path):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Failed to load the image.")
        return

    # Display the original image
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # Create an empty output image to hold values
    thin = np.zeros(image.shape, dtype="uint8")

    # Structuring Element (Cross)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Loop until erosion leads to an empty set
    while cv2.countNonZero(image) != 0:
        # Erosion
        erode = cv2.erode(image, kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset, thin)
        # Set the eroded image for the next iteration
        image = erode.copy()

    # Display the result image
    cv2.imshow("Hit-or-Miss Thinned Image", thin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hit_or_miss_skeletonization("../kepek/kep.jpg")
