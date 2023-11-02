#!/bin/env python3

import cv2
import numpy as np


def load_image(image_path, mode=cv2.IMREAD_COLOR):
    image = cv2.imread(image_path, mode)
    if image is None:
        print("Error: Image not found")
        exit(1)
    return image


def main():
    print("Hello CV2")

    # Read and display the original image
    image = load_image("../kepek/esik.jpg")

    # Display images
    cv2.imshow("Original", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
