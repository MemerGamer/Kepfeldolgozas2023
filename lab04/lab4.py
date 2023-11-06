#!/bin/env python3
import cv2
import numpy as np

# Gradient masks
Mvp = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Mvn = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
Mfp = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
Mfn = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Define parameters for canny filter
max_low_threshold = 100
title_trackbar = "Min Threshold:"
ratio = 3
kernel_size = 3


def main():
    print("Lab 4: Image filtering")
    img_path = "../kepek/bogyok.jpg"
    while True:
        print("Choose a task:")
        print("1. Gradient filters")
        print("2. Canny filter")
        print("q. Quit")

        choice = input()
        if choice == "1":
            gradient_filters(img_path)
        elif choice == "2":
            canny_filters(img_path)
        elif choice == "q":
            break
        else:
            print("Invalid choice. Please select a valid task.")
    print("Bye!")


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found")
        exit(1)
    return image


def gradient_filters(image_path):
    image = load_image(image_path)

    # Apply the masks
    grad_vp = cv2.filter2D(image, -1, Mvp)
    grad_vn = cv2.filter2D(image, -1, Mvn)
    grad_fp = cv2.filter2D(image, -1, Mfp)
    grad_fn = cv2.filter2D(image, -1, Mfn)

    # Combine the results
    vertical_edges = (grad_vp + grad_vn) / 2
    horizontal_edges = (grad_fp + grad_fn) / 2
    all_edges = (vertical_edges + horizontal_edges) / 2

    # Apply thresholding
    cv2.threshold(vertical_edges, 10, 255, cv2.THRESH_BINARY, vertical_edges)

    # Display images
    cv2.imshow("Original", image)

    cv2.imshow("Gradient vp", grad_vp)
    cv2.imshow("Gradient vn", grad_vn)
    cv2.imshow("Gradient fp", grad_fp)
    cv2.imshow("Gradient fn", grad_fn)

    cv2.imshow("Vertical edges", vertical_edges)
    cv2.imshow("Horizontal edges", horizontal_edges)
    cv2.imshow("All edges", all_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_threshold_change(image, val, windows_name):
    low_threshold = val
    img_blur = cv2.blur(image, (3, 3))
    detected_edges = cv2.Canny(
        img_blur, low_threshold, low_threshold * ratio, kernel_size
    )
    mask = detected_edges != 0
    mask = mask.astype(np.uint8)
    img_canny = image * mask
    cv2.imshow(windows_name, img_canny)


def canny_filters(image_path):
    # Read the image
    image = load_image(image_path)

    # Create window for displaying canny image
    windows_name = "Canny Filter"
    cv2.namedWindow(windows_name)

    # Create trackbar for canny threshold
    cv2.createTrackbar(
        title_trackbar,
        windows_name,
        0,
        max_low_threshold,
        lambda val: on_threshold_change(image, val, windows_name),
    )
    on_threshold_change(image, 0, windows_name)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
