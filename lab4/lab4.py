#!/bin/env python3
import cv2
import numpy as np

# Gradient masks
Mvp = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Mvn = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
Mfp = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
Mfn = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def main():
    image = cv2.imread("../kepek/bogyok.jpg", cv2.IMREAD_GRAYSCALE)

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


if __name__ == "__main__":
    main()
