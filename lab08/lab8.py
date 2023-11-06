#!/bin/env python3
import cv2
import numpy as np


def main():
    # Read and display the original image
    original_image = cv2.imread("../kepek/agy.bmp", cv2.IMREAD_COLOR)
    cv2.imshow("Original Image", original_image)

    # Values of m for segmentation
    m_values = [1.5, 2.0, 3.0, 4.0]

    for m in m_values:
        segmented_image = segment_image(original_image, m)
        cv2.imshow(f"Segmented Image (m={m})", segmented_image[0])
        draw_fuzzy_membership_function(
            segmented_image[1], m, segmented_image[2], segmented_image[3]
        )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_image(image, m):
    Ng = 256
    c = 3
    eps = 1e-8

    H = np.zeros(Ng, dtype=int)
    u = np.zeros((c, Ng), dtype=float)
    v = np.zeros(c, dtype=float)

    for row in image:
        for pixel in row:
            H[pixel] += 1

    for i in range(c):
        v[i] = 255.0 * (i + 1) / (2.0 * c)

    u, v = calculate_fuzzy_partition(u, v, H, m, eps)

    # Create a lookup table for pixel mapping
    lut = np.zeros((1, Ng), dtype=np.uint8)
    for l in range(Ng):
        winner = 0
        for i in range(1, c):
            if u[i, l] > u[winner, l]:
                winner = i
        lut[0, l] = round(v[winner])

    # Apply the lookup table to the image for segmentation
    segmented_image = cv2.LUT(image, lut)

    return [segmented_image, u, c, Ng]


def draw_fuzzy_membership_function(u, m, c, Ng):
    # Draw the fuzzy membership functions
    # Create a new black image for the membership functions
    membership_function_image = np.zeros((400, 768, 3), dtype=np.uint8)

    # Draw the membership functions
    for i in range(c):
        for l in range(Ng):
            x = 1 + 3 * l
            y = round(400.0 * (1.0 - u[i, l]))
            color = (0, 0, 0)  # Black background

            if i == 0:
                color = (255, 0, 0)  # Red
            elif i == 1:
                color = (0, 255, 0)  # Green
            elif i == 2:
                color = (0, 0, 255)  # Blue

            cv2.circle(membership_function_image, (x, y), 2, color, -1)

    cv2.imshow(f"Fuzzy Membership Function (m={m})", membership_function_image)
    cv2.waitKey(0)


def calculate_fuzzy_partition(u, v, H, m, eps):
    c, Ng = len(v), len(H)
    mm = -1.0 / (m - 1)
    d2 = np.zeros((c, Ng))

    for _ in range(20):
        for l in range(Ng):
            for i in range(c):
                d2[i, l] = (l - v[i]) ** 2
            winner = 0
            for i in range(c):
                if d2[winner, l] > d2[i, l]:
                    winner = i
            if d2[winner, l] < eps:
                for i in range(c):
                    u[i, l] = 0.0
                u[winner, l] = 1.0
            else:
                total = 0
                for i in range(c):
                    u[i, l] = d2[i, l] ** mm
                    total += u[i, l]
                for i in range(c):
                    u[i, l] /= total

        for i in range(c):
            sum_up, sum_dn = 0.0, 0.0
            for l in range(Ng):
                sum_up += H[l] * (u[i, l] ** m) * l
                sum_dn += H[l] * (u[i, l] ** m)
            v[i] = sum_up / sum_dn

    return u, v


if __name__ == "__main__":
    main()
