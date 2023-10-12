#!/bin/env python3
import cv2
import numpy as np

Ng = 256


def main():
    print("Lab 5: Histogram Equalization")
    image = load_image("../kepek/siena.jpg")
    # Draw original image and histogram
    cv2.imshow("Original Image", image)
    drawHist(image)
    # Draw equalized image and histogram
    eq_image = equalizeHist(image)
    drawHist(eq_image, "Equalized Histogram")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as a color image
    if image is None:
        print("Error: Image not found")
        exit(1)
    return image


def drawHist(im, name="Histogram"):
    # Create histogram image
    imHist = np.zeros((3 * Ng, Ng * im.shape[2], 3), dtype=np.uint8)

    imHist[0:Ng, 0 : 3 * Ng] = (Ng - 1, 0, 0)  # Blue background
    imHist[Ng : 2 * Ng, 0 : 3 * Ng] = (0, Ng - 1, 0)  # Green background
    imHist[2 * Ng : 3 * Ng, 0 : 3 * Ng] = (0, 0, Ng - 1)  # Red background

    channel_order = [0, 1, 2]  # Blue, Green, Red

    for idx, ch in enumerate(channel_order):
        roi = imHist[Ng * idx : Ng * (idx + 1), 0 : 3 * Ng]

        #  Calculate the histogram for the current channel
        hist = np.zeros(Ng, dtype=int)
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                hist[im[y, x, ch]] += 1

        # Search for the maximum value in the histogram
        maxCol = np.max(hist)
        for i in range(Ng):
            # Three strikes and you refactor
            colHeight = int(round(250 * hist[i] / maxCol))
            if colHeight > 0:
                cv2.rectangle(
                    roi,
                    (3 * i, Ng - colHeight),
                    (3 * (i + 1) - 1, Ng - 1),
                    (i, i, i),
                    thickness=cv2.FILLED,
                )
    # Display the histogram
    cv2.imshow(name, imHist)


def equalizeHist(im):
    if im.shape[2] % 2 == 0:
        return

    # If the image is color, convert it to YCrCb encoding
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)

    # Calculate the histogram for the 0-th channel
    H = np.zeros(Ng, dtype=int)
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            H[im[y, x, 0]] += 1

    # Calculate the new colors
    uj = np.zeros(Ng, dtype=int)
    sum = 0
    for n in range(Ng):
        uj[n] = (sum + H[n] // 2) // (im.shape[0] * im.shape[1] // Ng)
        if uj[n] > Ng - 1:
            uj[n] = Ng - 1
        sum += H[n]

    # Recolor the image using the new colors in the 0-th channel
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            im[y, x, 0] = uj[im[y, x, 0]]

    # If the image was color, convert it back to BGR encoding
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_YCrCb2BGR)

    # Display or return the processed image as needed
    cv2.imshow("Equalized Image", im)
    return im


if __name__ == "__main__":
    main()
