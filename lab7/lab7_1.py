#!/usr/bin/python3
import cv2
import numpy as np


def main():
    # Load the binary image of the amoeba
    imBe = cv2.imread("../kepek/amoba.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if imBe is None:
        print("Error: Image not loaded.")
        return

    # Display the input image
    cv2.imshow("Input Image", imBe)
    cv2.waitKey(0)

    # Perform the distance transformation
    imD = cv2.distanceTransform(imBe, cv2.DIST_L2, 3)

    # Convert the 32-bit output to 8-bit
    imD = cv2.normalize(imD, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display the distance map image
    cv2.imshow("Dilated Image", imD)
    cv2.waitKey(0)

    # Create a 5x5 circular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Dilate the distance map
    imX = cv2.dilate(imD, kernel)

    # TODO: Copy the dilated image back to the distance map where the original distance map is non-zero
    imKi = np.where(imD > 0, imX, imD)

    # Display the final image representing amoeba diameter
    cv2.imshow("Final Image", imKi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
