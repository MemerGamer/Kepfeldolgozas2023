#!/usr/bin/python3
import cv2
import numpy as np


def main():
    # Load the binary image of the amoeba
    amoeba_image = cv2.imread("../kepek/amoba.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if amoeba_image is None:
        print("Error: Image not loaded.")
        return

    # Display the input image
    cv2.imshow("Input Image", amoeba_image)
    cv2.waitKey(0)

    # Perform the distance transformation
    distance_map = cv2.distanceTransform(amoeba_image, cv2.DIST_L2, 3)

    # Convert the 32-bit output to 8-bit
    distance_map = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display the distance map image
    cv2.imshow("Dilated Image", distance_map)
    cv2.waitKey(0)

    # Create a 5x5 circular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Dilate the distance map
    dilated_distance_map = cv2.dilate(distance_map, kernel)

    # TODO: Copy the dilated image back to the distance map where the original distance map is non-zero
    result = np.where(distance_map > 0, dilated_distance_map, distance_map)

    # Display the final image representing amoeba diameter
    cv2.imshow("Final Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
