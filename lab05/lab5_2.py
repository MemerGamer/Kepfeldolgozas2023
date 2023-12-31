#!/bin/env python3
import cv2
import numpy as np


def main():
    print("Lab 5: Color calibration")
    image = load_image("../kepek/forest.png")
    while True:
        print("What do you want to do?")
        print("1. Adjust brightness, contrast and gamma")
        print("2. Adjust brightness, contrast and saturation")
        print("Q. Quit")
        choice = input("Your choice: ")
        if choice == "1":
            f1(image)
        elif choice == "2":
            f2(image)
        elif choice.lower() == "q":
            break
        else:
            print("Invalid choice")


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as a color image
    if image is None:
        print("Error: Image not found")
        exit(1)
    return image


def f1(image):
    # Adjust brightness and gamma
    brightness = 0
    contrast = 2
    gamma = 2

    lut = np.zeros((1, 256, 3), dtype=np.uint8)

    for i in range(256):
        norm_value = i / 255.0
        val = min(max(pow(norm_value, gamma) * contrast + brightness, 0), 1)
        val = int(val * 255)
        lut[0, i] = (val, val, val)

    # Apply the lookup table to the image
    output_image = cv2.LUT(image, lut)

    # Display the result
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f2(input_image):
    # Initialize parameters
    b = 0
    c = 2
    s = 2

    # Calculate the custom color matrix
    t = (1.0 - c) / 2.0
    sr = (1 - s) * 0.3086
    sg = (1 - s) * 0.6094
    sb = (1 - s) * 0.0820
    custom_matrix = np.array(
        [
            [c * (sr + s), c * sr, c * sr, 0],
            [c * sg, c * (sg + s), c * sg, 0],
            [c * sb, c * sb, c * (sb + s), 0],
            [t + b, t + b, t + b, 1],
        ]
    )

    # Convert image to float and add alpha channel
    output_image = (input_image / 255.0).astype(np.float32)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)

    # Reshape the image for matrix multiplication and apply the custom matrix
    output_image = output_image.reshape(
        (output_image.shape[0] * output_image.shape[1], 4)
    )
    output_image = np.dot(output_image, custom_matrix)
    output_image = output_image.reshape((input_image.shape[0], input_image.shape[1], 4))

    # Convert the image back to 8-bit BGR format
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    output_image = output_image[:, :, :3]
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

    # Display the result
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
