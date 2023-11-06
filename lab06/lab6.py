#!/bin/env python3
import cv2
import numpy as np


def main():
    print("Lab 6: Morphological operations")
    while True:
        print("What do you want to do?")
        print("Choose an exercise to run:")
        print("A. Erosion and Dilation on Binary Image")
        print("B. Erosion and Dilation with Elliptical Structuring Element")
        print("C. Drawing Lines and Applying Dilation and Erosion")
        print("D. Morphological Gradient on Color Image")
        print("E. Complex Image and Top-Hat & Black-Hat Transformations")
        print("Q. Quit")
        choice = input("Your choice: ")
        if choice == "A":
            f1()
        elif choice == "B":
            f2()
        elif choice == "C":
            f3()
        elif choice == "D":
            f4()
        elif choice == "E":
            f5()
        elif choice.lower() == "q":
            break
        else:
            print("Invalid choice")


def load_image(image_path, grayscale=False):
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as a color image
    if image is None:
        print("Error: Image not found")
        exit(1)
    return image


def f1():
    image = load_image("../kepek/pityoka.png", True)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for _ in range(10):
        image = cv2.erode(image, structuring_element)

    for _ in range(10):
        image = cv2.dilate(image, structuring_element)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f2():
    image = load_image("../kepek/pityoka.png", True)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(10):
        image = cv2.erode(image, structuring_element)

    for _ in range(10):
        image = cv2.dilate(image, structuring_element)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f3():
    image = load_image("../kepek/bond.jpg")
    black_line_color = (0, 0, 0)
    white_line_color = (255, 255, 255)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Draw black lines on the image
    cv2.line(image, (100, 100), (200, 200), black_line_color, 2)

    # Apply dilation
    image_dilated = cv2.dilate(image, structuring_element)

    # Draw white lines on the original image
    cv2.line(image, (300, 300), (400, 400), white_line_color, 2)

    # Apply erosion
    image_eroded = cv2.erode(image, structuring_element)

    cv2.imshow("Original image", image)
    cv2.imshow("Dilated image", image_dilated)
    cv2.imshow("Eroded image", image_eroded)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f4():
    # Load the color image (replace with your image path)
    image = load_image("../kepek/bond.jpg")

    # Define a structuring element for the morphological gradient (use a larger size)
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (9, 9)
    )  # Adjust the size as needed

    # Apply the morphological gradient using cv2.morphologyEx
    gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, structuring_element)

    # Display the original and gradient images
    cv2.imshow("Original Image", image)
    cv2.imshow("Morphological Gradient", gradient_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f5():
    # Load the image with white shapes on a black background (e.g., "kukac.png")
    white_shapes_image = load_image("../kepek/kukac.png", True)

    # Create an intensity gradient image
    gradient_image = np.zeros(white_shapes_image.shape, dtype=np.uint8)
    for x in range(gradient_image.shape[1]):
        gradient_image[:, x] = (x * 256) // gradient_image.shape[1]

    # Combine the background and gradient images
    complex_image = cv2.addWeighted(gradient_image, 0.9, white_shapes_image, 0.1, 0)

    # Apply the top-hat transformation
    structuring_element_size = 11  # Adjust the size as needed
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (structuring_element_size, structuring_element_size)
    )
    top_hat_image = cv2.morphologyEx(
        complex_image, cv2.MORPH_TOPHAT, structuring_element
    )

    # Threshold the resulting images
    _, top_hat_thresholded = cv2.threshold(top_hat_image, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow("White-Hat Complex Image", complex_image)

    complex_image = cv2.addWeighted(gradient_image, 0.9, white_shapes_image, -0.1, 25)

    # Apply the black-hat transformation
    black_hat_image = cv2.morphologyEx(
        complex_image, cv2.MORPH_BLACKHAT, structuring_element
    )

    # Threshold the black-hat image
    _, black_hat_thresholded = cv2.threshold(
        black_hat_image, 10, 255, cv2.THRESH_BINARY
    )

    # Display the complex image, top-hat result, and black-hat result
    cv2.imshow("Black-Hat Complex Image", complex_image)
    cv2.imshow("Top-Hat Result", top_hat_thresholded)
    cv2.imshow("Black-Hat Result", black_hat_thresholded)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
