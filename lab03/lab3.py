#!/bin/env python3
import cv2
import numpy as np


def main():
    while True:
        print("Choose a task:")
        print("I. Altalanos konvolucios szurok - I")
        print("II. Gauss-fele blur szurok - II")
        print("III. Median szuro - III")
        print("q. Quit")

        choice = input()

        if choice == "I":
            altalanos_konvolucios_szurok()
        if choice == "II":
            gauss_fele_blur_szurok()
        if choice == "III":
            median_szuro()
        elif choice == "q":
            break
        else:
            print("Invalid choice. Please select a valid task.")


def altalanos_konvolucios_szurok():
    print("I. Altalanos konvolucios szurok")
    image_path = "../kepek/repulo.JPG"
    while True:
        print("Choose a task:")
        print("1. Task 1")
        print("2. Task 2")
        print("3. Task 3")
        print("q. Quit")

        choice = input()

        if choice == "1":
            task1(image_path)
        elif choice == "2":
            task2(image_path)
        elif choice == "3":
            task3(image_path)
        elif choice == "q":
            break
        else:
            print("Invalid choice. Please select a valid task.")


def shift_image(image, direction):
    return np.roll(image, direction, (0, 1))


def task1(image_path):
    print("Task 1: Shifting the image")

    imBe = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    while True:
        cv2.imshow("Shifted Image", imBe)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        # Shift the image one pixel to the right
        imBe = shift_image(imBe, 1)

    cv2.destroyAllWindows()


def task2(image_path):
    print("Task 2: Low-pass filter")

    kernel = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])

    imBe = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    while True:
        cv2.imshow("Filtered Image", imBe)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

        # Apply the filter to the image
        imBe = cv2.filter2D(imBe, -1, kernel)

    cv2.destroyAllWindows()


def task3(image_path):
    print("Task 3: High-pass filter")

    k_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    imBe = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    for k in k_values:
        cv2.imshow(f"Filtered Image (k={k})", imBe)
        cv2.waitKey(0)

        if cv2.waitKey == ord("q"):
            break

        # Define the high-pass filter kernel
        kernel = np.array(
            [
                [0, -k / 4, 0],
                [-k / 4, 1 + k, -k / 4],
                [0, -k / 4, 0],
            ]
        )

        # Apply the filter to the image
        imBe = cv2.filter2D(imBe, -1, kernel)

    cv2.destroyAllWindows()


def gauss_fele_blur_szurok():
    print("II. Gauss-fele blur szurok")
    image_path = "../kepek/eper.jpg"
    imBe = cv2.imread(image_path, cv2.IMREAD_COLOR)
    k_values = [3, 5, 7, 9, 11]

    for k in k_values:
        # Apply the filter to the image
        blurred_image = cv2.blur(imBe, (k, k))

        # Apply the gaussian blur filter to the image
        gaussian_blurred_image = cv2.GaussianBlur(imBe, (k, k), 0)

        cv2.imshow(f"Blur (k={k})", blurred_image)
        cv2.imshow(f"Gaussian Blur (k={k})", gaussian_blurred_image)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def median_szuro():
    print("III. Median szuro")
    image_path = "../kepek/repulo.JPG"

    imBe = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Task 1: Drawing lines on the image
    for db in range(20):
        pt1 = (np.random.randint(0, imBe.shape[1]), np.random.randint(0, imBe.shape[0]))
        pt2 = (np.random.randint(0, imBe.shape[1]), np.random.randint(0, imBe.shape[0]))
        line_color = (0, 0, 0)  # Black
        line_thickness = 1 + db % 2
        cv2.line(imBe, pt1, pt2, line_color, line_thickness)

    cv2.imshow("Lines on Image", imBe)
    cv2.waitKey(0)

    k_values = [3, 5, 7, 9, 11, 21]  # Increasing neighborhood sizes

    for k in k_values:
        # Apply the median filter to the image
        median_filtered_image = cv2.medianBlur(imBe, k)

        cv2.imshow(f"Median Filter (k={k})", median_filtered_image)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Task 2: Create a two-color image and apply median filtering
    white_amoba = np.zeros(imBe.shape, dtype=np.uint8)
    white_amoba.fill(255)  # Set the entire image to white
    cv2.fillPoly(
        white_amoba,
        [np.array([[150, 200], [300, 200], [225, 350]], dtype=np.int32)],
        (0, 0, 0),
    )  # Draw a black amoeba
    cv2.imshow("Amoeba", white_amoba)

    for k in range(21, 201, 10):  # Vary the kernel size from 21x21 to 201x201
        # Apply the median filter to the amoeba image
        amoeba_filtered = cv2.medianBlur(white_amoba, k)
        cv2.imshow(f"Amoeba Median Filter (k={k})", amoeba_filtered)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
