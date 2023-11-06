#!/bin/env python

import cv2
import numpy as np


def regionGrowing(image, p0):
    count = 0
    fifo = [(0, 0)] * 0x100000
    nextIn = 0
    nextOut = 0
    pbf = p0
    pja = p0
    if image[p0[0], p0[1]] < 128:
        return (None, None)
    fifo[nextIn] = p0
    nextIn += 1
    image[p0[0], p0[1]] = 100
    while nextIn > nextOut:
        p = fifo[nextOut]
        nextOut += 1
        count += 1
        if p[0] > 0:
            if image[p[0] - 1, p[1]] > 128:
                fifo[nextIn] = (p[0] - 1, p[1])
                nextIn += 1
                image[p[0] - 1, p[1]] = 100
                if pbf[0] > p[0] - 1:
                    pbf = (p[0] - 1, pbf[1])
        if p[0] < image.shape[0] - 1:
            if image[p[0] + 1, p[1]] > 128:
                fifo[nextIn] = (p[0] + 1, p[1])
                nextIn += 1
                image[p[0] + 1, p[1]] = 100
                if pja[0] < p[0] + 1:
                    pja = (p[0] + 1, pja[1])
        if p[1] > 0:
            if image[p[0], p[1] - 1] > 128:
                fifo[nextIn] = (p[0], p[1] - 1)
                nextIn += 1
                image[p[0], p[1] - 1] = 100
                if pbf[1] > p[1] - 1:
                    pbf = (pbf[0], p[1] - 1)
        if p[1] < image.shape[1] - 1:
            if image[p[0], p[1] + 1] > 128:
                fifo[nextIn] = (p[0], p[1] + 1)
                nextIn += 1
                image[p[0], p[1] + 1] = 100
                if pja[1] < p[1] + 1:
                    pja = (pja[0], p[1] + 1)
    return count, pbf, pja


def main():
    # cap = cv2.VideoCapture("../videok/IMG_6909.MOV") # I couldn't get this video to work because have missing codecs
    cap = cv2.VideoCapture("../videok/MVI_0022.AVI")

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    roiSize = 0
    nrRect = 0
    roi = (0, 0, 0, 0)
    first_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Play the video in color
        cv2.imshow("Color", frame)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray_frame)

        # Edge detection using Canny
        edges = cv2.Canny(gray_frame, 100, 200)
        cv2.imshow("Edge Detection", edges)

        # Apply Median Blur
        blurred_frame = cv2.medianBlur(frame, 5)
        cv2.imshow("Median Blur", blurred_frame)

        # Apply Gaussian Blur
        gaussian_blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        cv2.imshow("Gaussian Blur", gaussian_blur_frame)

        # Apply Low-Pass Filter
        low_pass_frame = cv2.pyrDown(frame)
        cv2.imshow("Low-Pass Filter", low_pass_frame)

        # Apply High-Pass Filter (Laplace)
        laplace_frame = cv2.Laplacian(frame, cv2.CV_64F)
        cv2.imshow("High-Pass Filter (Laplace)", laplace_frame)

        # Apply Histogram Equalization
        gray_frame_equalized = cv2.equalizeHist(gray_frame)
        cv2.imshow("Histogram Equalization", gray_frame_equalized)

        # Resize the frame
        resized_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Resized", resized_frame)

        # Detect motion
        if first_frame is None:
            first_frame = gray_frame
        else:
            frame_diff = cv2.absdiff(first_frame, gray_frame)
            _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            thresholded = cv2.erode(thresholded, None, iterations=2)
            thresholded = cv2.dilate(thresholded, None, iterations=2)

            # Iterate through the thresholded image and use regionGrowing
            for x in range(thresholded.shape[0]):
                for y in range(thresholded.shape[1]):
                    if thresholded[x, y] > 128:
                        res, pbf, pja = regionGrowing(thresholded, (x, y))
                        if res > 500:
                            if nrRect == 0 or roiSize < res:
                                roi = (
                                    pbf[1],
                                    pbf[0],
                                    pja[1] - pbf[1] + 1,
                                    pja[0] - pbf[0] + 1,
                                )
                                roiSize = res
                            nrRect += 1

            if nrRect > 0:
                frame = cv2.rectangle(
                    frame,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(27) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
