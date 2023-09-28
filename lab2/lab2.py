#!/bin/env python3
import cv2 as cv
import numpy as np


def main():
    # Load the colored image
    im = cv.imread("../kepek/eper.jpg")

    # Initialize the index for placing images in the mosaic
    index = 0

    # Create a larger canvas (mozaik) for displaying images
    imBig = np.zeros((im.shape[0] * 6, im.shape[1] * 3, im.shape[2]), dtype=np.uint8)
    imBig[:, :] = (128, 128, 255)  # Set the background color to light blue

    # Create an all-zero matrix of the same size as the input image
    z = np.zeros_like(im)

    # Task 1: Separate color channels (R, G, B)
    b, g, r = cv.split(im)
    showMyImage(imBig, cv.merge((b, z, z)), index)
    showMyImage(imBig, cv.merge((z, g, z)), index)
    showMyImage(imBig, cv.merge((z, z, r)), index)

    # Task 2: Replace two color channels with zeros
    showMyImage(imBig, cv.merge((b, g, z)), index)
    showMyImage(imBig, cv.merge((b, z, r)), index)
    showMyImage(imBig, cv.merge((z, g, r)), index)

    # Task 3: Permute color channels
    showMyImage(imBig, cv.merge((r, g, b)), index)
    showMyImage(imBig, cv.merge((g, b, r)), index)
    showMyImage(imBig, cv.merge((b, r, g)), index)

    # Task 4: Replace one color channel with its negative
    ib = cv.bitwise_not(b)
    ig = cv.bitwise_not(g)
    ir = cv.bitwise_not(r)
    showMyImage(imBig, cv.merge((ib, g, r)), index)
    showMyImage(imBig, cv.merge((b, ig, r)), index)
    showMyImage(imBig, cv.merge((b, g, ir)), index)

    # Task 5: Replace Y channel with its negative in YCrCb color space
    ycrcb = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    ny = cv.bitwise_not(y)
    showMyImage(
        imBig,
        cv.cvtColor(cv.merge((ny, cr, cb), flags=cv.COLOR_YCrCb2BGR), cv.COLOR_BGR2RGB),
        index,
    )

    # Task 6: Add the negative of the original image to the mosaic
    negative_im = cv.bitwise_not(im)
    showMyImage(imBig, negative_im, index)

    cv.destroyAllWindows()


def showMyImage(imBig, im, index):
    row = index // 6
    col = index % 6

    # Resize the input image to match the dimensions of imBig
    im_resized = cv.resize(im, (imBig.shape[1] // 3, imBig.shape[0] // 6))

    # Ensure that the resized image has 3 channels (BGR)
    if im_resized.shape[2] != 3:
        im_resized = cv.cvtColor(
            im_resized, cv.COLOR_GRAY2BGR
        )  # Convert to BGR if not already

    imBig[
        row * im_resized.shape[0] : (row + 1) * im_resized.shape[0],
        col * im_resized.shape[1] : (col + 1) * im_resized.shape[1],
    ] = im_resized

    cv.imshow("Ablak", imBig)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
