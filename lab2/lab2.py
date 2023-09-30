#!/usr/bin/env python3
import cv2
import numpy as np


def main():
    # Load the input color image (eper.jpg)
    im = cv2.imread("../kepek/eper.jpg")

    # Task 1: Create images with one color channel set to zero
    b, g, r = cv2.split(im)
    zb = np.zeros_like(b)
    zg = np.zeros_like(g)
    zr = np.zeros_like(r)

    im_b = cv2.merge([zb, g, r])
    im_g = cv2.merge([b, zg, r])
    im_r = cv2.merge([b, g, zr])

    # Task 2: Create images with two color channels set to zero
    im_bg = cv2.merge([zb, zg, r])
    im_gr = cv2.merge([b, zg, zr])
    im_br = cv2.merge([zb, g, zr])

    # Task 3: Permute color channels
    im_permuted_1 = cv2.merge([r, g, b])
    im_permuted_2 = cv2.merge([r, b, g])
    im_permuted_3 = cv2.merge([g, r, b])
    im_permuted_4 = cv2.merge([g, b, r])
    im_permuted_5 = cv2.merge([b, r, g])
    im_permuted_6 = cv2.merge([b, g, r])

    # Task 4: Create images with one color channel replaced by its negative
    im_neg_b = cv2.merge([~b, g, r])
    im_neg_g = cv2.merge([b, ~g, r])
    im_neg_r = cv2.merge([b, g, ~r])

    # Task 5: Replace the Y channel with its negative using YCrCb encoding
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(im_ycrcb)
    im_neg_y = cv2.merge([~y, cr, cb])
    im_neg_ycrcb = cv2.cvtColor(im_neg_y, cv2.COLOR_YCrCb2BGR)

    # Task 6: Create the negative of the original image
    im_neg = ~im

    # Create a big image to display results
    im_big = np.zeros((im.shape[0] * 3, im.shape[1] * 6, im.shape[2]), dtype=np.uint8)
    im_big[:] = (128, 128, 255)  # Set the background color

    # Function to display images on the big image
    def show_my_image(im_big, im, index):
        im_copy = im.copy()
        row_start = (index // 6) * im.shape[0]
        row_end = row_start + im.shape[0]
        col_start = (index % 6) * im.shape[1]
        col_end = col_start + im.shape[1]
        im_big[row_start:row_end, col_start:col_end] = im_copy
        return index + 1

    index = 0

    # Display images on the big image
    index = show_my_image(im_big, im, index)  # Display the original RGB image
    index = show_my_image(im_big, im_b, index)
    index = show_my_image(im_big, im_g, index)
    index = show_my_image(im_big, im_r, index)
    index = show_my_image(im_big, im_gr, index)
    index = show_my_image(im_big, im_br, index)

    index = show_my_image(im_big, im_bg, index)
    index = show_my_image(im_big, im_permuted_6, index)
    index = show_my_image(im_big, im_permuted_5, index)
    index = show_my_image(im_big, im_permuted_4, index)
    index = show_my_image(im_big, im_permuted_3, index)
    index = show_my_image(im_big, im_permuted_2, index)

    index = show_my_image(im_big, im_permuted_1, index)
    index = show_my_image(im_big, im_neg_b, index)
    index = show_my_image(im_big, im_neg_g, index)
    index = show_my_image(im_big, im_neg_r, index)
    index = show_my_image(im_big, im_neg_ycrcb, index)
    show_my_image(im_big, im_neg, index)

    cv2.imshow("Ablak", im_big)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
