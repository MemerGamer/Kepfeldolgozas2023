#!/bin/env python
import cv2
import numpy as np


def getColor(im, x, y):
    # Get the color (BGR) at the specified (x, y) location in the image.
    blue, green, red = im[y, x]
    return blue, green, red


def setColor(im, x, y, blue, green, red):
    # Set the color (BGR) at the specified (x, y) location in the image.
    im[y, x] = [blue, green, red]


def getGray(im, x, y):
    # Get the grayscale value at the specified (x, y) location in the image.
    gray_value = im[y, x]
    return gray_value


def setGray(im, x, y, v):
    # Set the grayscale value at the specified (x, y) location in the image.
    im[y, x] = v


def compare(p1, p2):
    # Helper function for quicksort. Compares two elements.
    if p1 > p2:
        return 1
    elif p1 < p2:
        return -1
    return 0


def watershed():
    bits = [1, 2, 4, 8, 16, 32, 62, 128]
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    imColor = cv2.imread("../kepek/3.JPG", 1)
    im0 = cv2.cvtColor(imColor, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Color", imColor)
    cv2.waitKey()
    cv2.imshow("Grey", im0)
    cv2.waitKey()
    imG = im0.copy()
    imE = im0.copy()
    imKi = im0.copy()
    imBe = im0.copy()
    imSegm = imColor.copy()
    imSegmMed = imColor.copy()
    imMap = im0.copy()
    imL = np.zeros_like(im0, dtype=np.int16)
    imBlue, imGreen, imRed = cv2.split(imColor)
    imSum = np.zeros_like(im0, dtype=np.uint8)
    imL = cv2.Sobel(imBlue, cv2.CV_16S, 1, 0)
    imE = cv2.convertScaleAbs(imL)
    imL = cv2.Sobel(imBlue, cv2.CV_16S, 0, 1)
    imG = cv2.convertScaleAbs(imL)
    imE = imG + imE
    imSum = cv2.addWeighted(imSum, 1, imG, 0.33, 0)
    imL = cv2.Sobel(imGreen, cv2.CV_16S, 1, 0)
    imE = cv2.convertScaleAbs(imL)
    imL = cv2.Sobel(imGreen, cv2.CV_16S, 0, 1)
    imG = cv2.convertScaleAbs(imL)
    imE = imG + imE
    imSum = cv2.addWeighted(imSum, 1, imG, 0.33, 0)
    imL = cv2.Sobel(imRed, cv2.CV_16S, 1, 0)
    imE = cv2.convertScaleAbs(imL)
    imL = cv2.Sobel(imRed, cv2.CV_16S, 0, 1)
    imG = cv2.convertScaleAbs(imL)
    imE = imG + imE
    imSum = cv2.addWeighted(imSum, 1, imG, 0.33, 0)
    imG = cv2.GaussianBlur(imG, (9, 9), 0)
    cv2.imshow("Gradiens", imG)
    cv2.waitKey()

    # step 0
    imE = cv2.erode(imG, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    imSegm[:, :, :] = 50
    imSegmMed[:, :, :] = 150
    imBe[:, :] = 0
    imKi[:, :] = 8
    imMap[:, :] = 0

    # step 1
    for x in range(imBe.shape[0]):
        for y in range(imBe.shape[1]):
            fp = imG[x, y]
            q = imE[x, y]
            if q < fp:
                for irany in range(8):
                    if (
                        x + dx[irany] >= 0
                        and x + dx[irany] < imBe.shape[0]
                        and y + dy[irany] >= 0
                        and y + dy[irany] < imBe.shape[1]
                    ):
                        fpv = imG[x + dx[irany], y + dy[irany]]
                        if fpv == q:
                            imKi[x, y] = irany
                            imMap[x, y] = 255
                            volt = imBe[x + dx[irany], y + dy[irany]]
                            adunk = bits[irany]
                            lesz = volt | adunk
                            imBe[x + dx[irany], y + dy[irany]] = lesz
                            break

    cv2.imshow("Ablak", imMap)
    cv2.waitKey()

    # step 2
    fifo = [(0, 0)] * (imBe.shape[0] * imBe.shape[1])
    nextIn = 0
    nextOut = 0

    for x in range(imBe.shape[0]):
        for y in range(imBe.shape[1]):
            fp = imG[x, y]
            pout = imKi[x, y]
            if pout == 8:
                continue
            added = 0
            for irany in range(8):
                if (
                    x + dx[irany] >= 0
                    and x + dx[irany] < imBe.shape[0]
                    and y + dy[irany] >= 0
                    and y + dy[irany] < imBe.shape[1]
                ):
                    fpv = imG[x + dx[irany], y + dy[irany]]
                    pvout = imKi[x + dx[irany], y + dy[irany]]
                    if fpv == fp and pvout == 8:
                        if added == 0:
                            fifo[nextIn] = (x, y)
                            nextIn += 1
                            added += 1

    while nextOut < nextIn:
        px, py = fifo[nextOut]
        nextOut += 1
        fp = imG[px, py]
        for irany in range(8):
            if (
                px + dx[irany] >= 0
                and px + dx[irany] < imBe.shape[0]
                and py + dy[irany] >= 0
                and py + dy[irany] < imBe.shape[1]
            ):
                fpv = imG[px + dx[irany], py + dy[irany]]
                pvout = imKi[px + dx[irany], py + dy[irany]]
                if fp == fpv and pvout == 8:
                    imKi[px + dx[irany], py + dy[irany]] = (irany + 4) % 8
                    imMap[px + dx[irany], py + dy[irany]] = 255
                    imBe[px, py] = bits[(irany + 4) % 8] | imBe[px, py]
                    fifo[nextIn] = (px + dx[irany], py + dy[irany])
                    nextIn += 1

    cv2.imshow("Ablak", imMap)
    cv2.waitKey()

    # step 3
    stack = [(0, 0)] * (imBe.shape[0] * imBe.shape[1] * 10)
    nrStack = 0
    for x in range(imBe.shape[0]):
        for y in range(imBe.shape[1]):
            fp = imG[x, y]
            pout = imKi[x, y]
            if not pout == 8:
                continue

            for irany in range(8):
                if (
                    x + dx[irany] >= 0
                    and x + dx[irany] < imBe.shape[0]
                    and y + dy[irany] >= 0
                    and y + dy[irany] < imBe.shape[1]
                ):
                    fpv = imG[x + dx[irany], y + dy[irany]]
                    pvout = imKi[x + dx[irany], y + dy[irany]]
                    if pvout == 8 and fp == fpv:
                        imKi[x + dx[irany], y + dy[irany]] = (irany + 4) % 8
                        imMap[x + dx[irany], y + dy[irany]] = 255
                        imBe[x, y] = bits[(irany + 4) % 8] | imBe[x, y]
                        stack[nrStack] = (x + dx[irany], y + dy[irany])
                        nrStack += 1

            while nrStack > 0:
                nrStack -= 1
                pvx, pvy = stack[nrStack]
                fpv = imG[pvx, pvy]
                pvout = imKi[pvx, pvy]
                for irany in range(8):
                    if (
                        pvx + dx[irany] >= 0
                        and pvx + dx[irany] < imBe.shape[0]
                        and pvy + dy[irany] >= 0
                        and pvy + dy[irany] < imBe.shape[1]
                    ):
                        fpvv = imG[pvx + dx[irany], pvy + dy[irany]]
                        pvvout = imKi[pvx + dx[irany], pvy + dy[irany]]
                        if (
                            fpv == fpvv
                            and pvvout == 8
                            and not (pvx + dx[irany] == x and pvy + dy[irany] == y)
                        ):
                            imMap[pvx + dx[irany], pvy + dy[irany]] = 255
                            imKi[pvx + dx[irany], pvy + dy[irany]] = (irany + 4) % 8
                            imBe[pvx, pvy] = bits[(irany + 4) % 8] | imBe[pvx, pvy]
                            stack[nrStack] = (pvx + dx[irany], pvy + dy[irany])
                            nrStack += 1

    cv2.imshow("Ablak", imMap)
    cv2.waitKey()

    # step 4
    medbuf = []
    label = 0
    nextIn = 0
    spotSum = [0] * 3

    for x in range(imBe.shape[0]):
        for y in range(imBe.shape[1]):
            pout = imKi[x, y]
            if not pout == 8:
                continue
            stack[nrStack] = (x, y)
            nrStack += 1

            while nrStack > 0:
                nrStack -= 1

                pvx, pvy = stack[nrStack]
                fifo[nextIn] = (pvx, pvy)
                nextIn += 1
                [b, g, r] = imColor[pvx, pvy]
                spotSum[0] += b
                spotSum[1] += g
                spotSum[2] += r
                o = int(r) * 0x10000 + int(g) * 0x100 + int(b)

                o += np.uint32(
                    round(float(r * 0.299) + float(g * 0.587) + float(b * 0.114))
                    * 0x1000000
                )
                medbuf.append(o)
                pvin = imBe[pvx, pvy]
                for irany in range(8):
                    if (
                        (bits[irany] & pvin) > 0
                        and not (
                            (pvx + dx[(irany + 4) % 8], pvy + dy[(irany + 4) % 8])
                            in fifo[:nextIn]
                        )
                        and pvx + dx[(irany + 4) % 8] > 0
                        and pvx + dx[(irany + 4) % 8] < imBe.shape[0]
                        and pvy + dy[(irany + 4) % 8] > 0
                        and pvy + dy[(irany + 4) % 8] < imBe.shape[1]
                    ):
                        stack[nrStack] = (
                            pvx + dx[(irany + 4) % 8],
                            pvy + dy[(irany + 4) % 8],
                        )
                        nrStack += 1

            label += 1
            if nextIn < 2:
                print(nextIn)

            for i in range(3):
                spotSum[i] = round(spotSum[i] / nextIn)
            medbuf = medbuf[:nextIn]
            medbuf.sort()
            medR = (medbuf[nextIn // 2] % 0x1000000) / 0x10000
            medG = (medbuf[nextIn // 2] % 0x10000) / 0x100
            medB = medbuf[nextIn // 2] % 0x100
            for i in range(nextIn):
                imSegm[fifo[i][0], fifo[i][1]] = [spotSum[0], spotSum[1], spotSum[2]]
                imSegmMed[fifo[i][0], fifo[i][1]] = [medB, medG, medR]
            nextIn = 0

    print("Regions: {0}".format(label))
    cv2.imshow("Median", imSegmMed)
    cv2.imshow("Atlag", imSegm)
    cv2.waitKey()


def main():
    print("Lab 10: Watershed")
    watershed()


if __name__ == "__main__":
    main()
