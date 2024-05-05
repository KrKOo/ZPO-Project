# Description: Python implementation of the Codebook algorithm.

import cv2
import numpy as np
import os

CHANNELS = 3
WIDTH = 854
HEIGHT = 480


class CodeElement:
    def __init__(self):
        self.channels = CHANNELS
        self.learnHigh = np.zeros((self.channels, 1))
        self.learnLow = np.zeros((self.channels, 1))
        self.max = np.zeros((self.channels, 1))
        self.min = np.zeros((self.channels, 1))
        self.stale = 0


class Codebook:
    def __init__(self):
        self.elems = []
        self.t = 0


def updateCodebook(pixel, cb, cbBounds, numChannels):
    high = np.zeros((CHANNELS, 1))
    low = np.zeros((CHANNELS, 1))
    for c in range(numChannels):
        high[c] = pixel[c] + cbBounds[c]
        if high[c] > 255:
            high[c] = 255

        low[c] = pixel[c] - cbBounds[c]
        if low[c] < 0:
            low[c] = 0

    matchChannel = 0

    ii = 0
    for i in range(len(cb.elems)):
        for n in range(numChannels):
            if (
                cb.elems[i].learnLow[n] <= pixel[n]
                and pixel[n] <= cb.elems[i].learnHigh[n]
            ):
                matchChannel += 1

        if matchChannel == numChannels:
            cb.elems[i].lastUpdate = cb.t

            for c in range(numChannels):
                if cb.elems[i].max[c] < pixel[c]:
                    cb.elems[i].max[c] = pixel[c]
                elif cb.elems[i].min[c] > pixel[c]:
                    cb.elems[i].min[c] = pixel[c]

            break
        ii += 1

    for s in range(len(cb.elems)):
        negRun = cb.t - cb.elems[s].lastUpdate
        if cb.elems[s].stale < negRun:
            cb.elems[s].stale = 1

    if ii == len(cb.elems):
        ce = CodeElement()
        for c in range(numChannels):
            ce.learnHigh[c] = high[c]
            ce.learnLow[c] = low[c]
            ce.max[c] = pixel[c]
            ce.min[c] = pixel[c]

        ce.lastUpdate = cb.t
        ce.stale = 0

        cb.elems.append(ce)

    for c in range(numChannels):
        if cb.elems[ii].learnHigh[c] < high[c]:
            cb.elems[ii].learnHigh[c] += 1

        if cb.elems[ii].learnLow[c] > low[c]:
            cb.elems[ii].learnLow[c] -= 1

    return ii


def clearStaleEntires(cb):
    staleThresh = cb.t / 2
    keep = np.zeros(len(cb.elems))

    keepCnt = 0

    for i in range(len(cb.elems)):
        if cb.elems[i].stale > staleThresh:
            keep[i] = 0
        else:
            keep[i] = 1
            keepCnt += 1

    k = 0
    numCleared = 0

    for i in range(len(cb.elems)):
        if keep[i]:
            cb.elems[k] = cb.elems[i]
            cb.elems[k].lastUpdate = 0
            k += 1
        else:
            numCleared += 1

    cb.elems = cb.elems[:k]
    return numCleared


def backgroundDiff(pixel, cb, numChannels, minMod, maxMod):
    ii = 0
    for i in range(len(cb.elems)):
        matchChannel = 0
        for n in range(numChannels):
            if (
                cb.elems[i].min[n] - minMod[n] <= pixel[n]
                and pixel[n] <= cb.elems[i].max[n] + maxMod[n]
            ):
                matchChannel += 1
            else:
                break

        if matchChannel == numChannels:
            break
        ii += 1

    if ii >= len(cb.elems):
        return 255

    return 0


codebooks = np.array([[Codebook() for i in range(WIDTH)] for j in range(HEIGHT)])

# Load image sequence from the butterfly folder
files = [f for f in os.listdir("butterfly")]
files.sort()

images = [cv2.imread("butterfly/" + f) for f in files]

for image in images:

    newPixels = np.array(
        [[0 for i in range(WIDTH)] for j in range(HEIGHT)], dtype=np.uint8
    )
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            newPixels[x][y] = backgroundDiff(
                image[x, y], codebooks[x, y], 3, [10, 10, 10], [10, 10, 10]
            )

    cv2.imshow("frame", newPixels)

    print("Updating codebook")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel = image[x, y]
            updateCodebook(pixel, codebooks[x, y], [10, 10, 10], 3)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
