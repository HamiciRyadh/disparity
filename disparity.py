import cv2
import numpy as np


def calculate_disparity(d, image1, image2, window, error_function, arg_min_max_function):
    disparity_1 = np.empty(image1.shape, dtype=np.uint8)
    disparity_2 = np.empty(image1.shape, dtype=np.uint8)
    image1 = cv2.copyMakeBorder(image1, window, window, d+window, d+window, cv2.BORDER_CONSTANT)
    image2 = cv2.copyMakeBorder(image2, window, window, d+window, d+window, cv2.BORDER_CONSTANT)

    errors_1 = np.empty(d+1, dtype=np.float32)
    errors_2 = np.empty(d+1, dtype=np.float32)
    for r in range(window, image1.shape[0] - window):
        for c in range(d+window, image1.shape[1] - window - d):
            for i in range(-d, 1):
                errors_1.itemset(i+d, error_function(image1[r-window: r+window+1, c-window: c+window+1],
                                                     image2[r-window: r+window+1, i+c-window: i+c+window+1]))
            disparity_1.itemset((r - window, c - window - d), d-arg_min_max_function(errors_1))
            for i in range(0, d+1):
                errors_2.itemset(i, error_function(image2[r-window: r+window+1, c-window: c+window+1],
                                                   image1[r-window: r+window+1, i+c-window: i+c+window+1]))
            disparity_2.itemset((r - window, c - window - d), arg_min_max_function(errors_2))

    return disparity_1, disparity_2


def ssd(image1, image2):
    return np.sum(((image1 - image2)**2))


def sad(image1, image2):
    return np.sum(np.abs(image1 - image2))


def cc(image1, image2):
    return np.sum(image1 * image2)


def nc(image1, image2):
    temp = np.sum(image1 * image2)
    return 0 if temp == 0 else temp/(temp**0.5)

