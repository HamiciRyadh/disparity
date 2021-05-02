import cv2
import numpy as np
import disparity as dp
import os


if __name__ == "__main__":
    base_input_path = "resources/input/"
    base_output_path = "resources/output/"
    directories = [name for name in os.listdir("resources/input")]
    for directory in directories:
        print("Working on " + directory)
        img1 = np.int32(cv2.imread(base_input_path + "/" + directory + "/view1.png", cv2.IMREAD_GRAYSCALE))
        img2 = np.int32(cv2.imread(base_input_path + "/" + directory + "/view5.png", cv2.IMREAD_GRAYSCALE))

        # SSD
        print("SSD")
        im12, im21 = dp.calculate_disparity(100, img1, img2, 1, dp.sad, np.argmin)
        cv2.imwrite(base_output_path + "/" + directory + "/SSD/disp1.png", im12)
        cv2.imwrite(base_output_path + "/" + directory + "/SSD/disp5.png", im21)

        # SAD
        print("SAD")
        im12, im21 = dp.calculate_disparity(100, img1, img2, 1, dp.sad, np.argmin)
        cv2.imwrite(base_output_path + "/" + directory + "/SAD/disp1.png", im12)
        cv2.imwrite(base_output_path + "/" + directory + "/SAD/disp5.png", im21)

        # CC
        print("CC")
        im12, im21 = dp.calculate_disparity(100, img1, img2, 1, dp.cc, np.argmax)
        cv2.imwrite(base_output_path + "/" + directory + "/CC/disp1.png", im12)
        cv2.imwrite(base_output_path + "/" + directory + "/CC/disp5.png", im21)

        # NC
        print("NC")
        im12, im21 = dp.calculate_disparity(100, img1, img2, 1, dp.nc, np.argmax)
        cv2.imwrite(base_output_path + "/" + directory + "/NC/disp1.png", im12)
        cv2.imwrite(base_output_path + "/" + directory + "/NC/disp5.png", im21)
