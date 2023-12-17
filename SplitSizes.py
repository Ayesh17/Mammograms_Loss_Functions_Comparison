import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

path = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/allMasks/"
l_path = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/largeMasks/"
m_path = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/mediumMasks/"
s_path = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/smallMasks/"

ratio = []

for file in os.listdir(path):
    if file.endswith(".png"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        n_white_pix = np.sum(img == 255)
        if n_white_pix > 0 and n_white_pix < 150:
            print(file, "White pixels: ", n_white_pix)
            shutil.copy(path+file, s_path+file)
            npy_file = file.replace('.png', '.npy')
            shutil.copy(path+npy_file, s_path+npy_file)
        elif n_white_pix > 150 and n_white_pix < 500:
            print(file, "White pixels: ", n_white_pix)
            shutil.copy(path+file, m_path+file)
            npy_file = file.replace('.png', '.npy')
            shutil.copy(path+npy_file, m_path+npy_file)
        elif n_white_pix > 500:
            print(file, "White pixels: ", n_white_pix)
            shutil.copy(path+file, l_path+file)
            npy_file = file.replace('.png', '.npy')
            shutil.copy(path+npy_file, l_path+npy_file)
        ratio.append(n_white_pix)

arr = np.array(ratio)
plt.hist(arr, bins='auto')
plt.title("Histogram of mask sizes")
plt.xlabel("Number of white mask pixels per image")
plt.ylabel("Number of occurances")
plt.show()