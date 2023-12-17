import os
import random
import shutil

path = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/sizeSplitTestData/largeMasks/"
dest = "C:/Users/kirby/Downloads/mammograms/mammograms/dataset/CBIS-DDSM/sizeSplitTestData/testLarge/"
files = os.listdir(path)
files_count = int(len(files) / 5)

print(files_count)

for filename in random.sample(files, files_count):
    if "Training" in filename:
        new_filename = filename.replace("Training", "Test")
        shutil.move(path+filename, dest+new_filename)
        # print(filename)
    else:
        shutil.move(path+filename, dest+filename)