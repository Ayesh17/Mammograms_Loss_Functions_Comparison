import os
import config
from shutil import copyfile
from sklearn.model_selection import train_test_split


root_dir= os.path.join(config.DATASET_PATH)

 # Define the directory names for images and masks images
images_dir = os.path.join(root_dir, 'images')
masks_dir = os.path.join(root_dir, 'masks')

# Get the list of all images and masks image filenames
images = os.listdir(images_dir)
masks = os.listdir(masks_dir)

# Split the dataset into train and test sets
train_images, test_images = train_test_split(images, test_size=0.15, random_state=42)
train_masks, test_masks = train_test_split(masks, test_size=0.15, random_state=42)

# Create train and test directories
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Create the target directory if it doesn't exist
train_images_dir = os.path.join(train_dir, 'images')
if not os.path.exists(train_images_dir):
    os.makedirs(train_images_dir)

train_masks_dir = os.path.join(train_dir, 'masks')
if not os.path.exists(train_masks_dir):
    os.makedirs(train_masks_dir)

test_images_dir = os.path.join(test_dir, 'images')
if not os.path.exists(test_images_dir):
    os.makedirs(test_images_dir)

test_masks_dir = os.path.join(test_dir, 'masks')
if not os.path.exists(test_masks_dir):
    os.makedirs(test_masks_dir)


# Copy train images to their respective folders
for image in train_images:
    copyfile(os.path.join(images_dir, image), os.path.join(train_images_dir, image))

for image in train_masks:
    copyfile(os.path.join(masks_dir, image), os.path.join(train_masks_dir, image))

# Copy test images to their respective folders
for image in test_images:
    copyfile(os.path.join(images_dir, image), os.path.join(test_images_dir, image))

for image in test_masks:
    copyfile(os.path.join(masks_dir, image), os.path.join(test_masks_dir, image))