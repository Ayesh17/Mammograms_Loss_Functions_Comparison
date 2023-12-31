import os

DATASET_PATH = os.path.join("dataset", "train")
# DATASET_PATH = os.path.join("dataset", "CBIS", "train")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

EPOCHS = 50
Learning_rate = 0.0001

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

BASE_OUTPUT = "output"
ACCURACY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "acc_plot.png"])
LOSSES_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "losses_plot.png"])
IOU_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "iou_plot.png"])
DICE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "dice_plot.png"])
PIXEL_ACCURACY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "pixel_accuracyplot.png"])
RECALL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "recall_plot.png"])
SPECIFICTY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "specificty.png"])