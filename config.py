import os

DATASET_PATH = os.path.join("dataset")
# DATASET_PATH = os.path.join("dataset", "CBIS", "train")
# define the path to the images and masks dataset


EPOCHS = 120
Learning_rate = 0.0001
FOLDS = 5
BATCH_SIZE = 4


# BASE_OUTPUT = "output"
# ACCURACY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "acc_plot.png"])
# LOSSES_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "losses_plot.png"])
# IOU_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "iou_plot.png"])
# DICE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "dice_plot.png"])
# PIXEL_ACCURACY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "pixel_accuracyplot.png"])
# RECALL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "recall_plot.png"])
# SPECIFICTY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "specificty.png"])