import numpy as np
from PIL import Image

def convert_npy_to_png(npy_file_path, png_file_path):
    """Converts a .npy file to a .png file.

    Args:
        npy_file_path: The path to the .npy file.
        png_file_path: The path to the .png file to save the converted image to.
    """

    # Load the .npy file
    image_array = np.load(npy_file_path)

    # Create a PIL Image object from the image array
    image = Image.fromarray(image_array)

    # Save the image to a .png file
    image.save(png_file_path)

# Convert all .npy files in a directory to .png files
for npy_file_path in os.listdir('.'):
    if npy_file_path.endswith('.npy'):
        png_file_path = npy_file_path.replace('.npy', '.png')
        convert_npy_to_png(npy_file_path, png_file_path)
