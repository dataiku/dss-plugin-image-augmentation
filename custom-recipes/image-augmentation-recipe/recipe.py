import dataiku
from dataiku.customrecipe import *
import logging
import numpy as np

from dataiku import pandasutils as pdu
import keras
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import ImageDataGenerator
import string
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Image Augmentation Plugin %(levelname)s - %(message)s')

input_names = get_input_names_for_role('input_image_folder')
output_names = get_output_names_for_role('output_image_folder')

scaling_factor = get_recipe_config()['scaling_factor']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
input_images_folder = dataiku.Folder(input_names[0])
input_images_filenames = input_images_folder.list_paths_in_partition()

output_folder = dataiku.Folder(output_names[0])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=False,
        rotation_range=30,
        fill_mode='nearest',
        brightness_range=(0.7, 1.3),
        zoom_range = [0.7,1]
        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def random_string(length):
    pool = string.letters + string.digits
    return ''.join(random.choice(pool) for i in range(length))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for sample_image in input_images_filenames:

    # Read image from DSS folder as bytes
    try:
        with input_images_folder.get_download_stream(sample_image) as stream:
            data = stream.readlines()
    except:
        raise ('Could not open file: {}'.format(sample_image))
        continue

    img_bytes = ''.join(data)
    img_array = np.array(Image.open(BytesIO(img_bytes)))
    # Add one more dimension (for Keras input flow)
    img_array = np.expand_dims(img_array, axis=0)

    flow_gen = datagen.flow(img_array, y=None, batch_size=1)

    # Press run cells to view different images
    for i, img in enumerate(flow_gen):
        output_im = Image.fromarray(np.uint8(img[0])).resize((500,500))
        im_resize = output_im.resize((500, 500))

        buf = BytesIO()
        im_resize.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        with output_folder.get_writer("{}.jpg".format(random_string(10))) as w:
            w.write(byte_im)

        if i > scaling_factor:
            break