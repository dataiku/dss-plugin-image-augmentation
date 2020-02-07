import dataiku
from dataiku.customrecipe import *
import logging
import numpy as np

from dataiku import pandasutils as pdu
import keras
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Image Augmentation Plugin %(levelname)s - %(message)s')

input_names = get_input_names_for_role('input_image_folder')
output_names = get_output_names_for_role('output_image_folder')

scaling_factor = get_recipe_config()['scaling_factor']
height = get_recipe_config()['height']
width = get_recipe_config()['width']

if height > 1024:
    height = 1024
    logger.info("[+] Cast images height to {} pixels".format(height))
if width > 1024:
    width = 1024
    logger.info("[+] Cast images width to {} pixels".format(width))

input_images_folder = dataiku.Folder(input_names[0])
input_images_filenames = input_images_folder.list_paths_in_partition()

output_folder = dataiku.Folder(output_names[0])

datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

for sample_image in input_images_filenames:
    try:
        with input_images_folder.get_download_stream(sample_image) as stream:
            data = stream.readlines()
    except:
        raise ('Could not open file: {}'.format(sample_image))
        logger.warning("WARNING [-] Could not open file: {}".format(sample_image))
        continue

    img_bytes = ''.join(data)
    # resize image into specified dimensions
    img_resize = Image.open(BytesIO(img_bytes)).resize((height, width))
    img_array = np.array(img_resize)

    for i in range(scaling_factor):
        img = datagen.random_transform(img_array)
        new_img = Image.fromarray(np.uint8(img))

        buf = BytesIO()
        new_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        with output_folder.get_writer(sample_image.split('.')[0] + " _" + str(i) + ".jpg") as w:
            w.write(byte_im)

    old_img = Image.fromarray(np.uint8(img_array))
    buf = BytesIO()
    old_img.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    with output_folder.get_writer(sample_image) as w:
        w.write(byte_im)
