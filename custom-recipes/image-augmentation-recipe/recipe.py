import dataiku
from dataiku.customrecipe import *
import logging
import numpy as np
from imageaugmentation import *

from dataiku import pandasutils as pdu
import keras
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Image Augmentation Plugin %(levelname)s - %(message)s')

# get input and output folders
input_folder, output_folder = get_input_ouput()
# get images parameters
scaling_factor, height, width = get_images_params(get_recipe_config())
# get ImageDataGenerator object with right parameters
datagen = get_generator_object(get_recipe_config())

input_filenames = input_folder.list_paths_in_partition()

for sample_image in input_filenames:
    try:
        with input_folder.get_download_stream(sample_image) as stream:
            data = stream.readlines()
    except Exception as e:
        logger.warning("[-] Could not download file: {}. Got the following exception: {}".format(sample_image, str(e)))
        continue
        
    img_bytes = b"".join(data)
    # resize image into specified dimensions
    try:
        img = Image.open(BytesIO(img_bytes)).convert('RGB')  
    except Exception as e:
        logger.warning("[-] Could not open image: {}. Got the following exception: {}".format(sample_image, str(e)))
        continue
    
    img_resize = img.resize((height, width))
    img_array = np.array(img_resize)

    for i in range(scaling_factor):
        img = datagen.random_transform(img_array, seed=42)
        new_img = Image.fromarray(np.uint8(img))

        buf = BytesIO()
        new_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        with output_folder.get_writer(sample_image.split('.')[0] + "_" + str(i) + ".jpg") as w:
            w.write(byte_im)
    
    buf = BytesIO()
    img_resize.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    with output_folder.get_writer(sample_image) as w:
        w.write(byte_im)

