from keras.preprocessing.image import ImageDataGenerator
import logging
import dataiku
from dataiku.customrecipe import *

def get_input_ouput():
    input_names = get_input_names_for_role('input_image_folder')[0]
    output_names = get_output_names_for_role('output_image_folder')[0]
    input_folder = dataiku.Folder(input_names)
    output_folder = dataiku.Folder(output_names)
    return input_folder, output_folder
    
    
def get_images_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    scaling_factor = int(_p('scaling_factor'))
    height = int(_p('height'))
    width = int(_p('width'))
    
    if scaling_factor > 50:
        raise ValueError("Maximum scaling factor is 50")
    if height > 1024:
        height = 1024
        logger.info("[+] Cast images height to {} pixels".format(height))
    if width > 1024:
        width = 1024
        logger.info("[+] Cast images width to {} pixels".format(width))
   
    return scaling_factor, height, width

def get_generator_object(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    custom_gen = _p('use_custom_transformation')
    
    if custom_gen:
        zoom_range = _p('zoom_range')
        rotation_range = int(_p('rotation_range'))
        shear_range = _p('shear_range')
        horizontal_flip = _p('horizontal_flip')
    else:
        zoom_range=0.2
        rotation_range=20
        shear_range=0.2
        horizontal_flip=True
    
    datagen = ImageDataGenerator(
                    zoom_range=zoom_range,
                    shear_range=shear_range,
                    rotation_range=rotation_range,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=horizontal_flip)
    return datagen, custom_gen





