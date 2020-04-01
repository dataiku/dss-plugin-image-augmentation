from keras.preprocessing.image import ImageDataGenerator

def get_generator_object(gen_params):

    datagen = ImageDataGenerator(
                    zoom_range=gen_params['zoom_range'],
                    shear_range=gen_params['shear_range'],
                    rotation_range=gen_params['rotation_range'],
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=gen_params['horizontal_flip'])

    return datagen, gen_params['custom_gen']
