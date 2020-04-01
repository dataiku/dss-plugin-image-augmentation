# dss-plugin-image-augmentation
This plugin is used for image augmentation in order to get more training data in a machine learning model.
This plugin takes as input a folder of images and outputs in a folder more images generated from the input images.

Multiple images (depending on the "Augmentation factor" parameter) are generated for each image using the keras augmentation class (ImageDataGenerator). They are generated using random transformation techniques.

Morevoer, images are resized to a fixed shape (input by users) using the resize function of the PIL package.




