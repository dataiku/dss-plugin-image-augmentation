# Image augmentation plugin
This plugin can be used to generate more training data from existing images.

## Usage
**Input:** A folder containing images
**Output:** A folder containing more images, generated from the input ones.

Multiple images (depending on the "Augmentation factor" parameter) are generated for each image using the keras augmentation class (ImageDataGenerator). They are generated using random transformation techniques.

Moreover, images are resized to a fixed shape (input by users) using the resize function of the PIL package.
