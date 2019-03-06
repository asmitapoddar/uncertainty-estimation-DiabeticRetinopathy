from PIL import Image
import os
import numpy as np

def preprocess():
    with open(os.path.join(os.getcwd(), 'Result/data', 'image.png'), 'r+b') as f:
        with Image.open(f) as image:
            # use one of these filter options to resize the image
            cover = image.resize((256, 256), Image.NEAREST)
            #cover = resizeimage.resize_cover(image, [256, 256])
            width, height = cover.size  # Get dimensions

            left = (width - 224) / 2
            top = (height - 224) / 2
            right = (width + 224) / 2
            bottom = (height + 224) / 2

            cover.crop((left, top, right, bottom))

            data = np.asarray(cover, dtype="int32")

            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
            mean = [0.6000, 0.3946, 0.6041],
            std=[0.2124, 0.2335, 0.2360]
            standardized_images_out = (data - mean) / std

            #standardized_images = Image.fromarray(np.asarray(np.clip(standardized_images_out, 0, 255), dtype="uint8"), "L")
            standardized_images = Image.fromarray(standardized_images_out.astype('uint8'), 'RGB')

            cover.save('preprocessing_results/test-image-cover.jpeg', cover.format)
            standardized_images.save('preprocessing_results/standardized_images.jpeg', standardized_images.format)
