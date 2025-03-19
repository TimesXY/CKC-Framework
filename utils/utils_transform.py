import random
import numpy as np

import torchvision.transforms.functional as F


class VideoTransform:
    def __init__(self, output_size, flip_prob=0.5, crop_prob=0.5, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)):
        """
        Parameters:
            output_size (int or tuple): Output image size. If int, both height and width are the same;
                                        if tuple, it represents (height, width).
            flip_prob (float): Probability of random flipping (default: 0.5).
            crop_prob (float): Probability of random cropping (default: 0.5).
            scale (tuple): Range of the cropped area relative to the original image area.
            ratio (tuple): Aspect ratio range of the cropped area.
        """
        self.output_size = output_size
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.scale = scale
        self.ratio = ratio

    def __call__(self, images):
        # Get the size of the current video frame, assuming all frames are the same size
        width, height = images[0].size

        # Generate random transformation parameters
        vertical_flip = random.random() < self.flip_prob
        horizontal_flip = random.random() < self.flip_prob
        do_random_crop = random.random() < self.crop_prob

        # Generate crop parameters if random cropping is applied
        if do_random_crop:
            crop_width, crop_height = self._get_random_crop_size(width, height, self.scale, self.ratio)
            x1 = random.randint(0, width - crop_width)
            y1 = random.randint(0, height - crop_height)
        else:
            # No cropping, use the full image
            x1, y1, crop_width, crop_height = 0, 0, width, height

        transformed_images = []
        for img in images:
            # Crop the image
            img = F.crop(img, y1, x1, crop_height, crop_width)

            # Apply random vertical flip
            if vertical_flip:
                img = F.vflip(img)

            # Apply random horizontal flip
            if horizontal_flip:
                img = F.hflip(img)

            # Resize the image to the specified output size
            img = F.resize(img, [self.output_size, self.output_size])

            # Convert to tensor
            img = F.to_tensor(img)

            transformed_images.append(img)
        return transformed_images

    def _get_random_crop_size(self, width, height, scale, ratio):
        """Generate a random crop size based on image dimensions, aspect ratio, and scale range"""
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(np.sqrt(target_area / aspect_ratio)))

            if crop_width <= width and crop_height <= height:
                return crop_width, crop_height

        # If no suitable size is found, use the smallest side for center cropping
        min_side = min(width, height)
        return min_side, min_side
