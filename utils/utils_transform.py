import random
import numpy as np

import torchvision.transforms.functional as F


class VideoTransform:
    def __init__(self, output_size, flip_prob=0.5, crop_prob=0.5, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)):
        """
        参数：
            output_size (int or tuple): 输出图像的尺寸，int 类型表示长宽相同，tuple 类型表示 (height, width)。
            flip_prob (float): 随机翻转的概率，默认 0.5。
            crop_prob (float): 随机裁剪的概率，默认 0.5。
            scale (tuple): 随机裁剪区域相对于原图面积的比例范围。
            ratio (tuple): 随机裁剪区域的宽高比范围。
        """
        self.output_size = output_size
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.scale = scale
        self.ratio = ratio

    def __call__(self, images):
        # 获取当前视频帧的尺寸，假设所有帧大小相同
        width, height = images[0].size

        # 生成一次随机参数
        vertical_flip = random.random() < self.flip_prob
        horizontal_flip = random.random() < self.flip_prob
        do_random_crop = random.random() < self.crop_prob

        # 如果需要随机裁剪，生成裁剪参数
        if do_random_crop:
            crop_width, crop_height = self._get_random_crop_size(width, height, self.scale, self.ratio)
            x1 = random.randint(0, width - crop_width)
            y1 = random.randint(0, height - crop_height)
        else:
            # 不需要随机裁剪，使用全图
            x1, y1, crop_width, crop_height = 0, 0, width, height

        transformed_images = []
        for img in images:
            # 裁剪
            img = F.crop(img, y1, x1, crop_height, crop_width)

            # 随机垂直翻转
            if vertical_flip:
                img = F.vflip(img)

            # 随机水平翻转
            if horizontal_flip:
                img = F.hflip(img)

            # 调整图像大小到指定的输出尺寸
            img = F.resize(img, [self.output_size, self.output_size])

            # 转换为张量
            img = F.to_tensor(img)

            transformed_images.append(img)
        return transformed_images

    def _get_random_crop_size(self, width, height, scale, ratio):
        """根据图像尺寸、比例和缩放范围生成随机裁剪尺寸"""
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(np.sqrt(target_area / aspect_ratio)))

            if crop_width <= width and crop_height <= height:
                return crop_width, crop_height

        # 如果循环未找到合适的尺寸，使用最小边长进行中心裁剪
        min_side = min(width, height)
        return min_side, min_side
