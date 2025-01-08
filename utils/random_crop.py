import os
import random
import numpy as np
import SimpleITK as sitk


class RandomCrop:
    """
    Randomly crop the image and label in a sample. This is typically used for data augmentation.
    A drop ratio is implemented to randomly allow crops with empty labels (default is 0.1).
    This transformation is only applicable in training mode.

    Args:
        output_size (tuple or int): Desired output size. If int, a cubic crop is made.
        drop_ratio (float): Probability of allowing a crop with empty labels (default is 0.1).
        min_pixel (int): Minimum number of non-zero pixels required in the label crop (default is 1).
    """

    def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'

        # Validate output_size
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        elif isinstance(output_size, tuple) and len(output_size) == 3:
            self.output_size = output_size
        else:
            raise ValueError("output_size must be an integer or a tuple of length 3.")

        # Validate drop_ratio
        if not (0 <= drop_ratio <= 1):
            raise ValueError("drop_ratio must be between 0 and 1.")
        self.drop_ratio = drop_ratio

        # Validate min_pixel
        if not (isinstance(min_pixel, int) and min_pixel >= 0):
            raise ValueError("min_pixel must be a non-negative integer.")
        self.min_pixel = min_pixel

    def __call__(self, sample):
        """
        Apply random cropping to the sample.

        Args:
            sample (dict): A dictionary containing 'image' and 'label' (both SimpleITK images).

        Returns:
            dict: A dictionary containing the cropped 'image' and 'label'.
        """
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size

        # Ensure the crop size is not larger than the image size
        if any(new > old for new, old in zip(size_new, size_old)):
            raise ValueError(f"Crop size {size_new} must not be larger than the image size {size_old}.")

        # Initialize the ROI filter
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize(size_new)

        # Keep trying until a valid crop is found
        while True:
            # Generate random start indices for the crop
            start_indices = [
                np.random.randint(0, (old - new)//2) if old > new else 0
                for old, new in zip(size_old, size_new)
            ]
            # print(start_indices)

            # Check if the ROI is within the image bounds
            if all(0 <= start <= (old - new) for start, old, new in zip(start_indices, size_old, size_new)):
                roi_filter.SetIndex(start_indices)
            else:
                raise RuntimeError("Generated ROI is outside the image bounds.")

            # Crop the label
            label_crop = roi_filter.Execute(label)

            # Check if the label crop meets the minimum pixel requirement
            stat_filter = sitk.StatisticsImageFilter()
            stat_filter.Execute(label_crop)
            if stat_filter.GetSum() >= self.min_pixel:
                # Valid crop found
                break
            elif self._should_drop_empty_crop():
                # Allow empty crop with probability drop_ratio
                break

        # Crop the image
        image_crop = roi_filter.Execute(image)

        return {'image': image_crop, 'label': label_crop}

    def _should_drop_empty_crop(self):
        """
        Determine whether to allow a crop with an empty label based on the drop ratio.

        Returns:
            bool: True if the empty crop should be allowed, False otherwise.
        """
        return random.random() < self.drop_ratio


if __name__ == '__main__':
    # 创建 RandomCrop 实例
    random_crop = RandomCrop(output_size=(128, 128, 64), drop_ratio=0.1, min_pixel=10)

    path = "../data/brain/train/"
    for p in os.listdir(path):

        sample = {
            'image': sitk.ReadImage(os.path.join(path, p, "cbct.nii.gz")),
            'label': sitk.ReadImage(os.path.join(path, p, "ct.nii.gz"))
        }

        # 应用随机裁剪
        try:
            cropped_sample = random_crop(sample)
            print(f"Cropped image size: {cropped_sample['image'].GetSize()}")
            # sitk.WriteImage(cropped_sample['image'], "cropped_image.nii.gz")
            # sitk.WriteImage(cropped_sample['label'], "cropped_label.nii.gz")
        except RuntimeError as e:
            print(f"Error during random crop: {e}")