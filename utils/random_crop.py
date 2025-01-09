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

        # Convert SimpleITK images to NumPy arrays
        image_np = sitk.GetArrayFromImage(image)
        label_np = sitk.GetArrayFromImage(label)

        # Keep trying until a valid crop is found
        while True:
            # Generate random start indices for the crop
            start_indices = [
                np.random.randint(0, (old - new)) if old > new else 0
                for old, new in zip(size_old, size_new)
            ]
            for i in range(3):
                if not start_indices[i] + size_new[i] < size_old[i]:
                    start_indices[i] = size_old[i] - size_new[i]

            # Calculate the end indices
            end_indices = [start + new for start, new in zip(start_indices, size_new)]

            image_crop_np = image_np[
                            start_indices[2]:end_indices[2],  # z (depth)
                            start_indices[1]:end_indices[1],  # y (height)
                            start_indices[0]:end_indices[0]  # x (width)
                            ]
            label_crop_np = label_np[
                            start_indices[2]:end_indices[2],  # z (depth)
                            start_indices[1]:end_indices[1],  # y (height)
                            start_indices[0]:end_indices[0]  # x (width)
                            ]

            # Check if the label crop meets the minimum pixel requirement
            if label_crop_np.sum() >= self.min_pixel:
                # Valid crop found
                break
            elif self._should_drop_empty_crop():
                # Allow empty crop with probability drop_ratio
                break

        # Convert the cropped NumPy arrays back to SimpleITK images
        image_crop = sitk.GetImageFromArray(image_crop_np)
        label_crop = sitk.GetImageFromArray(label_crop_np)

        # Copy metadata from the original images
        # image_crop.CopyInformation(image)
        # label_crop.CopyInformation(label)

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
    random_crop = RandomCrop(output_size=(128, 128, 64), drop_ratio=0.1, min_pixel=1048)

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
            print(f"Cropped label size: {cropped_sample['label'].GetSize()}")
            # sitk.WriteImage(cropped_sample['image'], "cropped_image.nii.gz")
            # sitk.WriteImage(cropped_sample['label'], "cropped_label.nii.gz")
        except RuntimeError as e:
            print(f"Error during random crop: {e}")
