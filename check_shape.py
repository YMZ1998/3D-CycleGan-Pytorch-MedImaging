import os

import SimpleITK as sitk

if __name__ == '__main__':
    path = './data/brain/train'
    for p in os.listdir(path):
        cbct_path = os.path.join(path, p, 'cbct.nii.gz')
        cbct = sitk.ReadImage(cbct_path)
        print(cbct.GetSize())
