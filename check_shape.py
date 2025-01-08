import os

import SimpleITK as sitk

if __name__ == '__main__':
    path = './data/brain/train'
    for p in os.listdir(path):
        cbct_path = os.path.join(path, p, 'cbct.nii.gz')
        cbct = sitk.ReadImage(cbct_path)
        ct_path = os.path.join(path, p, 'ct.nii.gz')
        ct = sitk.ReadImage(ct_path)
        print(cbct.GetSize())
        print(ct.GetSize())

