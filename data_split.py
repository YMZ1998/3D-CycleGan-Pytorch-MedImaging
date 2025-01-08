import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def copy_files(file_list, origin_path, dest_path, item):
    for file in tqdm(file_list, desc=f"Copying {item} files"):
        src_path = os.path.join(origin_path, file)
        dst_path = os.path.join(dest_path, file)

        try:
            shutil.copytree(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {file}: {e}")


if __name__ == '__main__':
    anatomy = 'pelvis'

    data_path = os.path.join(r'D:\Data\SynthRAD\Task2', anatomy)
    result_path = os.path.join(r'./data', anatomy)

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    train_path = os.path.join(result_path, 'train')
    test_path = os.path.join(result_path, 'test')

    paths = os.listdir(data_path)[:100]
    print(f"Total files: {len(paths)}")

    train, test = train_test_split(paths, test_size=0.2, random_state=42)
    print(f"Train files: {len(train)}")
    print(f"Test files: {len(test)}")

    copy_files(train, data_path, train_path, 'train')
    copy_files(test, data_path, test_path, 'test')
