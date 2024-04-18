import sys
from PIL import Image
import numpy as np
import os


def convert_pixels(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)
            # 使用Pillow打开图像
            img = Image.open(file_path)
            img_array = np.array(img)

            # 将值为255的像素改为1
            img_array[img_array == 255] = 1

            # 将修改后的numpy数组转换回Pillow图像，并保存
            img_modified = Image.fromarray(img_array)
            img_modified.save(file_path)
            print(f"Processed {filename}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 0255_2_01.py <folder_path>")
    else:
        folder_path = sys.argv[1]
        convert_pixels(folder_path)
