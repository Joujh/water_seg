import sys
import os
from PIL import Image

def resize_and_convert_images(folder_path, width, height):
    if not os.path.exists(folder_path):
        print(f"文件夹路径 '{folder_path}' 不存在。")
        return

    processed_files = 0
    for file in os.listdir(folder_path):
        # 检查文件是否为目标图像格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    # 调整大小
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # 构建新的文件名，将文件扩展名更改为.png
                    new_file_name = os.path.splitext(file)[0].lower() + '.png'
                    new_file_path = os.path.join(folder_path, new_file_name)
                    # 保存为PNG格式
                    img.save(new_file_path, format='PNG')
                    processed_files += 1
                    # 如果新文件名与原文件名不同，则删除原文件
                    if new_file_name != file:
                        os.remove(file_path)
            except Exception as e:
                print(f"处理文件 {file} 时发生错误: {e}")
    print(f"在文件夹 '{folder_path}' 中处理了 {processed_files} 张图片。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python script.py <img_folder_path> <label_folder_path>")
        sys.exit(1)

    img_folder_path = sys.argv[1]
    label_folder_path = sys.argv[2]
    resize_and_convert_images(img_folder_path, 2048, 1024)
    resize_and_convert_images(label_folder_path, 2048, 1024)
