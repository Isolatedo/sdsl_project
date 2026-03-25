# 这个脚本会遍历文件夹下的所有 PNG 标签图，并统计：
# 纯背景图片数量：图像中没有任何目标像素（全为0）。
# 包含脉管癌栓的图片数量：图像中存在像素值为 1 的点。
# 包含肿瘤芽的图片数量：图像中存在像素值为 2 的点。
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def count_label_distribution(save_dir):
    # 定位 SegmentationClass 文件夹
    seg_dir = os.path.join(save_dir, 'SegmentationClass')
    
    if not os.path.exists(seg_dir):
        print(f"Error: 文件夹不存在 {seg_dir}")
        return

    # 获取所有 png 文件
    # 假设你的标签文件是 .png 格式
    file_list = glob.glob(os.path.join(seg_dir, "*.png"))
    total_files = len(file_list)
    print(f"Found {total_files} label files in {seg_dir}")

    # 初始化计数器
    count_bg_only = 0      # 纯背景 (全0)
    count_has_cls1 = 0     # 包含 脉管癌栓 (含1)
    count_has_cls2 = 0     # 包含 肿瘤芽 (含2)
    
    # 错误文件计数
    count_error = 0

    print("Start counting...")
    for file_path in tqdm(file_list):
        # 以灰度模式读取 (确保读入的是 0, 1, 2 这样的单通道索引值)
        # cv2.IMREAD_UNCHANGED 也可以，但 IMREAD_GRAYSCALE (0) 更稳妥
        label = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if label is None:
            count_error += 1
            continue

        # 获取该图中所有的唯一像素值
        unique_values = np.unique(label)

        # 1. 判断是否是纯背景
        # 如果唯一值只有 [0]，或者最大值是 0
        if np.max(unique_values) == 0:
            count_bg_only += 1
        
        # 2. 判断是否包含 脉管癌栓 (像素值 1)
        if 1 in unique_values:
            count_has_cls1 += 1
            
        # 3. 判断是否包含 肿瘤芽 (像素值 2)
        if 2 in unique_values:
            count_has_cls2 += 1

    # 打印结果
    print("\n" + "="*30)
    print("统计结果 (Statistics Result):")
    print("="*30)
    print(f"总图片数 (Total Images): {total_files}")
    print(f"纯背景图片数 (Pure Background, All 0): {count_bg_only}  ({count_bg_only/total_files*100:.2f}%)")
    print(f"含脉管癌栓图片数 (Has Class 1, Embolus): {count_has_cls1}  ({count_has_cls1/total_files*100:.2f}%)")
    print(f"含肿瘤芽图片数 (Has Class 2, Tumor Buds): {count_has_cls2}  ({count_has_cls2/total_files*100:.2f}%)")
    
    # 注意：一张图可能同时包含 1 和 2，所以上面三个数加起来可能不等于总数
    # 如果你想知道既有1又有2的图片，可以加一个逻辑：
    # if 1 in unique_values and 2 in unique_values: count_both += 1
    
    if count_error > 0:
        print(f"\n[Warning] 有 {count_error} 张图片读取失败。")

if __name__ == "__main__":
    save_dir = '/sdsl/code/WSI_DATA3'
    
    count_label_distribution(save_dir)