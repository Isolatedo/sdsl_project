import os
import cv2
import numpy as np
from tqdm import tqdm

def compare_prediction_with_gt(pred_dir, gt_dir, save_dir, label_colors):
    """
    将预测可视化图与真实标签可视化图进行拼接对比
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取预测文件夹下的所有图片
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('predict.png')]
    
    print(f"找到 {len(pred_files)} 张预测结果，开始生成对比图...")

    for pred_name in tqdm(pred_files):
        # 1. 解析文件名
        # 假设预测图是 "0_0_predict.png"，对应的 GT 应该是 "0_0.png"
        file_id = pred_name.replace('_predict.png', '') 
        gt_name = f"{file_id}.png"
        
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        # 2. 读取图片
        # 读取预测图 (RGB)
        img_pred = cv2.imread(pred_path)
        
        # 读取 GT 掩码 (灰度索引图)
        mask_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 如果找不到对应的 GT 文件，跳过
        if img_pred is None:
            print(f"无法读取预测图: {pred_path}")
            continue
        if mask_gt is None:
            # 这种情况可能发生，如果预测图是对的但GT文件夹里没有对应的（虽然应该是一一对应的）
            # print(f"警告: 未找到对应的真实标签 {gt_name}，跳过。")
            continue

        # 3. 将 GT 掩码转换为彩色可视化图
        h, w = mask_gt.shape
        img_gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 给 GT 上色 (背景默认为黑色)
        for label_idx, color in label_colors.items():
            img_gt_vis[mask_gt == label_idx] = color

        # 4. 调整尺寸 (以防万一尺寸不一致)
        if img_pred.shape[:2] != img_gt_vis.shape[:2]:
            img_gt_vis = cv2.resize(img_gt_vis, (img_pred.shape[1], img_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 5. 添加文字标签 (可选，为了看清楚哪边是哪边)
        # 在图片顶部添加文字背景条，或者直接写在图片上
        cv2.putText(img_pred, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_gt_vis, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 6. 左右拼接 (Horizontal Stack)
        # 中间加一条白线隔开
        separator = np.ones((h, 10, 3), dtype=np.uint8) * 255 
        combined_img = np.hstack([img_pred, separator, img_gt_vis])

        # 7. 保存
        save_name = f"{file_id}_compare.png"
        cv2.imwrite(os.path.join(save_dir, save_name), combined_img)

    print(f"对比图已生成完毕，保存在: {save_dir}")

if __name__ == '__main__':
    # ================= 配置路径 =================
    
    # 你的预测结果文件夹
    PRED_DIR = '/sdsl/code/post_data/patch_predict'
    
    # 你的真实标签文件夹 (单通道索引图)
    GT_DIR = '/sdsl/code/WSI_DATA/SegmentationClass'
    
    # 结果保存路径
    SAVE_DIR = '/sdsl/code/post_data/gt_and_pre_compare'

    # 颜色配置 (BGR格式) - 保持和你之前的配置一致
    LABEL_COLORS = {
        0: (0, 0, 0),       # 背景: 黑色
        1: (0, 255, 0),     # 脉管癌栓: 绿色
        2: (0, 0, 255)      # 肿瘤芽: 红色 (OpenCV BGR)
    }
    # ===========================================

    compare_prediction_with_gt(PRED_DIR, GT_DIR, SAVE_DIR, LABEL_COLORS)