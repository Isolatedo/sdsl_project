import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
import random
from tqdm import tqdm
from openslide_func import openSlide as di_openSlide

# 解决了数据泄露
# from parse_embolus import read_region_kfb 
from pydaily import filesystem

def get_level_dim_dict(slide_path):
    level_dim_dict = {}
    slide = di_openSlide(slide_path) 
    
    if slide is None:
        print(f"错误：无法读取切片 {slide_path}，跳过处理。")
        return None
    dims = slide.level_dimensions
    downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict

def get_contours(slide_path, level, label_dict, ext='.kfb'):
    """解析 JSON 标注文件获取轮廓"""
    # 兼容处理 JSON 读取错误
    index_file = slide_path.replace(ext, '.json')
    roi_contours = []
    roi_labels = []

    info_dict = None
    # 尝试多种编码读取
    encodings = ['utf-8', 'gbk', 'GBK']
    for enc in encodings:
        try:
            with open(index_file, 'r', encoding=enc) as f:
                info_dict = json.load(f)
            break
        except:
            continue
    
    if info_dict is None:
        # 如果找不到 json 或者读取全失败
        return [], []

    roilist = info_dict.get('annotation', [])
    # print('{} roi regions are labeled in '.format(len(roilist)))
    
    for i, roi_dict in enumerate(roilist):
        try:
            remark_name = roi_dict["label"]
            # 如果标签不在我们需要处理的字典里，跳过
            if remark_name not in label_dict:
                continue
            remark = label_dict[remark_name]
        except:
            continue

        path = roi_dict["position"]
        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))

        corrds_np = np.array(corrds, dtype=np.int32)
        roi_contours.append(corrds_np)
        roi_labels.append(remark)
        
    return roi_contours, roi_labels

def deduplicate_paths(file_paths):
    """去重逻辑"""
    file_dict = {}
    for path in file_paths:
        file_name = path.split('/')[-1]
        if file_name in file_dict:
            file_dict[file_name].append(path)
        else:
            file_dict[file_name] = [path]

    unique_paths = []
    for file_name, paths in file_dict.items():
        unique_paths.append(paths[0])
    return unique_paths

def split_patches(image, mask, save_dir, label_count_dict, save_ind, subset_type='train'):
    """ 
    切图并保存
    subset_type: 'train', 'val', 'test' 
    """
    
    # 1. 基础参数设置
    h, w = image.shape[0], image.shape[1]
    patch_size = 384
    h_overlap = 192 
    w_overlap = 192 
    
    label_color = {
        '脉管癌栓': (0, 255, 0),  
        '肿瘤芽': (255, 0, 0)    
    }
    
    label_num = mask.shape[-1] 

    # === 采样率设置 ===
    if subset_type == 'train':
        background_keep_rate = 0.0001  # <--- 修改这里，0 表示彻底不留背景
    else:
        # 验证集和测试集建议稍微保留一点，否则无法评估模型对正常组织的抗干扰能力
        background_keep_rate = 0.001
    
    target_pixel_threshold = 255 * 5 

    # 3. 打开对应的列表文件 (追加模式)
    list_file_path = os.path.join(save_dir, 'ImageSets', 'Segmentation', f'{subset_type}.txt')
    f_list = open(list_file_path, 'a')

    f_index = 0

    # 4. 开始切图循环
    # desc 显示当前切片归属
    for x in tqdm(range(0, h - patch_size + 1, patch_size - h_overlap), desc=f"Splitting {save_ind} ({subset_type})", leave=False):
        for y in range(0, w - patch_size + 1, patch_size - w_overlap):
            
            patch = image[x:x+patch_size, y:y+patch_size, :]
            patch_label = mask[x:x+patch_size, y:y+patch_size, :]
            
            # --- 过滤逻辑 ---
            if np.mean(patch) > 235: 
                continue
            
            target_channels = patch_label[:, :, 1:] 
            labeled_pixel_sum = np.sum(target_channels)
            
            is_positive = False
            if labeled_pixel_sum >= target_pixel_threshold:
                is_positive = True
            else:
                # 如果 background_keep_rate 设为 0，这里会直接 continue，丢弃所有背景
                if background_keep_rate == 0:
                    continue
                elif random.random() > background_keep_rate:
                    continue

            # 生成 label (单通道 0,1,2)
            patch_label_class = np.argmax(patch_label, axis=-1)

            # 5. 保存数据
            save_name_base = "%s_%d" % (save_ind, f_index)
            
            # 保存原图
            cv2.imwrite(os.path.join(save_dir, 'JPEGImages', save_name_base + ".jpg"), patch)
            # 保存 Label
            cv2.imwrite(os.path.join(save_dir, 'SegmentationClass', save_name_base + ".png"), patch_label_class)

            # 6. 可视化 (仅用于检查，不参与训练)
            temp_mask = 0.4 * (1 - patch_label[:, :, 0] / 255)
            temp_mask[temp_mask == 0] = 1
            temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)
            image_and_label = temp_mask * patch 
            
            for i in range(label_num):
                if i == 0: continue
                cur_label = patch_label[:, :, i]
                cur_label = cur_label[:, :, np.newaxis] // 255
                cur_label = np.repeat(cur_label, 3, -1)
                if (i - 1) < len(label_color):
                    cur_color = list(label_color.values())[i - 1]
                    label_region_color = cur_label * cur_color
                    image_and_label = image_and_label + 0.6 * label_region_color
            
            cv2.imwrite(os.path.join(save_dir, 'VisPatch', save_name_base + ".jpg"), image_and_label)

            # 7. 写入 txt
            # 这里的逻辑修改了：不再取模，而是直接写入当前 subset 对应的文件
            f_list.write(save_name_base + "\n")
            
            f_index += 1

    f_list.close()
    # print(f"Finished splitting {save_ind}. Total patches: {f_index}")

def vis_anno(slide_path, roi_contours, roi_labels, level, save_dir, index, label_dict, subset_type='train'):
    """主处理函数：读取区域 -> 绘制掩码 -> 调用切图"""
    label_num = len(label_dict) + 1 # +1 for background

    # 1. 获取尺寸
    level_dim_dict = get_level_dim_dict(slide_path)
    if not level_dim_dict:
        return
    scale = level_dim_dict[level][1]
    dim = level_dim_dict[level][0] 

    # 2. 打开切片
    slide = di_openSlide(slide_path)
    if slide is None:
        return

    try:
        # 3. 读取区域
        region_data = slide.read_region(location=(0, 0), level=level, size=dim)
        
        # 4. 格式转换
        if isinstance(region_data, np.ndarray):
            thumbnail = region_data
            thumbnail = thumbnail[:, :, ::-1] # RGB -> BGR
        else:
            thumbnail = np.array(region_data)
            if thumbnail.shape[2] == 4:
                thumbnail = thumbnail[:, :, :3]
            thumbnail = thumbnail[:, :, ::-1] # RGB -> BGR

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"读取切片失败 {os.path.basename(slide_path)}: {e}")
        return

    # 生成大图 Mask
    h, w = thumbnail.shape[0], thumbnail.shape[1]
    mask_fortrain = np.zeros((h, w, label_num))
    
    for i, roi_contour in enumerate(roi_contours):
        # 坐标缩放
        x_coord = np.array([int(contour[0]/ 2**level) for contour in roi_contour])
        y_coord = np.array([int(contour[1]/2**level) for contour in roi_contour])
        coords_np = np.concatenate((x_coord[:, np.newaxis], y_coord[:, np.newaxis]), axis=1)
        
        cur_mask_fortrain = np.zeros((h, w), dtype=np.uint8)
        label = roi_labels[i]
        
        cv2.drawContours(cur_mask_fortrain, [coords_np], -1, 1, cv2.FILLED) 
        mask_fortrain[:, :, label] += cur_mask_fortrain * 255

    mask_fortrain = np.clip(mask_fortrain, 0, 255)
    
    # 调用切图，传入 subset_type
    split_patches(thumbnail, mask_fortrain, 
                  save_dir=save_dir, 
                  save_ind=index, 
                  label_count_dict=label_dict, 
                  subset_type=subset_type)

if __name__ == '__main__':
    # === 配置路径 ===
    save_dir = '/sdsl/code/WSI_DATA3'
    img_folder = '/sdsl/data'
    
    # 类别定义
    label_dict = {
        "脉管癌栓": 1,
        "肿瘤芽": 2,
    }

    # 创建目录
    if not os.path.exists(os.path.join(save_dir, 'ImageSets')):
        os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisWSI'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    
    # === 关键步骤0: 清理旧的 txt 文件 ===
    # 避免多次运行导致内容重复追加
    for split_name in ['train', 'val', 'test']:
        txt_path = os.path.join(save_dir, 'ImageSets', 'Segmentation', f'{split_name}.txt')
        if os.path.exists(txt_path):
            print(f"Removing old {split_name}.txt")
            os.remove(txt_path)

    level = 1
    
    # === 关键步骤1: 收集并配对所有切片 ===
    valid_slides = [] # 存储有效切片信息的列表
    
    ext_list = ['.kfb', '.svs']
    all_img_paths = []
    
    print("正在扫描文件...")
    for ext in ext_list:
        files = filesystem.find_ext_files(img_folder, ext)
        all_img_paths.extend(deduplicate_paths(files))
    
    json_list = filesystem.find_ext_files(img_folder, '.json')
    # 建立 json 查找表
    json_map = {os.path.splitext(os.path.basename(p))[0]: p for p in json_list}
    
    # 配对 Slide 和 JSON
    for slide_path in all_img_paths:
        ext = os.path.splitext(slide_path)[1]
        json_name = slide_path.replace(ext, '.json')
        base_name = os.path.splitext(os.path.basename(json_name))[0]
        
        if base_name in json_map:
            valid_slides.append({
                'slide_path': slide_path,
                'json_path': json_map[base_name],
                'base_name': base_name,
                'ext': ext
            })
    
    print(f"找到有效配对切片数: {len(valid_slides)}")

    # === 关键步骤2: WSI 层级随机打乱与划分 ===
    random.seed(42) # 固定种子，保证复现
    random.shuffle(valid_slides)
    
    total_slides = len(valid_slides)
    train_num = int(total_slides * 0.8)
    val_num = int(total_slides * 0.1)
    # 剩下的就是 test
    
    print(f"划分方案 -> Train: {train_num}, Val: {val_num}, Test: {total_slides - train_num - val_num}")

    # === 关键步骤3: 遍历处理 ===
    global_index = 0 # 用于文件命名的唯一索引
    
    for i, item in enumerate(tqdm(valid_slides, desc="Processing WSIs")):
        # 确定当前切片属于哪个集
        if i < train_num:
            subset_type = 'train'
        elif i < train_num + val_num:
            subset_type = 'val'
        else:
            subset_type = 'test'
            
        slide_path = item['slide_path']
        json_path = item['json_path']
        ext = item['ext']
        
        # 检查是否能获取 dimensions
        level_dim_dict = get_level_dim_dict(slide_path)
        if level_dim_dict is None or not level_dim_dict:
            continue
            
        # 获取轮廓
        roi_contours, roi_labels = get_contours(slide_path, level, label_dict, ext)
        
        if len(roi_contours) == 0:
            print(f"警告：切片 {item['base_name']} json存在但无法解析出有效轮廓，跳过。")
            continue

        # 处理并保存 (传入 subset_type)
        vis_anno(slide_path, roi_contours, roi_labels, 
                 level=level, 
                 save_dir=save_dir, 
                 index=global_index, 
                 label_dict=label_dict,
                 subset_type=subset_type)
        
        global_index += 1
        
    print("所有处理完成！")