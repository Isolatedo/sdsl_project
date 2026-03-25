import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
####import kfb.kfbslide as kfbslide
from openslide_func import openSlide as di_openSlide
from parse_embolus import read_region_kfb
import random
from tqdm import tqdm

# import deepdish as dd
# slide = openslide.OpenSlide(slide_path)
# print(type(slide))
def get_properties(slide_path):
    with openslide.OpenSlide(slide_path) as slide:
        ## level_count: 这张图片有几个级别的分辨率, 0 表示最高分辨率, 最低分辨率
        print('level_count', slide.level_count, '\n')
        ## dimensions: level 为 0 时的 (width, height), 也就是最高分辨率的情况下 slide 的宽和高（元组）
        print('dimensions', slide.dimensions, '\n')
        ## level_dimensions: 每个 level 的 (width, height)
        print('level_dimensions', slide.level_dimensions, '\n')
        ## level_downsamples: 每个 level 下采样的倍数, 相对于 level 0, 即 level_dimension[k] = dimensions / level_downsamples[k]
        print('level_downsamples', slide.level_downsamples, '\n')
        ## associated_images: 也是 metadata, 不过 dict 的值都是一张 pil 图片.
        print('associated_images', slide.associated_images, '\n')
        ## whole-slide 的 metadata, 是一个类似 dict 的对象, 其值都是 字符串
        proper = slide.properties
        print(proper['mirax.DATAFILE.FILE_0'])

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

def read_region(slide_path):
    ## 其中 location 是读取区域的左上角在 level 0 中的坐标，level 表示我们要读取的是第几个 level 的图片，
    # size 是 (width, height), 返回的是 PIL.Image.
    # 注意：不管 level 是不是 0，location 的定位都是根据 level 0 来的，而 size 是在不同 level 上选取的。
    with openslide.OpenSlide(slide_path) as slide:
        region = slide.read_region((47712, 94343), 5, (512, 512))
        region = np.array(region)
        imsave('/data2/Caijt/WSI_ROI/region.png', region)

def get_contours(slide_path,level,ext='.kfb'):
    dir_name = os.path.dirname(slide_path)
    index_file = slide_path.replace(ext,'.json')
    roi_contours = []
    roi_labels = []

    try:
        f=open(index_file,'r')
        info_dict = json.load(f)
    except:
        try:
            f=open(index_file,'r', encoding='gbk')
            info_dict = json.load(f)
        except:
            try:
                f = open(index_file, 'r', encoding='GBK')
                info_dict = json.load(f)
            except:
                try:
                    f=open(index_file,'r', encoding='utf-8')
                    info_dict = json.load(f)
                except:
                    raise Exception("encoding error")

    roilist = info_dict['annotation']
    print('{} roi regions are labeled in '.format(len(roilist)))
    for i, roi_dict in enumerate(roilist):
        path = roi_dict["position"]
        try:
            remark_name = roi_dict["label"]
        except:
            continue
        try:
            remark=label_dict[remark_name]
        except:
            continue
        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))

        corrds_np = np.array(corrds, dtype=np.int32)
        # corrds_np[:, 0] = corrds_np[:, 0] + int(roi_dict['offset_left'])
        # corrds_np[:, 1] = corrds_np[:, 1] + int(roi_dict['offset_top'])
        # print("corrds_np")
        # print(corrds_np)
        roi_contours.append(corrds_np)
        roi_labels.append(remark)
    f.close()
    # mpp = float(roilist[0]['micrometer_per_pixel_x'])
    return roi_contours, roi_labels


def best_level_downsample(slide_path):
    with openslide.OpenSlide(slide_path) as slide:
        for i in range(10, 500, 25):
            print("对于下采样%d倍，其最好的level是%d" % (i, slide.get_best_level_for_downsample(i)))



def get_thumbnail(slide_path, size = (1920, 1920)):
    ## size 是缩略图的 (width, height)，注意到，这个缩略图是保持比例的，
    ## 所以其会将 width 和 height 中最大的那个达到指定的值，另一个等比例缩放。
    with openslide.OpenSlide(slide_path) as slide:
        thumbnail = slide.get_thumbnail(size)
        thumbnail = np.array(thumbnail)
        imsave('/data2/Caijt/WSI_ROI/thumbnail.png', thumbnail)

def get_keys(d, value):
    List = [k for k,v in d.items() if v == value]
    if len(List) == 0:
        List = ["背景"]
    return List

def split_patches(image, mask, save_dir, label_count_dict, save_ind):
    """ 
    split large image into small patches 
    保留所有正样本，仅随机保留极少量负样本（背景）
    """
    
    # 1. 基础参数设置
    h, w = image.shape[0], image.shape[1]
    patch_size = 384
    h_overlap = 192 # 高度重叠
    w_overlap = 192 # 宽度重叠
    
    # 定义类别颜色 (BGR格式，用于可视化)
    # 对应 mask 的 index: 1, 2... (0是背景)
    label_color = {
        '脉管癌栓': (0, 255, 0),  # Green
        '肿瘤芽': (255, 0, 0)    # Blue (OpenCV is BGR)
    }
    
    # 自动获取 label 数量 (背景 + 目标类)
    label_num = mask.shape[-1] 

    # ================= 方案B 核心参数 =================
    # 背景采样率：决定保留多少纯背景 Patch
    # 0.01 代表保留 1% 的背景。如果背景非常多，建议设为 0.005 或更低
    background_keep_rate = 0.0008
    
    # 像素阈值：Patch 中至少有多少目标像素才被视为“正样本”
    # 255 * 5 意味着大约 5 个像素点。对于“肿瘤芽”这种微小目标，阈值要设低
    target_pixel_threshold = 255 * 5 
    # =================================================

    # 2. 确保保存目录存在
    os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)

    # 3. 打开记录文件
    f_train = open(os.path.join(save_dir, 'ImageSets','Segmentation','train.txt'),'a')
    f_val = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'a')
    f_test = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'a')

    f_index = 0

    # 4. 开始切图循环
    # 使用 tqdm 显示进度，desc 方便查看当前处理的是哪张大图
    for x in tqdm(range(0, h - patch_size + 1, patch_size - h_overlap), desc=f"Splitting WSI {save_ind}"):
        for y in range(0, w - patch_size + 1, patch_size - w_overlap):
            
            # 截取图像和掩码
            patch = image[x:x+patch_size, y:y+patch_size, :]
            patch_label = mask[x:x+patch_size, y:y+patch_size, :]
            
            # --- 过滤逻辑开始 ---

            # A. 过滤无效区域 (切片边缘的纯黑/纯白区域)
            # 如果你的背景是白色，请改用: if np.mean(patch) > 240: continue
            if np.mean(patch) > 235: 
                continue
            # B. 计算目标像素量 (忽略通道0背景，只看通道1及之后的通道)
            # patch_label 形状为 (384, 384, n_classes)，值为 0 或 255
            target_channels = patch_label[:, :, 1:] 
            labeled_pixel_sum = np.sum(target_channels)
            
            # C. 判定是否保留
            if labeled_pixel_sum >= target_pixel_threshold:
                # 情况1: 正样本 (包含目标) -> 必须保留
                pass 
            else:
                # 情况2: 负样本 (纯背景/正常组织)
                # 产生一个 0~1 的随机数，如果大于保留率，则跳过
                if random.random() > background_keep_rate:
                    continue 
                # 只有极少数幸运的背景 patch 会走到这里被保留
            
            # --- 过滤逻辑结束，能走到这里的都是要保存的 ---

            # 生成 SegmentationClass 需要的单通道 Label (0, 1, 2...)
            patch_label_class = np.argmax(patch_label, axis=-1)

            # 5. 保存数据
            # 保存原图
            cv2.imwrite(os.path.join(save_dir, 'JPEGImages', "%s_%d.jpg" % (save_ind, f_index)), patch)
            # 保存 Label (png格式，值是 0,1,2，不要看它是黑的，实际上有值)
            cv2.imwrite(os.path.join(save_dir, 'SegmentationClass', "%s_%d.png" % (save_ind, f_index)), patch_label_class)

            # 6. 可视化 (VisPatch)
            # 生成带颜色的 mask 叠加图，方便人工检查
            # 这里的逻辑是：先给背景打一层半透明掩码，再给目标上色
            temp_mask = 0.4 * (1 - patch_label[:, :, 0] / 255)
            temp_mask[temp_mask == 0] = 1
            temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)
            image_and_label = temp_mask * patch # 变暗背景
            
            for i in range(label_num):
                if i == 0: continue # 跳过背景通道
                
                # 获取当前类别的 mask
                cur_label = patch_label[:, :, i]
                cur_label = cur_label[:, :, np.newaxis] // 255
                cur_label = np.repeat(cur_label, 3, -1)
                
                # 获取颜色
                # 这里的逻辑是为了匹配 label_color 字典的顺序
                # 假设 label_color 的定义顺序和 mask 通道顺序一致 (1, 2...)
                if (i - 1) < len(label_color):
                    cur_color = list(label_color.values())[i - 1]
                    label_region_color = cur_label * cur_color
                    # 叠加颜色
                    image_and_label = image_and_label + 0.6 * label_region_color
            
            cv2.imwrite(os.path.join(save_dir, 'VisPatch', "%s_%d.jpg" % (save_ind, f_index)), image_and_label)

            # 7. 写入数据集列表 txt
            # 每 4 个取 1 个做验证/测试 (25%)，其余做训练 (75%)
            if f_index % 4 == 0:
                f_val.write("%s_%d\n" % (save_ind, f_index))
                f_test.write("%s_%d\n" % (save_ind, f_index))
            else:
                f_train.write("%s_%d\n" % (save_ind, f_index))
            
            f_index += 1

    # 关闭文件句柄
    f_train.close()
    f_test.close()
    f_val.close()
    print(f"Finished splitting {save_ind}. Total patches: {f_index}")

def vis_contour(slide_path, roi_contours, roi_labels, level_dim_dict, alpha = 1000000, level = 5):
    base_name = os.path.basename(slide_path).split('.')[0]
    dim = level_dim_dict[0][0]
    scale = level_dim_dict[level][1]
    for i, roi_contour in enumerate(roi_contours):
        roi_contour = np.array(roi_contour)
        roi_label = roi_labels[i]
        # print('countour_shape', roi_contour.shape)
        left_coord = np.maximum(min(roi_contour[:, 0]), 0)
        top_coord = np.maximum(min(roi_contour[:, 1]), 0)
        # print('start_point', left_coord, top_coord)
        right_coord = np.minimum(max(roi_contour[:, 0]), dim[0])
        bottom_coord = np.minimum(max(roi_contour[:, 1]), dim[1])
        height = bottom_coord - top_coord
        width = right_coord - left_coord
        # print('height, width', height, width)
        try:
            with openslide.OpenSlide(slide_path) as slide:
                region = slide.read_region((left_coord, top_coord), level, (int(width/scale), int(height/scale)))
                resized_x_coords = list(map(int, ((roi_contour[:, 0] - left_coord) / scale).tolist()))
                resized_y_coords = list(map(int, ((roi_contour[:, 1] - top_coord) / scale).tolist()))
                coords = np.concatenate((np.expand_dims(np.expand_dims(np.array(resized_x_coords), axis= -1), axis = -1),
                                         np.expand_dims(np.expand_dims(np.array(resized_y_coords), axis= -1), axis = -1)),
                                        axis=-1)
                # print('coords', coords)
                region = np.array(region)[:, :, :3]
                region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                # print('region_shape', region.shape)
                # region = np.int8(np.array(region))
                # cv2.drawContours(region, coords, -1, (0, 255, 0), 5)
                area = cv2.contourArea(coords)
                if area < 70000:
                    print('< 7 * 1e4, invalid scale')
                    sample_num = 0
                elif area < 1000000:
                    sample_num = 5 * np.log2(10 * area / alpha)
                    print('7 * 1e4 ~ 1e6')
                elif area < 10000000:
                    sample_num = 50 * np.log2(area / alpha)
                    print('1e6 ~ 1e7')
                else:
                    sample_num = 10 * (1+np.log2(area/(alpha*10)))
                    print(' > 1e7')
                print('area, sample_num', area, sample_num)
                if sample_num > 0:
                    bboxes = sample_patches(region, sample_num, coords, roi_label, base_name, i)
                    for bbox in bboxes:
                        cv2.rectangle(region, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (255, 0, 0), 5)

                    cv2.imwrite('/data2/Caijt/WSI_ROI/vis_contour_v4/vis_contour_{}_{}.png'.format(base_name, i), region)
            # pdb.set_trace()
        except:
            pass

def vis_anno(slide_path, roi_contours, roi_labels, level, save_dir, index):
    # 1. 获取尺寸
    level_dim_dict = get_level_dim_dict(slide_path)
    if not level_dim_dict:
        return
    scale = level_dim_dict[level][1]
    dim = level_dim_dict[level][0]  # (Width, Height)

    # 2. 打开切片
    # 确保你的 di_openSlide 能够正确返回这个新的 KfbSlide 实例
    slide = di_openSlide(slide_path)
    if slide is None:
        return

    try:
        # 3. 读取区域 (利用新版 KfbSlide 的自动拼接功能)
        # 无论 svs 还是 kfb，都直接传 Level 0 坐标 (0,0) 和当前层级的尺寸 dim
        # 新的 KfbSlide 会自己在内部处理循环拼接
        region_data = slide.read_region(location=(0, 0), level=level, size=dim)
        
        # 4. 统一数据格式 (处理 KfbSlide 返回 Numpy 而 OpenSlide 返回 PIL 的区别)
        if isinstance(region_data, np.ndarray):
            # 如果是 KFB (返回的是 Numpy 数组)
            thumbnail = region_data
            # 确保是 RGB -> BGR (OpenCV 需要 BGR)
            # 假设 KfbSlide 内部转换为了 RGB (看代码 convert("RGB") 是有的)
            thumbnail = thumbnail[:, :, ::-1]
        else:
            # 如果是 SVS (OpenSlide 返回的是 PIL Image)
            thumbnail = np.array(region_data)
            # RGBA -> RGB -> BGR
            if thumbnail.shape[2] == 4:
                thumbnail = thumbnail[:, :, :3]
            thumbnail = thumbnail[:, :, ::-1]

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"读取切片失败 {os.path.basename(slide_path)}: {e}")
        return

    # --- 以下逻辑保持不变 ---
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
    split_patches(thumbnail, mask_fortrain, save_dir=save_dir, save_ind=index, label_count_dict=label_dict)

def check_overlap(bboxes, bbox):
    overlap = False
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    # print('bbox', bbox)
    for cur_bbox in bboxes:
        x_left_bound, x_right_bound, y_top_bound, y_bottom_bound = cur_bbox[0], cur_bbox[2], cur_bbox[1], cur_bbox[3]
        if (x1 <= x_right_bound and x1 >= x_left_bound) or (x2 <= x_right_bound and x2 >= x_left_bound):
            if (y1 <= y_bottom_bound and y1 >= y_top_bound) or (y2 <= y_bottom_bound and y2 >= y_top_bound):
                overlap = True
                return overlap
    return overlap

def sample_patches(region, sample_num, contour, roi_label, base_name, i):
    bboxes = []
    size_w, size_h = 256, 256
    margin = 30
    num = 0
    sample_times = 0
    height, width = region.shape[0], region.shape[1]
    print('selection')
    # print('height, width', height, width)
    while num < sample_num and sample_times < 100000:
        sample_times += 1
        start_x = random.randrange(0, width)
        start_y = random.randrange(0, height)
        start_point = (start_x, start_y)
        top_left = start_point
        top_right = (start_x + size_w, start_y)
        bottom_left = (start_x, start_y + size_h)
        bottom_right = (start_x + size_w, start_y + size_h)
        signed_dist0 = cv2.pointPolygonTest(contour, top_left, True)
        signed_dist1 = cv2.pointPolygonTest(contour, top_right, True)
        signed_dist2 = cv2.pointPolygonTest(contour, bottom_left, True)
        signed_dist3 = cv2.pointPolygonTest(contour, bottom_right, True)
        if (signed_dist0 > margin) and (signed_dist1 > margin) and \
                (signed_dist2 > margin) and (signed_dist3 > margin):
            x1, y1, x2, y2= top_left[0], top_left[1], bottom_right[0], bottom_right[1]
            bbox = (x1, y1, x2, y2)
            print('here')
            if not check_overlap(bboxes, bbox):
                bboxes.append(bbox)
                cv2.imwrite('/data2/Caijt/WSI_ROI/patches_v4/{}/vis_patches_{}_{}_{}.png'.format(roi_label, base_name, i, num),
                            region[y1:y2, x1:x2])
                num += 1
                print('congratulations!!!')
    return bboxes
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def deduplicate_paths(file_paths):
    file_dict = {}  # 用于存储文件名及其对应的路径列表的字典

    # 构建文件名及其对应的路径列表的字典
    for path in file_paths:
        file_name = path.split('/')[-1]  # 获取文件名
        if file_name in file_dict:
            file_dict[file_name].append(path)
        else:
            file_dict[file_name] = [path]

    # 提取唯一路径，生成新的列表
    unique_paths = []
    for file_name, paths in file_dict.items():
        unique_paths.append(paths[0])  # 只保留一个路径

    return unique_paths



if __name__ == '__main__':
    # get_properties(slide_path)
    save_dir='/sdsl/code/WSI_DATA'
    img_folder = '/sdsl/data'
    # class映射的值必须从1开始，且为连续的整数
    label_dict = {
                  "脉管癌栓": 1,
                  "肿瘤芽": 2,
                  }

    label_color = {
        '脉管癌栓': (0, 255, 0),
        '肿瘤芽': (255, 0, 0)
    }

    # label_num = 6 + 1

    label_num = 2 + 1
    if not os.path.exists(os.path.join(save_dir, 'ImageSets')):
        os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisWSI'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    # mpp=0.25, level=5; mpp=0.5, level=4;
    level = 1
    index = 0
    mpp_dic = {"mpp=0.25": 0, "mpp=0.50": 0, "other": 0}
    from pydaily import filesystem
    # ext_list = ['.kfb', '.mrxs', '.ndpi', '.sdpc']
    # ext_list = ['.mrxs']
    ext_list = ['.kfb', '.svs']
    for ext in ext_list:
        imglist = filesystem.find_ext_files(img_folder, ext)
        imglist = deduplicate_paths(imglist)
        image_name_list = [os.path.splitext(os.path.basename(i))[0] for i in imglist]
        json_list = filesystem.find_ext_files(img_folder, '.json')
        json_name_list = [os.path.splitext(os.path.basename(i))[0] for i in json_list]
        # print(image_name_list)
        # print("json_name_list")
        # print(json_name_list)
        res = set(json_name_list) - set(image_name_list)
        print(res)
        print(len(res))
        for ind,slide_path in tqdm(enumerate(imglist)):
            json_name = slide_path.replace(ext, '.json')
            json_name = os.path.splitext(os.path.basename(json_name))[0]
            slide_name = os.path.basename(slide_path)
            if json_name not in json_name_list:
                continue
            else:
                level_dim_dict = get_level_dim_dict(slide_path)
                if level_dim_dict is None or not level_dim_dict:
                    continue
                json_path = json_list[json_name_list.index(json_name)]
                roi_contours, roi_labels = get_contours(json_path,level)
                if (len(roi_contours) == 0):
                    print(f"无标注： {slide_name}")
                    continue

                vis_anno(slide_path, roi_contours, roi_labels, level=level, save_dir=save_dir, index=index)
                index += 1
    print(mpp_dic)



