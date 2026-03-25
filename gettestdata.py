import os
import shutil
import argparse

def extract_test_images(image_dir, test_list_file, output_dir):
    """
    从图像目录中提取测试集图像到指定目录
    
    参数:
    image_dir: 包含所有图像的目录路径
    test_list_file: 包含测试集文件名的文本文件路径
    output_dir: 输出测试集图像的目录路径
    """
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取测试集文件名列表
    with open(test_list_file, 'r', encoding='utf-8') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(test_files)} 个测试集文件")
    
    # 统计变量
    success_count = 0
    fail_count = 0
    
    # 提取每个测试集图像
    for filename in test_files:
        filename += '.jpg'
        # 构建源文件路径
        src_path = os.path.join(image_dir, filename)
        
        # 构建目标文件路径
        dst_path = os.path.join(output_dir, filename)
        
        try:
            # 检查源文件是否存在
            if os.path.exists(src_path):
                # 复制文件到目标目录
                shutil.copy2(src_path, dst_path)
                success_count += 1
            else:
                print(f"警告: 文件不存在: {src_path}")
                fail_count += 1
                
        except Exception as e:
            print(f"错误: 复制文件 {filename} 时出错: {str(e)}")
            fail_count += 1
    
    # 打印统计信息
    print(f"\n提取完成!")
    print(f"成功提取: {success_count} 个文件")
    print(f"提取失败: {fail_count} 个文件")
    print(f"测试集已保存到: {os.path.abspath(output_dir)}")

def main():
    image_dir = '/sdsl/code/WSI_DATA/JPEGImages'
    test_list = '/sdsl/code/WSI_DATA/ImageSets/Segmentation/test.txt'
    output_dir = '/sdsl/code/post_data/tests'
    
    # 调用提取函数
    extract_test_images(image_dir, test_list, output_dir)

if __name__ == "__main__":
    main()