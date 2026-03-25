import sqlite3
import json
from pydaily import filesystem
import os
import glob

def db2json(db_path, json_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rois = []
    try:
        cursor.execute("""
            SELECT MarkGroup.groupName, Mark_label_None.position 
            FROM Mark_label_None 
            LEFT JOIN MarkGroup 
            ON Mark_label_None.groupId->"$[0]" = MarkGroup.id
        """)
    except Exception as e:
        print(f"处理 {db_path} 时出错: {e}")
        # 可以选择删除有问题的db文件，或者只是跳过
        # os.remove(db_path)
        return 0

    contents = cursor.fetchall()
    for content in contents:
        roi = {'label': content[0], 'position': json.loads(content[1])}
        rois.append(roi)
    table = {'annotation': rois}
    json_str = json.dumps(table, ensure_ascii=False, indent=4)
    with open(json_path, 'w', encoding='utf-8') as writer:
        writer.write(json_str)
    
    return 1

def get_svs_kfb_name(db_path):
    """
    查找同目录下的svs或kfb文件，并返回文件名（不含扩展名）
    """
    db_dir = os.path.dirname(db_path)
    
    # 查找svs文件
    svs_files = glob.glob(os.path.join(db_dir, "*.svs"))
    if svs_files:
        # 取第一个svs文件
        svs_file = svs_files[0]
        return os.path.splitext(os.path.basename(svs_file))[0]
    
    # 如果没有svs文件，查找kfb文件
    kfb_files = glob.glob(os.path.join(db_dir, "*.kfb"))
    if kfb_files:
        # 取第一个kfb文件
        kfb_file = kfb_files[0]
        return os.path.splitext(os.path.basename(kfb_file))[0]
    
    # 如果都没有，使用db文件名
    return os.path.splitext(os.path.basename(db_path))[0]

if __name__ == '__main__':
    # 假设data文件夹路径
    data_dir = '/sdsl/data'
    
    # 查找所有.db文件
    db_files = filesystem.find_ext_files(data_dir, ".db")
    
    print(f"找到 {len(db_files)} 个db文件")
    
    conversion_count = 0
    for a_db_file in db_files:
        # 获取同目录下的svs/kfb文件名（如果没有则用db文件名）
        base_name = get_svs_kfb_name(a_db_file)
        
        # 构建json文件路径（与db文件在同一目录，使用svs/kfb文件名）
        db_dir = os.path.dirname(a_db_file)
        json_path = os.path.join(db_dir, f"{base_name}.json")
        
        # 调用转换函数
        result = db2json(a_db_file, json_path)
        conversion_count += result
    
    print(f"转换完成! 成功转换 {conversion_count} 个文件")