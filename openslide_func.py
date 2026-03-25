from Slide.KfbSlide.kfbslide import KfbSlide
import os.path
import openslide  # 专门用于处理 SVS, TIFF, NDPI 等
from scipy import misc
from threading import Lock
from LRUCacheDict import LRUCacheDict

slides = LRUCacheDict()
_dict_lock = Lock()

def openSlide(filename):

    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    with _dict_lock:
        if filename in slides:
            return slides[filename]

        slide = None
        try:
            # 1. 处理 KFB (江丰)
            if ext == 'kfb':
                slide = KfbSlide(filename)
            
            # 注：OpenSlide 其实支持 svs, tif, ndpi, vms, mrxs 等多种格式
            elif ext in ['svs', 'tif', 'tiff', 'ndpi', 'mrxs', 'scn']:
                slide = openslide.OpenSlide(filename)
            
            # 3. 不支持的格式
            else:
                print(f"不支持的切片格式: {ext}")
                return None

        except Exception as e:
            print(f"打开切片失败 {filename}: {e}")
            return None

        if slide is not None:
            slides[filename] = slide
            print("切片加载完成：" + filename)
        
        return slide