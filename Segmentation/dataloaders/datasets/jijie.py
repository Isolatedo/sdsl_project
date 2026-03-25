from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from dataloaders.datasets.pascal import VOCSegmentation
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class FeiaiSegmentation(Dataset):
    """
    Jijie dataset with class filtering support
    """
    NUM_CLASSES = 13+1

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('jijie'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        
        # 类别过滤配置
        self.selected_classes = getattr(args, 'selected_classes', None)  # 用户选择的类别ID列表
        if self.selected_classes is not None:
            # +1 是为了包含背景类（索引0）
            self.NUM_CLASSES = len(self.selected_classes) + 1
            # 创建类别映射：原始类别ID -> 新类别ID
            self.class_mapping = {0: 0}  # 背景类保持为0
            for new_id, old_id in enumerate(self.selected_classes, 1):
                self.class_mapping[old_id] = new_id
            print(f"启用类别过滤: 选择类别 {self.selected_classes}")
            print(f"类别映射: {self.class_mapping}")
            print(f"新的类别数量: {self.NUM_CLASSES}")
        else:
            self.class_mapping = None
            print("使用所有13个类别进行训练")

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
            elif split == 'test':
                return self.transform_val(sample)  # 测试集使用与验证集相同的变换
        
        # 如果没有匹配的分割，返回验证集变换（而不是None）
        return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        
        # 如果启用了类别过滤，则重新映射类别标签
        if self.class_mapping is not None:
            _target = self._remap_classes(_target)

        return _img, _target
    
    def _remap_classes(self, target_pil):
        """
        重新映射类别标签
        将原始的13+1类标签映射到用户选择的类别
        """
        target_array = np.array(target_pil)
        new_target = np.zeros_like(target_array)
        
        # 应用类别映射
        for old_class, new_class in self.class_mapping.items():
            new_target[target_array == old_class] = new_class
        
        # 将未选择的类别标记为背景（0）
        # 对于不在映射中的类别，保持为0（背景）
        mask = np.zeros_like(target_array, dtype=bool)
        for old_class in self.class_mapping.keys():
            mask |= (target_array == old_class)
        new_target[~mask] = 0
        
        return Image.fromarray(new_target.astype(np.uint8))

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
