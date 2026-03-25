# 对test集进行评估
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from modeling.deeplab import *
from dataloaders.datasets.jcai_region import JcaiSegmentation

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nan_to_num(Acc)
        return np.mean(Acc)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nan_to_num(MIoU)
        return np.mean(MIoU)

    def Class_IOU(self):
        # 计算每一类的 IoU
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return np.nan_to_num(MIoU)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3+ Evaluation")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='jcai_region',
                        help='dataset name')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for evaluation (default: 1)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, default 0')
    parser.add_argument('--resume', type=str, default=None, required=True,
                        help='put the path to the checkpoint file')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disables CUDA evaluation')
    parser.add_argument('--split', type=str, default='test',
                        help='split to evaluate on (test or val)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    print(f"Evaluation settings: Dataset: {args.dataset}, Split: {args.split}")
    print(f"Loading checkpoint: {args.resume}")

    # 1. 定义数据加载器
    test_set = JcaiSegmentation(args, split=args.split)
    
    # 这里的 batch_size 建议设为 1 或者显存允许的最大值，shuffle 必须为 False
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.workers, pin_memory=True)

    # 2. 定义模型
    # 类别数：背景 + 脉管癌栓 + 肿瘤芽 = 3
    nclass = test_set.NUM_CLASSES 
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=False, # Eval 模式不需要 sync_bn
                    freeze_bn=False)

    # 3. 加载权重
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    
    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    
    # 处理 DataParallel 带来的 'module.' 前缀问题
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)

    if args.cuda:
        model = model.cuda()
    
    model.eval()
    evaluator = Evaluator(nclass)
    evaluator.reset()

    # 4. 开始评估
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            
        with torch.no_grad():
            output = model(image)
        
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        
        # 添加到评估器
        evaluator.add_batch(target, pred)

    # 5. 计算并打印指标
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    Class_IoU = evaluator.Class_IOU()

    print('\n' + "="*50)
    print("Evaluation Results on [ {} ] set".format(args.split))
    print("="*50)
    print(f"Total Images: {len(test_set)}")
    print(f"Pixel Accuracy (OA):       {Acc:.4f}")
    print(f"Mean IoU (mIoU):           {mIoU:.4f}")
    print("-" * 50)
    print("Per Class IoU:")
    
    # 你的类别字典
    class_names = ['Background', 'Embolus (脉管癌栓)', 'Tumor Buds (肿瘤芽)']
    
    for i, iou in enumerate(Class_IoU):
        name = class_names[i] if i < len(class_names) else f"Class_{i}"
        print(f"  {i}: {name:<20} : {iou:.4f}")
    
    print("="*50)

    # 简单的加权计算：(癌栓IoU + 肿瘤芽IoU) / 2，不看背景
    # 这有助于你判断病灶识别能力
    if len(Class_IoU) >= 3:
        lesion_miou = (Class_IoU[1] + Class_IoU[2]) / 2.0
        print(f"Lesion mIoU (Class 1 & 2): {lesion_miou:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()


'''
python eval.py \
  --backbone resnet \
  --dataset jcai_region \
  --resume /sdsl/code/Segmentation/run/jcai_region/sdsl/experiment_22/checkpoint_best_bg_embolus.pth.tar \
  --gpu-ids 0 \
  --split test
'''