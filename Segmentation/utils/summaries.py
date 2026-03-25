import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    #def visualize_image(self, writer, dataset, image, target, output, global_step):
    #    grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        #writer.add_image('Image', grid_image, global_step)
       # grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
        #                                               dataset=dataset), 3, normalize=True, range=(0, 255))
       # writer.add_image('Predicted label', grid_image, global_step)
       # grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
       #                                                dataset=dataset), 3, normalize=True, range=(0, 255))
        #writer.add_image('Groundtruth label', grid_image, global_step)
    def visualize_image(self, writer, dataset, image, target, output, global_step):
        # 可视化输入图像
        grid_image = make_grid(image[:3].clone().cpu().data, nrow=3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        # 可视化预测结果
        predicted_labels = torch.max(output[:3], 1)[1].detach().cpu().numpy()
        grid_image = make_grid(
            decode_seg_map_sequence(predicted_labels, dataset=dataset),
            nrow=3,
            normalize=True  # 如果启用 normalize，则需要 range 参数
        )
        writer.add_image('Predicted label', grid_image, global_step)

        # 可视化真实标签
        groundtruth_labels = torch.squeeze(target[:3], 1).detach().cpu().numpy()
        grid_image = make_grid(
            decode_seg_map_sequence(groundtruth_labels, dataset=dataset),
            nrow=3,
            normalize=True  # 如果启用 normalize，则需要 range 参数
        )
        writer.add_image('Groundtruth label', grid_image, global_step)