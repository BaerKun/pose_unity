import numpy as np
import torch
from model import PoseBody25, Skeleton
import util

_try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseBody25Detector:
    def __init__(self, weights_path: str):
        self.model = PoseBody25()
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.num_joints = 25
        self.num_heatmap = 26
        self.num_paf = 52

    def __call__(self, ori_img: np.ndarray, show_heatmap_paf: bool = False, device: torch.device = _try_cuda) -> list[Skeleton]:
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [0.5]
        box_size = 368
        post_box_shape = (box_size, int(box_size * ori_img.shape[1] / ori_img.shape[0]))
        threshold_joint = 0.1
        threshold_limb = 0.05
        multiplier = [x * box_size / ori_img.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((self.num_heatmap, *post_box_shape))
        paf_avg = np.zeros((self.num_paf, *post_box_shape))

        self.model.eval()
        for m in multiplier:
            scale = m
            data = util.preprocess_image2tensor(ori_img, scale)

            with torch.no_grad():
                data = data.to(device)
                self.model.to(device)
                heatmap, paf = self.model(data)
                heatmap = heatmap.cpu().numpy()
                paf = paf.cpu().numpy()

            heatmap, paf = util.postprocess_heatmap_paf(heatmap, paf, post_box_shape)
            if show_heatmap_paf:
                util.show_heatmaps_paf(heatmap, paf)
            heatmap_avg += heatmap
            paf_avg += + paf
        else:
            heatmap_avg /= len(multiplier)
            paf_avg /= len(multiplier)

        joints = util.nms_heatmap(heatmap_avg, threshold_joint)
        candidate_connections = util.connect_joints(joints, paf_avg, ori_img.shape[0], threshold_limb)
        skeletons = util.rebuild_skeletons(candidate_connections)
        for skel in skeletons:
            skel.resize(ori_img.shape[0] / box_size)
        return skeletons