import cv2
import numpy as np
import math
import os

from scipy.ndimage.filters import gaussian_filter
import torch

from model import Joint, Skeleton

# 遍历时到关节点和paf_xy的映射
# 从颈部向四肢和头部关节点的拓扑排序，保证遍历时，当前连接不可能指向已遍历过的关节点
map2joints = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
              [13, 14], [1, 0], [0, 15], [15, 17],
              [0, 16], [16, 18], [2, 17], [5, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]

map2paf = [[0, 1], [14, 15], [22, 23], [16, 17], [18, 19], [24, 25], [26, 27], [6, 7], [2, 3], [4, 5], [8, 9], [10, 11],
           [12, 13], [30, 31], [32, 33],
           [36, 37], [34, 35], [38, 39], [20, 21], [28, 29], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], [50, 51]]

map2str = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
           "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
           "RBigToe", "RSmallToe", "RHeel", "Background"]


def draw_body_pose(img: np.ndarray, skeletons: list[Skeleton]):
    stick_width = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 0], [255, 255, 85], [255, 255, 170],
              [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255]]
    for skel in skeletons:
        for joint, color in zip(skel.joints, colors):
            if joint.score < 0.1:
                continue
            cv2.circle(img, joint.get_image_coord(), 4, color, thickness=-1)

    for skel in skeletons:
        for limb, color in zip(map2joints, colors):
            joint0 = skel[limb[0]]
            joint1 = skel[limb[1]]
            if joint0.score < 0.1 or joint1.score < 0.1:
                continue
            cur_canvas = img.copy()
            x0, y0 = joint0.get_image_coord()
            x1, y1 = joint1.get_image_coord()
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            polygon = cv2.ellipse2Poly(((x0 + x1) // 2, (y0 + y1) // 2), (int(length / 2), stick_width), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)

    return img


def pad_down_right_corner(img: np.ndarray) -> np.ndarray:
    stride = 8
    pad_value = 128
    h, w, _ = img.shape
    pad_d = 0 if h % stride == 0 else stride - h % stride
    pad_r = 0 if w % stride == 0 else stride - w % stride

    if pad_d == 0 and pad_r == 0:
        padded_img = img
    else:
        padded_img = cv2.copyMakeBorder(img, 0, pad_d, 0, pad_r, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img


def preprocess_image2tensor(img: np.ndarray, scale: float) -> torch.Tensor:
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded_img = pad_down_right_corner(resized_img)
    im = np.transpose(np.float32(padded_img), (2, 0, 1)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    data = torch.from_numpy(im).float()
    data.unsqueeze_(0)
    return data


def postprocess_heatmap_paf(heatmap: np.ndarray, paf: np.ndarray, output_shape: (int, int)):
    def __process(_x: np.ndarray):
        _y = np.transpose(np.squeeze(_x), (1, 2, 0))
        _y = cv2.resize(_y, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)
        return _y.transpose(2, 0, 1)

    return __process(heatmap), __process(paf)


# 返回所有关节点的候选点列表
def nms_heatmap(heatmaps: np.ndarray, threshold: float) -> list[list[Joint]]:
    joints = []

    for heatmap in heatmaps[:25]:
        smooth_heatmap = gaussian_filter(heatmap, sigma=3)

        map_left = np.zeros(smooth_heatmap.shape)
        map_left[1:, :] = smooth_heatmap[:-1, :]
        map_right = np.zeros(smooth_heatmap.shape)
        map_right[:-1, :] = smooth_heatmap[1:, :]
        map_up = np.zeros(smooth_heatmap.shape)
        map_up[:, 1:] = smooth_heatmap[:, :-1]
        map_down = np.zeros(smooth_heatmap.shape)
        map_down[:, :-1] = smooth_heatmap[:, 1:]

        # 求热图中超过阈值的峰值点，作为关节点的候选点
        peaks_binary = np.logical_and.reduce(
            (smooth_heatmap > threshold, smooth_heatmap >= map_left, smooth_heatmap >= map_right,
             smooth_heatmap >= map_up,
             smooth_heatmap >= map_down))  # 逻辑与

        peaks = np.argwhere(peaks_binary)
        candidate_joints = [Joint(x, y, heatmap[y, x].item()) for y, x in peaks]
        joints.append(candidate_joints)

    return joints


def connect_joints(joints: list[list[Joint]], paf: np.ndarray, ori_img_w, threshold) -> list[(Joint, Joint, float)]:
    matched_connections = []
    mid_num = 10

    # 用关节点向量和paf匹配，得到候选躯干
    for joints_idx, paf_idx in zip(map2joints, map2paf):
        score_mid = paf[paf_idx, :, :]
        candidate_joint0 = joints[joints_idx[0]]
        candidate_joint1 = joints[joints_idx[1]]
        num_joint0 = len(candidate_joint0)
        num_joint1 = len(candidate_joint1)
        if num_joint0 != 0 and num_joint1 != 0:
            candidate_connection = []
            for joint0 in candidate_joint0:
                for joint1 in candidate_joint1:
                    vec = np.subtract(joint1.xy, joint0.xy)
                    norm = np.linalg.norm(vec).item()
                    if norm == 0.:
                        continue
                    vec = np.divide(vec, norm)

                    start_end = zip(np.linspace(joint0.x, joint1.x, num=mid_num),
                                    np.linspace(joint0.y, joint1.y, num=mid_num))

                    vec_paf = np.array([score_mid[:, int(round(y)), int(round(x))] for x, y in start_end])

                    score_midpoints = np.multiply(vec_paf, vec).sum(axis=1)  # cos <vec, vec_paf>
                    score_with_dist_prior = (score_midpoints.mean().item() +
                                             min(ori_img_w / 2. / norm - 1., 0.))

                    if (score_with_dist_prior > 0. and
                            len(np.argwhere(score_midpoints > threshold)) > 0.8 * mid_num):
                        candidate_connection.append((joint0, joint1, score_with_dist_prior))

            candidate_connection = sorted(candidate_connection, key=lambda x: x[2], reverse=True)
            connection = []
            matched_joints_id = []
            for joint0, joint1, score in candidate_connection:
                if id(joint0) not in matched_joints_id and id(joint1) not in matched_joints_id:
                    connection.append((joint0, joint1, score))
                    matched_joints_id.append(id(joint0))
                    matched_joints_id.append(id(joint1))
                    if len(connection) >= min(num_joint0, num_joint1):
                        break

            matched_connections.append(connection)
        else:
            matched_connections.append([])

    return matched_connections


def rebuild_skeletons(connections: list[(Joint, Joint, float)]) -> list[Skeleton]:
    candidate_skeleton = []

    # 躯干尝试搭建骨架
    for (joint0_idx, joint1_idx), connection in zip(map2joints, connections):
        if not connection:
            continue
        for joint0, joint1, score in connection:
            for cand_skel in candidate_skeleton:
                if cand_skel[joint0_idx] is joint0:
                    # 因为拓扑排序，只可能是id0出现重复，
                    # 而一般id1不可能出现在之前的骨架中，
                    # 除非允许两个骨架的同一个关节位置共用一个关节点
                    cand_skel[joint1_idx] = joint1
                    cand_skel.num_joints += 1
                    cand_skel.score += score
                    break
            else:
                cand_skel = Skeleton()
                cand_skel[joint0_idx] = joint0
                cand_skel[joint1_idx] = joint1
                cand_skel.num_joints = 2
                cand_skel.score = joint0.score + joint1.score + score
                candidate_skeleton.append(cand_skel)

    skeletons = []
    for cand_skel in candidate_skeleton:
        if cand_skel.score >= 4. and cand_skel.num_joints / cand_skel.score >= 0.4:
            skeletons.append(cand_skel)

    return skeletons


def visualize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    img = heatmap.clip(min=0, max=1)
    img *= 255
    return img.astype(np.uint8)


def visualize_paf(paf_x: np.ndarray, paf_y: np.ndarray) -> np.ndarray:
    h, w = paf_x.shape
    step = 8
    threshold = 0.1

    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            vx, vy = paf_x[y, x], paf_y[y, x]
            magnitude = np.sqrt(vx ** 2 + vy ** 2)

            if magnitude > threshold:
                x_end = int(x + vx * step)
                y_end = int(y + vy * step)
                cv2.arrowedLine(img, (x, y), (x_end, y_end), 255, 1, tipLength=0.3)

    return img


def show_heatmaps_paf(heatmap: np.ndarray, paf: np.ndarray):
    import matplotlib.pyplot as plt

    heatmap_title_image = [(f'heatmap {_l}', visualize_heatmap(_h)) for _l, _h in zip(map2str, heatmap)]
    paf_title_image = [(f'paf {map2str[start]}-{map2str[end]}', visualize_paf(paf[_x], paf[_y])) for
                       (start, end), (_x, _y) in
                       zip(map2joints, map2paf)]

    flg, axes = plt.subplots(2, 26, figsize=(52, 4))

    for col in range(26):
        axes[0, col].imshow(heatmap_title_image[col][1])
        axes[0, col].set_title(heatmap_title_image[col][0])
        axes[1, col].imshow(paf_title_image[col][1])
        axes[1, col].set_title(paf_title_image[col][0])

    plt.tight_layout()
    plt.show()


def __generate_grid_xy(shape):
    grid_x = np.tile(np.arange(shape[1], dtype=np.float32), (shape[0], 1))
    grid_y = np.tile(np.arange(shape[0], dtype=np.float32), (shape[1], 1)).transpose()
    grid_xy = np.stack((grid_x, grid_y), axis=2)
    return grid_xy


def generate_heatmaps(shape: (int, int), joints_xy: list[list], sigma=5., *,
                      grid_xy: np.ndarray = None) -> np.ndarray:
    if grid_xy is None:
        grid_xy = __generate_grid_xy(shape)

    heatmaps = np.zeros((26, *shape), dtype=np.float32)
    for joints in joints_xy:
        for heatmap, xy in zip(heatmaps, joints):
            if xy is None:
                continue
            grid_vec = np.subtract(grid_xy, xy)
            square_distance = np.sum(np.square(grid_vec), axis=2)
            gaussian_heatmap = np.exp(-0.5 / sigma ** 2 * square_distance)
            np.maximum(heatmap, gaussian_heatmap, out=heatmap)

    heatmaps[25] = np.ones(shape, dtype=np.float32) - np.maximum.reduce(heatmaps)  # background
    return heatmaps


def generate_pafs(shape: (int, int), joints_xy: list[list], half_width=5., *, grid_xy: np.ndarray = None) -> np.ndarray:
    if grid_xy is None:
        grid_xy = __generate_grid_xy(shape)

    pafs = np.zeros((52, *shape), dtype=np.float32)
    for joints in joints_xy:
        for (j0, j1), (_x, _y) in zip(map2joints, map2paf):
            start = joints[j0]
            end = joints[j1]

            if start is None or end is None:
                continue

            vec = np.subtract(end, start)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0.:
                continue

            vec_unit = vec / vec_norm
            grid_vec = np.subtract(grid_xy, start)

            grid_dot = np.dot(grid_vec, vec_unit)
            grid_cross = np.cross(grid_vec, vec_unit)

            condition = np.logical_and.reduce(
                (grid_dot > 0., grid_dot < vec_norm, grid_cross > -half_width, grid_cross < half_width))

            paf = pafs[_x:_y + 1]
            roi = paf[:, condition]
            vec_o = np.expand_dims(vec_unit, 1) + roi
            vec_o /= np.linalg.norm(vec_o, axis=0)
            paf[:, condition] = vec_o

    return pafs
