import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import onnxruntime as ort
import cv2 as cv
#from classes import NuScenesLidarDataset

 # TODO: when you get SemanticKITTI installed, remember that their lidar uses
 # 64 beams, change H to 64 when working with that.
SEQ_DIR = 'semantickitti_seq08/dataset/sequences/08'

arch = yaml.safe_load(open("lidar-bonnetal/darknet53/arch_cfg.yaml"))
data = yaml.safe_load(open("lidar-bonnetal/darknet53/data_cfg.yaml"))
s = arch["dataset"]["sensor"]
H, W = s["img_prop"]["height"], s["img_prop"]["width"]
fov_up, fov_down = np.radians(s["fov_up"]), np.radians(s["fov_down"])
fov = abs(fov_up) + abs(fov_down)
means = np.array(s["img_means"], np.float32)[:, None, None]
stds = np.array(s["img_stds"], np.float32)[:, None, None]

lut = np.zeros(260, np.uint32)
for k, val in data["learning_map_inv"].items():
    lut[k] = val

providers = [
    ('CUDAExecutionProvider', {
        'cudnn_conv_algo_search': 'DEFAULT',  # try this first; 'EXHAUSTIVE' is the fallback option
    }),
    'CPUExecutionProvider',
]

sess = ort.InferenceSession("model.onnx", providers=providers)
input_name = sess.get_inputs()[0].name
print("Providers in use:", sess.get_providers())

def load_kitti_bin(filepath):
    scan = np.fromfile(filepath, dtype=np.float32)
    return scan.reshape((-1, 4))

def foveated_mask(points, near_r=10, far_r=100, keep_near=1.0, keep_far=0.1):
    depth = np.linalg.norm(points[:, :3], axis=1)
    frac = np.clip((depth - near_r) / (far_r - near_r), 0, 1)
    keep_prob = keep_near - frac * (keep_near - keep_far)
    return np.random.rand(len(points)) < keep_prob

def preprocess(pts):
    x, y, z, r = pts.T
    depth = np.linalg.norm(pts[:, :3], axis=1)
    yaw, pitch = -np.arctan2(y, x), np.arcsin(z / (depth + 1e-8))
    u = np.clip(np.floor(0.5 * (yaw / np.pi + 1) * W), 0, W - 1).astype(int)
    v = np.clip(np.floor((1 - (pitch + abs(fov_down)) / fov) * H), 0, H - 1).astype(int)
    order = np.argsort(depth)[::-1]
    img = np.full((5, H, W), -1, np.float32)
    img[:, v[order], u[order]] = np.stack([depth, x, y, z, r])[:, order]
    mask = img[0] > 0
    img = (img - means) / stds
    img *= mask
    return img[None].astype(np.float32), u, v


def load_class_image(idx):
    points = load_kitti_bin(f'{SEQ_DIR}/velodyne/{idx:06d}.bin')
    mask = foveated_mask(points)
    points = points[mask]

    inp, u, v = preprocess(points)

    t0 = time.time()
    probs = sess.run(None, {input_name: inp})[0]
    infer_time = time.time() - t0

    pred = probs[0].argmax(0)[v, u]
    labels = lut[pred]

    class_image = np.full((H, W), -1, dtype=np.int32)
    class_image[v, u] = labels
    return np.ma.masked_where(class_image < 0, class_image), infer_time


# --- set up the plot once, using frame 0 ---
fig, ax = plt.subplots(figsize=(14, 4))
first_image, _ = load_class_image(0)
img_display = ax.imshow(first_image, cmap='tab20')

for idx in range(4071):
    t0 = time.time()
    points = load_kitti_bin(f'{SEQ_DIR}/velodyne/{idx:06d}.bin')
    t1 = time.time()

    # mask = foveated_mask(points)
    # points = points[mask]
    t2 = time.time()

    inp, u, v = preprocess(points)
    t3 = time.time()

    probs = sess.run(None, {input_name: inp})[0]
    t4 = time.time()

    pred = probs[0].argmax(0)[v, u]
    labels = lut[pred]
    class_image = np.full((H, W), -1, dtype=np.int32)
    class_image[v, u] = labels
    t5 = time.time()

    img_display.set_data(class_image)
    ax.set_title(f'Frame {idx} | {1/(t4-t3):.1f} Hz (inference only)')
    plt.pause(0.001)
    t6 = time.time()

    print(f"load={t1-t0:.3f} mask={t2-t1:.3f} preprocess={t3-t2:.3f} "
          f"infer={t4-t3:.3f} postproc={t5-t4:.3f} plot={t6-t5:.3f}")
    # class_image, infer_time = load_class_image(idx)

    # img_display.set_data(class_image)
    # ax.set_title(f'Frame {idx} | {1/infer_time:.1f} Hz (inference only)')
    # plt.pause(0.001)

plt.show()





