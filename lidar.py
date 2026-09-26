from nuscenes.utils.data_classes import LidarPointCloud
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
     
def project_to_range_image(points, H=32, W=1024, f_up=10.708, f_down=-58.116): 
    f_up = np.radians(f_up)
    f_down = np.radians(f_down)
    fov = abs(f_up) + abs(f_down)
    
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    depth = np.sqrt(x**2 + y**2 + z**2)

    yaw = np.arctan2(y,x)
    pitch = np.arcsin(z/(depth+1e-8))

    u = 0.5*(1-yaw/np.pi)*W
    v = (1 -(pitch + abs(f_down))/fov)*H
    
    u = np.clip(u, 0, W - 1).astype(np.int32)
    v = np.clip(v, 0, H - 1).astype(np.int32)

    range_image = np.full((H, W, 5), -1, dtype=np.float32)  # channels: range, x, y, z, intensity
    range_image[v, u] = np.stack([depth, x, y, z, intensity], axis=1)

    print("unique row indices used:", np.unique(v).size, "out of", H)
    return range_image
    
#nusc = ns(version='v1.0-mini', dataroot='nuscenes-mini/', verbose=True)
pc = LidarPointCloud.from_file('nuscenes-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616447606.pcd.bin')
points = pc.points.T

depth = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
pitch = np.arcsin(points[:,2] / (depth + 1e-8))
print("pitch range (degrees):", np.degrees(pitch).min(), np.degrees(pitch).max())

range_image = project_to_range_image(points, f_up=np.degrees(pitch).max(), f_down=np.degrees(pitch).min())

plt.figure(figsize=(12, 3))
plt.imshow(range_image[:, :, 0], cmap='viridis', aspect='auto')
plt.colorbar(label='depth (m)')
plt.title('Range Image')
plt.xlabel('azimuth (u)')
plt.ylabel('elevation (v)')
plt.show()




