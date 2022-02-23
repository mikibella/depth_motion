import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import project
#import tensorflow as tf

fig = plt.figure(figsize=(30, 30))

img1 = cv2.imread("G:/Meine Ablage/full_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png")
img2 = Image.open("G:/Meine Ablage/full_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png")
img3 = Image.open("G:/Meine Ablage/full_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000002.png")
# Adds a subplot at the 1st position
fig.add_subplot(1, 3, 1)
  
# showing image
plt.imshow(img1)
plt.axis('off')
plt.title("First")
  
# Adds a subplot at the 2nd position
fig.add_subplot(1, 3, 2)
   
# showing image
plt.imshow(img2)
plt.axis('off')
plt.title("Second")

# Adds a subplot at the 2nd position
fig.add_subplot(1, 3, 3)
   
# showing image
plt.imshow(img3)
plt.axis('off')
plt.title("Third")

img1 = np.array(img1)
ego_vec= np.matrix('7.84912e-06;-0.00074030977;0.05050047;0.0008832767;0.0034075254;0.00034070958')
intrinsic=np.matrix('241.67446312399355,0.0,204.16801030595812;0.0,246.28486826666665,59.000832;0.0,0.0,1.0')
intrinsic_inv=np.linalg.inv(intrinsic)


pred_depth = np.load(os.path.join("G:/Meine Ablage/eval_motion/0000000001.npy"))
#pred_depth = 0.54 * 721 / (pred_depth)
pred_depth = pred_depth/pred_depth.max()*255
#pred_depth = pred_depth/pred_depth.max()*255
#pred_depth[pred_depth<1e-3]=0
#pred_depth[pred_depth>80]=0
egomotion_mat = project._egomotion_vec2mat(ego_vec,1)
print(egomotion_mat.shape)
pred_depth = cv2.resize(pred_depth,(1242,375),interpolation=cv2.INTER_LINEAR)
pred_depth= pred_depth.astype(np.float32)
intrinsic_inv= intrinsic_inv.astype(np.float32)
grid = _meshgrid_abs(375,1242)
grid = np.tile(np.expand_dims(grid, 0), [1, 1, 1])
print(grid.shape)
#grid = np.tile(np.expand_dims(grid, 0), [1, 1, 1])
pred_depth = np.reshape(pred_depth, [1, 1, 375 * 1242])
cam_coords = cam_coords = tf.matmul(intrinsic_inv, grid) * pred_depth
print(cam_coords.shape)
ones = np.ones([1, 1, 375 * 1242])
cam_coords_hom = np.concat([cam_coords, ones], axis=1)
# Get projection matrix for target camera frame to source pixel frame
hom_filler = np.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
hom_filler = np.tile(hom_filler, [1, 1, 1])
intrinsic_mat_hom = np.concat(
    [intrinsic, np.zeros([1, 3, 1])], axis=2)
intrinsic_mat_hom = np.concat([intrinsic_mat_hom, hom_filler], axis=1)
proj_target_cam_to_source_pixel = np.matmul(intrinsic_mat_hom, egomotion_mat)
source_pixel_coords = _cam2pixel(cam_coords_hom,
                                  proj_target_cam_to_source_pixel)
source_pixel_coords = np.reshape(source_pixel_coords,
                                  [1, 2, 375, 1242])
source_pixel_coords = np.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
projected_img, mask = _spatial_transformer(img1, source_pixel_coords)


fig2 = plt.figure(figsize=(30, 30))
fig.add_subplot(1, 3, 1)
  
# showing image
plt.imshow(projected_img)
plt.axis('off')
plt.title("First")
  
# Adds a subplot at the 2nd position
fig.add_subplot(1, 3, 2)
   
# showing image
plt.imshow(mask)
plt.axis('off')
plt.title("Second")