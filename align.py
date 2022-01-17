from alignment import align
import cv2

path1 ="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000004-seg.png"
path2 ="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000005-seg.png"
path3 = "C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000006-seg.png"
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
img3 = cv2.imread(path3)

img1_new,img2_new,img3_new = align(img1, img2, img3)


path1_new ="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000004-seg_new.png"
path2_new ="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000005-seg_new.png"
path3_new ="C:/Users/bellmi2/Documents/kitti/KITTI-Motion/output_img/KITTI-Motion_02/0000000006-seg_new.png"

cv2.imwrite(path1_new,img1_new)
cv2.imwrite(path2_new,img2_new)
cv2.imwrite(path3_new,img3_new)
