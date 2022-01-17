import cv2

image = cv2.imread("D:/bellmi2/train_3_frames_aligned/2011_09_26_drive_0001_sync_02/0000000106-fseg.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




print("test")