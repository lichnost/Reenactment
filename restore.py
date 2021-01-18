import os
import numpy as np
import cv2

file = np.loadtxt('/home/lichnost/programming/work/ml/head/data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt', dtype=str)
file = np.delete(file, [200, 201, 202, 203], axis=1)
for f in file:
    if f[-1] == '':
        continue
    path = 'WFLW_images/' + f[-1]
    image = cv2.imread('/home/lichnost/programming/work/ml/head/data/WFLW/' + path)
    f[-3] = image.shape[0]
    f[-2] = image.shape[1]
    f[-1] = path
np.savetxt('/home/lichnost/programming/work/ml/head/data/WFLW/WFLW_train_annos.txt', file, delimiter=' ', fmt='%s')