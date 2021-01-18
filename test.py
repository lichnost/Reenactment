import torch
import torchsummary
import numpy as np
from models.shape import FeatureExtractor, Transformer, ShapeGenerator, HomogeneousShapeLayer
from utils import *


shapes = np.load('/home/lichnost/programming/work/ml/head/data/WFLW/pca/train/train.npy')
shape_layer = HomogeneousShapeLayer(shapes, 2)
encoder = FeatureExtractor(boundary_num, shape_layer.num_params)
# generator = ShapeGenerator()
# transformer = Transformer(boundary_num, shape_layer, kp_num['WFLW'])
torchsummary.summary(encoder, (boundary_num, 64, 64), device='cpu')
