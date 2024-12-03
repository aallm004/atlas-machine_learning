#!/usr/bin/env python3
"""Module for class Yolo"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchors_count = output.shape[:3]
            
            # Get confidence and class probabilities
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

            # Process box coordinates
            box = np.zeros(output[..., :4].shape)
            
            # Create grid
            grid_x = np.arange(grid_width)
            grid_x = np.tile(grid_x, grid_height)
            grid_x = grid_x.reshape(grid_height, grid_width, 1)
            
            grid_y = np.arange(grid_height)
            grid_y = np.tile(grid_y, grid_width)
            grid_y = grid_y.reshape(grid_width, grid_height).T
            grid_y = grid_y.reshape(grid_height, grid_width, 1)
            
            # Box centers
            box[..., 0] = (1 / (1 + np.exp(-output[..., 0]))) + grid_x
            box[..., 1] = (1 / (1 + np.exp(-output[..., 1]))) + grid_y
            
            # Box width and height
            anchors = self.anchors[idx].reshape(1, 1, anchors_count, 2)
            box[..., 2] = np.exp(output[..., 2]) * anchors[..., 0]
            box[..., 3] = np.exp(output[..., 3]) * anchors[..., 1]
            
            # Convert to corner coordinates
            box[..., 0] = box[..., 0] / grid_width
            box[..., 1] = box[..., 1] / grid_height
            box[..., 2] = box[..., 2] / grid_width
            box[..., 3] = box[..., 3] / grid_height
            
            # Scale to image size
            box[..., 0] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 2] *= image_size[1]
            box[..., 3] *= image_size[0]
            
            # Convert to boundary box format
            box_mins = box[..., 0:2] - (box[..., 2:4] / 2)
            box_maxs = box[..., 0:2] + (box[..., 2:4] / 2)
            box = np.concatenate([box_mins, box_maxs], axis=-1)
            
            boxes.append(box)

        return boxes, box_confidences, box_class_probs
