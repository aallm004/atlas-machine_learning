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
            
            # Extract confidence and class probabilities
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

           # Process box coordinates
            box = np.zeros(output[..., :4].shape)
            
            # Center coordinates
            box[..., 0] = (1 / (1 + np.exp(-output[..., 0])) + 
                          np.tile(np.arange(grid_width), grid_height).reshape(grid_height, grid_width, 1))
            box[..., 1] = (1 / (1 + np.exp(-output[..., 1])) + 
                          np.tile(np.arange(grid_height), grid_width).reshape(grid_width, grid_height).T.reshape(grid_height, grid_width, 1))
            
            # Width and height
            box[..., 2] = np.exp(output[..., 2]) * self.anchors[idx, :, 0]
            box[..., 3] = np.exp(output[..., 3]) * self.anchors[idx, :, 1]

            box[..., 0] *= image_size[1] / grid_width
            box[..., 1] *= image_size[0] / grid_height
            box[..., 2] *= image_size[1] / grid_width
            box[..., 3] *= image_size[0] / grid_height

            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2
            box[..., 2] += box[..., 0]
            box[..., 3] += box[..., 1]

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
