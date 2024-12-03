#!/usr/bin/env python3
"""Module for class Yolo"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo with model and parameters"""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs
        
        Args:
            outputs: list of numpy.ndarrays containing predictions
            image_size: numpy.ndarray of image dimensions [height, width]
            
        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        
        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchors_count = output.shape[:3]
            
            # Extract box confidence scores
            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_conf)
            
            # Extract class probabilities
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)
            
            # Extract box coordinates
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]
            
            # Create grid indices
            cx = np.tile(np.arange(grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.tile(np.arange(grid_height), grid_width)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)
            
            # Calculate bounding box coordinates
            bx = (1 / (1 + np.exp(-tx))) + cx
            by = (1 / (1 + np.exp(-ty))) + cy
            bw = np.exp(tw) * self.anchors[idx, :, 0]
            bh = np.exp(th) * self.anchors[idx, :, 1]
            
            # Normalize to grid
            bx = bx / grid_width
            by = by / grid_height
            bw = bw / grid_width
            bh = bh / grid_height
            
            # Calculate corner coordinates (x1, y1, x2, y2)
            x1 = (bx - bw/2) * image_size[1]
            y1 = (by - bh/2) * image_size[0]
            x2 = (bx + bw/2) * image_size[1]
            y2 = (by + bh/2) * image_size[0]
            
            # Stack coordinates
            box = np.concatenate((x1, y1, x2, y2), axis=-1)
            boxes.append(box)
            
        return boxes, box_confidences, box_class_probs
