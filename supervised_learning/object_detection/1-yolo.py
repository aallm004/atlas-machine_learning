#!/usr/bin/env python3
"""module for process_outputs and yoloooo"""
import numpy as np
from tensorflow import keras


class Yolo:
    def __init__(self, model_path, classes_path, class_threshold,
                 nms_threshold, anchors):
        self.model = keras.models.load_model(model_path)
        self.class_names = [line.strip() for line in open(classes_path)]
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process YOLO model outputs for object detection"""
        boxes, confidences, class_probs = [], [], []
        img_height, img_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchors_count, _ = output.shape

            # Generate coordinate grids
            grid_x = np.arange(grid_w).reshape(1, grid_w, 1)
            grid_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            # Extract box parameters
            box_xy = 1 / (1 + np.exp(-output[..., :2]))
            box_xy[..., 0] = (box_xy[..., 0] + grid_x) / grid_w
            box_xy[..., 1] = (box_xy[..., 1] + grid_y) / grid_h

            # Process anchor dimensions
            anchor_sizes = self.anchors[i]
            box_wh = np.exp(output[..., 2:4])
            box_wh *= anchor_sizes.reshape(1, 1, -1, 2)
            box_wh[..., 0] /= self.model.input.shape[1]
            box_wh[..., 1] /= self.model.input.shape[2]

            # Convert to corner coordinates
            box_mins = box_xy - (box_wh / 2)
            box_maxes = box_xy + (box_wh / 2)

            # Scale to image size
            boxes_output = np.concatenate([
                box_mins[..., 0:1] * img_width,
                box_mins[..., 1:2] * img_height,
                box_maxes[..., 0:1] * img_width,
                box_maxes[..., 1:2] * img_height
            ], axis=-1)

            confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
            boxes.append(boxes_output)

        return boxes, confidences, class_probs
