#!/usr/bin/env python3
"""Module for class Yolo"""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm to perform object
    detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class Constructor:
                model_path: the path to where a Darknet Keras model is stored

                classes_path: the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found

                class_t: a float representing the box score threshold for the
                initial filtering step

                nms_t: a float representing the IOU threshold for non-max
                suppression
                anchors: a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                    outputs: the number of outputs (predictions) made by the
                    Darknet model
                    anchor_boxes: the number of anchor boxes used for each
                    prediction
                    2 => [anchor_nox_width, anchor_box_height]
            Public instance attributes:
                model: the Darknet Keras model
                class_names: a list of the class names for the model
                class_t:  the IOU threshold for non-max suppression
                anchors: the anchor boxes
            """
        
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
            self.class_t = class_t
            self.nms_t = nms_t
            self.anchors = anchors
    
    def process_outputs(self, outputs, image_size):
    """Process the outputs from the YOLO model"""
    boxes = []
    box_confidences = []
    box_class_probs = []

    for idx, output in enumerate(outputs):
        grid_height, grid_width, anchor_boxes, _ = output.shape
        
        # Process boxes
        box_xy = 1 / (1 + np.exp(-output[..., :2]))
        box_wh = np.exp(output[..., 2:4]) * self.anchors[idx]
        
        # Create grid
        grid_x, grid_y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
        grid = np.stack([grid_x, grid_y], axis=-1)
        grid = np.expand_dims(grid, axis=2)
        
        # Get coordinates relative to grid
        box_xy = box_xy + grid
        box_xy = box_xy / np.array([grid_width, grid_height])
        
        # Scale width and height relative to input size
        box_wh = box_wh / np.array([self.model.input.shape[1], self.model.input.shape[2]])
        
        # Scale to image size
        box_xy = box_xy * np.array([image_size[1], image_size[0]])
        box_wh = box_wh * np.array([image_size[1], image_size[0]])

        # Transform to corner coordinates
        box_mins = box_xy - (box_wh / 2)
        box_maxs = box_xy + (box_wh / 2)
        box = np.concatenate((box_mins, box_maxs), axis=-1)
        
        boxes.append(box)
        
        # Process confidences and class probabilities
        box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
        box_confidences.append(box_confidence)
        
        box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
        box_class_probs.append(box_class_prob)

    return boxes, box_confidences, box_class_probs
