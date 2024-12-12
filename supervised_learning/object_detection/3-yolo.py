#!/usr/bin/env python3
"""module for process_outputs and yoloooo"""
import numpy as np
from tensorflow import keras


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm to perform object
    detection"""
    def __init__(self, model_path, classes_path, class_t,
                 nms_t, anchors):
        self.model = keras.models.load_model(model_path)
        self.class_names = [line.strip() for line in open(classes_path)]
        self.class_t = class_t
        self.nms_t = nms_t
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively
            box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing the processed
            box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed box
            class probabilities for each output, respectively

        Returns: a tuple of (filtered_boxes, box_classes, box_scores)
            filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes:
                box_classes: a numpy.ndarray of shape (?,) containing the class
                number that each box in filtered_boxes predicts, respectively
                box_scores: a numpy.ndarray of shape (?) containing the box
                scores for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box = boxes[i]
            box_confidence = box_confidences[i]
            box_class_prob = box_class_probs[i]
            box_scores_combined = box_confidence * box_class_prob

            box_class = np.argmax(box_scores_combined, axis=-1)
            box_score = np.max(box_scores_combined, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Function to apply non_max suppression to filter ou t overlapping boxes
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the filtered bounding boxes
            box_classes: a numpy.ndarray of shape (?,) containing the class number that each  box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores for each box in filtered_boxes, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = (box_classes == cls)
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]
            order = cls_scores.argsort()[::-1]

            while len(order) > 0:
                i = order[0]
                box_predictions.append(cls_boxes[i])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[i])

                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h
                
                box_area = (cls_boxes[i, 2] - cls_boxes[i, 0]) * \
                    (cls_boxes[i, 3] - cls_boxes[i, 1])
                other_areas = (cls_boxes[order[:1], 2] - cls_boxes[order[1:], 0]) * \
                            (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
                union = box_area + other_areas - inter

                iou = inter / union

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

        if box_predictions:
            box_predictions = np.array(box_predictions)
            predicted_box_classes = np.array(predicted_box_classes)
            predicted_box_scores = np.array(predicted_box_scores)

        else:
            box_predictions = np.array([])
            predicted_box_classes = np.array(predicted_box_classes)
            predicted_box_scores - np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores
