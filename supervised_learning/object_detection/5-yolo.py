#!/usr/bin/env python3
"""module for process_outputs and yoloooo"""
import numpy as np
from tensorflow import keras
import os
import cv2


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm to perform object
    detection"""
    def __init__(self, model_path, classes_path, class_t,
                 nms_t, anchors):
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """
        Load images from folder
            folder_path: path to folder hoding all images to load
        Returns:
            tuple of (images, image_paths):
                images: list of loaded images as numpy.ndarrays
                image_paths: list of paths to individual images
        """
        loaded_images = []
        image_file_paths = []

        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_image_path = os.path.join(folder_path, image_file)
                loaded_image = cv2.imread(full_image_path)
                if loaded_image is not None:
                    loaded_images.append(loaded_image)
                    image_file_paths.append(full_image_path)

        return (loaded_images, image_file_paths)

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO model
            images: list of images as numpy..ndarrays
        Returns:
            tuple of (pimages, image_shapes):
                pimages: numpy.ndarray shape (ni, 2) containing original
                    heights and widths of images
                    2 => (image_height, image_width)
        """
        model_width = self.model.input.shape[1]
        model_height = self.model.input.shape[2]

        processed_images = []
        original_dimensions = []

        for raw_image in images:
            original_dimensions.append([raw_image.shape[0], raw_image.shape[1]])


            resized_image = cv2.resize(raw_image,
                                (model_width, model_height),
                                interpolation=cv2.INTER_CUBIC)

            
            normalized_image = np.round(resized_image / 255.0, decimals=8)

            processed_images.append(normalized_image)

        processed_images = np.array(processed_images)
        original_dimensions = np.array(original_dimensions)

        return (processed_images, original_dimensions)

    def process_outputs(self, outputs, image_size):
        """Process YOLO model outputs for object detection"""
        detected_boxes = []
        detection_confidences = []
        class_probabilities = []
        image_height, image_width = image_size

        for output_idx, current_output in enumerate(outputs):
            grid_height, grid_width, num_anchors, _ = current_output.shape

            # Generate coordinate grids
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Extract box parameters
            box_x_offset = current_output[..., 0]
            box_y_offset = current_output[..., 1]
            box_width_raw = current_output[..., 2]
            box_height_raw = current_output[..., 3]

            box_center_x = (1 / (1 + np.exp(-box_x_offset)) + grid_x) / grid_width
            box_center_y = (1 / (1 + np.exp(-box_y_offset)) + grid_y) / grid_height

            anchor_widths = self.anchors[output_idx, :, 0]
            anchor_heights = self.anchors[output_idx, :, 1]
            
            box_width = anchor_widths * np.exp(box_width_raw) / self.model.input.shape[1]
            box_height = anchor_heights * np.exp(box_height_raw) / self.model.input.shape[2]

            # Calculate corner coordinates
            box_x1 = (box_center_x - box_width / 2) * image_width
            box_y1 = (box_center_y - box_height / 2) * image_height
            box_x2 = (box_center_x + box_width / 2) * image_width
            box_y2 = (box_center_y + box_height / 2) * image_height

            # Update output with corner coordinates
            current_output[..., 0] = box_x1
            current_output[..., 1] = box_y1
            current_output[..., 2] = box_x2
            current_output[..., 3] = box_y2

            # Process confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-current_output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-current_output[..., 5:]))

            detected_boxes.append(current_output[..., :4])
            detection_confidences.append(box_confidence)
            class_probabilities.append(class_probs)

        return detected_boxes, detection_confidences, class_probabilities


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
            # Get dimensions for reshaping
            grid_h, grid_w, num_anchors, _ = boxes[i].shape
            
            # Reshape arrays
            flat_confidences = confidences[i].reshape(-1, 1)
            flat_class_probs = class_probs[i].reshape(-1, class_probs[i].shape[-1])

            # Calculate scores
            combined_scores = flat_confidences * flat_class_probs
            max_class_scores = np.max(combined_scores, axis=1)
            class_predictions = np.argmax(combined_scores, axis=1)

            # Filter based on threshold
            threshold_mask = max_class_scores >= self.class_t

            if len(threshold_mask) > 0:
                flat_boxes = boxes[i].reshape(-1, 4)
                filtered_boxes.append(flat_boxes[threshold_mask])
                filtered_classes.append(class_predictions[threshold_mask])
                filtered_scores.append(max_class_scores[threshold_mask])

        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            filtered_classes = np.concatenate(filtered_classes, axis=0)
            filtered_scores = np.concatenate(filtered_scores, axis=0)

        return filtered_boxes, filtered_classes, filtered_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression to remove overlapping boxes
        """
        final_boxes = []
        final_classes = []
        final_scores = []

        # Process each unique class
        unique_classes = np.unique(box_classes)

        for class_id in unique_classes:
            class_mask = box_classes == class_id
            class_boxes = filtered_boxes[class_mask]
            class_scores = box_scores[class_mask]

            # Sort by score in descending order
            score_order = np.argsort(-class_scores)
            sorted_boxes = class_boxes[score_order]
            sorted_scores = class_scores[score_order]

            while len(sorted_boxes) > 0:
                # Keep box with highest score
                final_boxes.append(sorted_boxes[0])
                final_classes.append(class_id)
                final_scores.append(sorted_scores[0])

                if len(sorted_boxes) == 1:
                    break

                # Calculate IoU with remaining boxes
                current_box = sorted_boxes[0]
                intersect_x1 = np.maximum(current_box[0], sorted_boxes[1:, 0])
                intersect_y1 = np.maximum(current_box[1], sorted_boxes[1:, 1])
                intersect_x2 = np.minimum(current_box[2], sorted_boxes[1:, 2])
                intersect_y2 = np.minimum(current_box[3], sorted_boxes[1:, 3])

                # Calculate areas
                intersect_area = np.maximum(0, intersect_x2 - intersect_x1) * \
                               np.maximum(0, intersect_y2 - intersect_y1)
                current_box_area = (current_box[2] - current_box[0]) * \
                                 (current_box[3] - current_box[1])
                remaining_box_areas = (sorted_boxes[1:, 2] - sorted_boxes[1:, 0]) * \
                                    (sorted_boxes[1:, 3] - sorted_boxes[1:, 1])
                union_area = current_box_area + remaining_box_areas - intersect_area

                # Calculate IoU
                iou_scores = intersect_area / union_area

                # Keep boxes with IoU less than threshold
                keep_mask = iou_scores < self.nms_t
                sorted_boxes = sorted_boxes[keep_mask + 1]
                sorted_scores = sorted_scores[keep_mask + 1]

        # Convert to numpy arrays
        if final_boxes:
            final_boxes = np.array(final_boxes)
            final_classes = np.array(final_classes)
            final_scores = np.array(final_scores)
        else:
            final_boxes = np.array([])
            final_classes = np.array([])
            final_scores = np.array([])

        return final_boxes, final_classes, final_scores
