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

            normalized_image = resized_image / 255.0

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
        Filter boxes based on class and box confidence values
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_per_class = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.extend(boxes[i][mask])
            box_classes.extend(box_class[mask])
            box_scores.extend(box_score[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

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

            while len(class_boxes) > 0:
                # Keep box with highest score
                max_idx = np.argmax(class_scores)
                final_boxes.append(class_boxes[max_idx])
                final_classes.append(class_id)
                final_scores.append(class_scores[max_idx])

                if len(class_boxes) == 1:
                    break

                # Remove the box with highest score
                class_boxes = np.delete(class_boxes, max_idx, axis=0)
                class_scores = np.delete(class_scores, max_idx)

                # Calculate IoU with remaining boxes
                ious = self.intersection_over_union(final_boxes[-1], class_boxes)
                iou_mask = ious < self.nms_t

                # Keep boxes with IoU less than threshold
                class_boxes = class_boxes[iou_mask]
                class_scores = class_scores[iou_mask]

        final_boxes = np.array(final_boxes)
        final_classes = np.array(final_classes)
        final_scores = np.array(final_scores)

        return final_boxes, final_classes, final_scores

    def intersection_over_union(self, box1, boxes):
        """Calculate intersection over union between box1 and boxes"""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with bounding boxes, class names and scores"""
        draw_img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = f"{box_scores[i]:.2f}"
            text = f"{class_name} {score}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness)
            
            text_x = x1
            text_y = max(y1 - 5, text_height)

            cv2.putText(draw_img, text, (text_x, text_y), font, font_scale,
                       (0, 0, 255), font_thickness, cv2.LINE_AA)

        cv2.imshow(file_name, draw_img)

        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            output_path = os.path.join('detections', file_name)
            cv2.imwrite(output_path, draw_img)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Perform object detection on all images in specified folder
        """
        # Load images
        images, image_paths = self.load_images(folder_path)
        
        if len(images) == 0:
            return [], []
            
        # Preprocess images
        processed_images, image_shapes = self.preprocess_images(images)
        
        # Get predictions
        outputs = self.model.predict(processed_images)
        
        # Process each image
        predictions = []
        
        for i, image in enumerate(images):
            # Extract outputs for current image
            image_outputs = [output[i:i+1] if len(output.shape) == 4 
                           else output[i] for output in outputs]
            
            # Process outputs
            boxes, confidences, class_probs = self.process_outputs(
                image_outputs, image_shapes[i])
            
            # Filter boxes
            filtered_boxes, filtered_classes, filtered_scores = self.filter_boxes(
                boxes, confidences, class_probs)
            
            # Apply non-max suppression
            final_boxes, final_classes, final_scores = self.non_max_suppression(
                filtered_boxes, filtered_classes, filtered_scores)
            
            # Add to predictions
            predictions.append((final_boxes, final_classes, final_scores))
            
            image_filename = os.path.basename(image_paths[i])
            self.show_boxes(
                image, final_boxes, final_classes, final_scores, image_filename)
        
        return predictions, image_paths
