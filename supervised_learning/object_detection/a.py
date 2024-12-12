#!/usr/bin/env python3
"""module for YOLO object detection"""
import cv2
import numpy as np
import os
from tensorflow import keras


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo class"""
        self.model = keras.models.load_model(model_path)
        self.class_names = [line.strip() for line in open(classes_path)]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """
        Load images from folder
        Args:
            folder_path: path to folder holding all images to load
        Returns:
            tuple of (images, image_paths):
                images: list of loaded images as numpy.ndarrays
                image_paths: list of paths to individual images
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO model
        Args:
            images: list of images as numpy.ndarrays
        Returns:
            tuple of (pimages, image_shapes):
                pimages: numpy.ndarray shape (ni, input_h, input_w, 3) containing
                    preprocessed images
                    ni: number of images
                    input_h: input height for Darknet model
                    input_w: input width for Darknet model
                    3: number of color channels
                image_shapes: numpy.ndarray of shape (ni, 2) containing original
                    heights and widths of images
                    2 => (image_height, image_width)
        """
        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value
        
        pimages = []
        image_shapes = []

        for image in images:
            # Store original image shape
            image_shapes.append([image.shape[0], image.shape[1]])
            
            # Resize image with inter-cubic interpolation
            resized = cv2.resize(image,
                               (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            
            # Rescale pixel values to [0, 1]
            preprocessed = resized / 255.0
            
            pimages.append(preprocessed)

        # Convert lists to numpy arrays with proper shapes
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)
