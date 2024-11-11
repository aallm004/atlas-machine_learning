#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cv2
from matplotlib import pyplot as plt


def resize_if_large(image, max_size=100):
    """Resize image if it's too large while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image


def create_mask_from_black(image, threshold=10):
    """Create mask identifying black pixels in the image"""
    #convert to grayscale if image is RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        mask = binary > 128

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        plt.show()
    else:
        mask = image < threshold
    
    return mask.astype(bool)

def get_neighborhood_matrix(mask):
    """Create sparse matrix for biharmonic equation"""
    h, w = mask.shape
    n = h * w

    #create diagonal indices
    diag = np.arange(n)

    #create neighborhood indices
    right = np.roll(diag, -1).reshape(h, w)
    left = np.roll(diag, 1).reshape(h, w)
    up = np.roll(diag, -w).reshape(h, w)
    down = np.roll(diag, w).reshape(h, w)

    #fix boundary indices
    right[:, -1] = diag.reshape(h, w)[:, -1]
    left[:, 0] = diag.reshape(h, w)[:, 0]
    up[0, :] = diag.reshape(h, w)[0, :]
    down[-1, :] = diag.reshape(h, w)[-1, :]

    #Flatten indices
    right = right.flatten()
    left = left.flatten()
    up = up.flatten()
    down = down.flatten()

    #Create sparse matrix
    data = np.ones(n * 5)
    row_ind = np.concatenate([diag, diag, diag, diag, diag])
    col_ind = np.concatenate([diag, right, left, up, down])

    A = sparse.coo_matrix((data, (row_ind, col_ind)), shape = (n, n))
    A = A.tocsr()

    #set masked pixels to identity
    mask_flat = mask.flatten()
    chunk_size = 500
    for i in range(0, len(mask_flat), chunk_size):
        end = min(i + chunk_size, len(mask_flat))
        chunk_mask = mask_flat[i:end]
        if np.any(chunk_mask):
            A[i:end, :] = 0
            A[i:end, i:end] = 1

    return A

def smooth_result(result, mask, original):
    """Apply photo-aware smoothing"""
    # Convert to float32 for processing
    result = result.astype(np.float32)
    
    # Create a narrow band around the inpainted region
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    band = dilated - mask.astype(np.uint8)
    
    # Blend the results in the band area
    blend_mask = cv2.GaussianBlur(band.astype(np.float32), (5,5), 0)
    blend_mask = blend_mask[:,:,np.newaxis] if len(result.shape) == 3 else blend_mask
    
    # Apply guided filter using the original image as guide
    for i in range(3):
        result[:,:,i] = cv2.guidedFilter(original[:,:,i], result[:,:,i], 
                                       radius=4, eps=0.1)
    
    # Blend with original image at the boundaries
    result = result * (1 - blend_mask) + original.astype(np.float32) * blend_mask
    
    return result.astype(np.uint8)


def biharmonic_inpaint(image, mask):
    """Perform biharmonic inpainting on masked regions"""
    # Handle RGB images
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=float)
        for channel in range(3):
            result[:, :, channel] = biharmonic_inpaint(image[:, :, channel], mask)
        return result
    
    # Convert to float
    img_float = image.astype(float)

    # Create system matrix
    A = get_neighborhood_matrix(mask)

    # Solve system
    b = img_float.flatten()
    x = spsolve(A, b)

    return x.reshape(image.shape)

def remove_black_line(image_path, output_path=None):
    """Remove black line from image using biharmonic inpaiting"""
    
    print('Reading image...')
    #read image
    image = cv2.imread('image_damaged.jpg')
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    #Resize if necessary
    image = resize_if_large(image)
    print("Processing")
    #Create mask of black pixels
    mask = create_mask_from_black(image)

    #Perform inpainting
    result = biharmonic_inpaint(image, mask)

    #Clip values and convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save result if output path is provided
    if output_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)

    return result

def show_results(original, result):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(result)
    plt.title('Inpainted Image')
    plt.axis('off')
    plt.show()

    #Add image and file for new image
    result = remove_black_line('image_damaged.jpg', 'image_repaired.jpg')

    #Display the results
    original = cv2.imread('image_damaged.jpg')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    show_results(original, result)

#Main execution block
if __name__ == "__main__":
    input_image = "image_damaged.jpg"
    output_image = "image_repaired.jpg"
    
    # Process the image
    result = remove_black_line(input_image, output_image)
    
    if result is not None:
        # Display the results
        original = cv2.imread(input_image)
        if original is not None:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            show_results(original, result)
        else:
            print(f"Error: Could not read original image: {input_image}")
