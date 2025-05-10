import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from skimage import feature
from preprocessing import load_ct_scan, save_as_dicom, apply_hu_windowing
    

def apply_sobel_edge_detection(image, threshold):
    """
    Apply Sobel edge detection to a CT image
    
    Parameters:
        image (ndarray): Input CT image
        threshold (float): Threshold for edge detection (0-1)
        
    Returns:
        edges (ndarray): Edge map of the image
    """
    # Normalize image for edge detection
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    
    # Apply Sobel filter
    sobel_x = ndimage.sobel(normalized, 0)
    sobel_y = ndimage.sobel(normalized, 1)
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Normalize magnitude
    magnitude = magnitude / np.max(magnitude)
    
    # Threshold to get binary edges
    edges = magnitude > threshold
    
    return edges.astype(float)

def apply_canny_edge_detection(image, sigma, low_threshold, high_threshold):
    """
    Apply Canny edge detection to a CT image
    
    Parameters:
        image (ndarray): Input CT image
        sigma (float): Standard deviation for Gaussian filter
        low_threshold (float): Lower threshold for hysteresis thresholding
        high_threshold (float): Higher threshold for hysteresis thresholding
        
    Returns:
        edges (ndarray): Binary edge map
    """
    # Normalize image for edge detection
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    
    # Apply Canny edge detection
    edges = feature.canny(
        normalized,
        sigma,
        low_threshold,
        high_threshold
    )
    
    return edges.astype(float)

def apply_sobel_then_canny(image, sobel_threshold = 0.08, canny_sigma=1.2,
                           canny_low_threshold=0.05, canny_high_threshold=0.15):
    """
    Optimized Sobel and Canny edge detection
    
    Parameters:
        image (ndarray): Input CT image
        sobel_threshold (float): Threshold value for sobel edge detection
        canny_sigma (float): Sigma value for Guassian filter
        canny_low_threshold (float): Lower threshold for hysteresis thresholding
        canny_high_threshold (float): Higher threshold for hysteresis thresholding
        
    Returns:
        canny_edges (ndarray): Binary edge map
    """
    # First apply Sobel edge detection
    sobel_edges = apply_sobel_edge_detection(image, sobel_threshold)
    
    # Then apply Canny edge detection on the Sobel result
    canny_edges = apply_canny_edge_detection(
        sobel_edges, 
        canny_sigma,
        canny_low_threshold,
        canny_high_threshold
    )
    
    return canny_edges

def detect_edges_in_ct(image_path, window_center, window_width, output_dir='edge_detection_results'):
    """
    Detect edges in a CT scan using multiple methods and save results
    
    Parameters:
        image_path (filepath): file path to ct scan
        window_center (integer): Center of window for visualization
        window_width (integer): Width of window for visualization
        output_dir (str): File output save as
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image and DICOM dataset
    img, dicom_dataset = load_ct_scan(image_path)
    
    # Store original image for later
    original_img = img.copy()
    
    # Apply preprocessing
    preprocessed_img = img
    
    # Apply edge detection methods
    sobel_edges = apply_sobel_edge_detection(preprocessed_img, threshold=0.1)
    canny_edges = apply_canny_edge_detection(preprocessed_img, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    sobel_canny_edges = apply_sobel_then_canny(preprocessed_img)
    
    # Save edge detection results as DICOM for more precise results
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    save_as_dicom(sobel_edges, dicom_dataset,
                os.path.join(output_dir, f"{base_filename}_sobel_edges.dcm"),
                description_suffix=' - Sobel Edge Detection')
    
    save_as_dicom(canny_edges, dicom_dataset,
                os.path.join(output_dir, f"{base_filename}_canny_edges.dcm"),
                description_suffix=' - Canny Edge Detection')
    
    save_as_dicom(sobel_canny_edges, dicom_dataset,
                os.path.join(output_dir, f"{base_filename}_sobel_canny_edges.dcm"),
                description_suffix=' - Sobel+Canny Edge Detection')
    
    # Apply windowing for display
    display_original = apply_hu_windowing(original_img, window_center, window_width)
    
    # Display the results as PNG for easier viewing
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(display_original, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    # Sobel edges
    axes[0, 1].imshow(sobel_edges, cmap='gray')
    axes[0, 1].set_title('Sobel Edge Detection')
    axes[0, 1].axis('off')
    
    # Canny edges
    axes[1, 0].imshow(canny_edges, cmap='gray')
    axes[1, 0].set_title('Canny Edge Detection')
    axes[1, 0].axis('off')
    
    # Sobel followed by Canny
    axes[1, 1].imshow(sobel_canny_edges, cmap='gray')
    axes[1, 1].set_title('Sobel + Canny Edge Detection')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_edge_detection_comparison.png"), dpi=300)
    
    print(f"All edge detection results saved to {output_dir}/")

if __name__ == "__main__":
    """
    Test results for edges.py
    Saves edge results for visualization
    """
    #Image path to be tested --> filtered scan
    image_path = "output.dcm"
    window_center = -700
    window_width = 1400
    
    # Detect edges using multiple methods
    detect_edges_in_ct(image_path, window_center, window_width)
