import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means
import pydicom

"""
CT scan preprocessing filters with proper HU handling and DICOM saving
- Tests Gaussian filter
- Tests Median filter
- Tests Non-local means filter
- Tests Contrast enhancement
- Tests combined application of all filters
- Saves results as DICOM files
"""

def apply_gaussian_filter(image, sigma):
    """Gaussian filter for noise reduction
       Removes background high-frequency noise

    Parameters:
        image (ndarray): Original CT Scan
        sigma (float): Sigma value for gaussian calculation

    Returns:
        Image: CT scan with guassian blur applied
    """
    return gaussian_filter(image, sigma)

# Median filter for salt-and-pepper noise
def apply_median_filter(image, size):
    """Median filter for noise reduction
       Removes "salt-and-pepper" noise in the image

    Parameters:
        image (ndarray): CT scan with guassian blur applied to it
        size (int): Size of the radius for the filter

    Returns:
        Image: CT scan with guassian blur and median filter
    """
    return median_filter(image, size)

# Non-local means filter for preserving edges while reducing noise
def apply_nlm_filter(image):
    """Non-Local Means filter for reducing noise
       Preserves edges and removes bigger noise missed by gaussian and median

    Parameters:
        image (ndarray): CT scan with gaussian blur and median filter applied

    Returns:
        filtered: CT scan with gaussian blur, median filter, and non-local means filter
    """
    #Normalize Image
    min_val = np.min(image)
    max_val = np.max(image)
    image_normalized = (image - min_val) / (max_val - min_val)
    
    # Apply NLM filter
    filtered = denoise_nl_means(image_normalized, patch_size=5, patch_distance=6, h=0.07, fast_mode=True)
    
    # Convert back to original range
    filtered = filtered * (max_val - min_val) + min_val
    
    return filtered


def apply_all_filters(image):
    """Applies all filters to the orignal CT scan
       1. Gaussian Blur
       2. Median Filter
       3. Non-Local Means Filter

    Parameters:
        image (ndarray): Original CT Scan

    Returns:
        filtered: CT scan with all filteres applied to it
    """
    # Step 1: Gaussian blur
    filtered = apply_gaussian_filter(image, sigma=1)
    
    # Step 2: Median filter
    filtered = apply_median_filter(filtered, size=3) 
    
    # Step 3: Non-local means filter
    filtered = apply_nlm_filter(filtered)
    
    return filtered


def save_as_dicom(processed_image, original_dicom, output_path, description_suffix=''):
    """Save a processed image as a DICOM file, preserving the original metadata

    Parameters:
        processed_image (ndarray): The processed image array
        original_dicom (pydicom.dataset.FileDataset): Original DICOM dataset object
        output_path (str): Path where the new DICOM file will be saved
        description_suffix (str, optional): Text to append to SeriesDescription. Defaults to ''
    """
    # Make a copy to avoid modifying the original
    ds = original_dicom.copy()
    
    # Convert back from HU to original pixel values if needed
    if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
        # Likely in HU space, convert to pixel values
        pixel_array = (processed_image - ds.RescaleIntercept) / ds.RescaleSlope
    else:
        pixel_array = processed_image
    
    # Ensure the pixel data is in the correct data type
    if ds.pixel_array.dtype != np.float64:
        pixel_array = np.clip(pixel_array, 
                            np.iinfo(ds.pixel_array.dtype).min if np.issubdtype(ds.pixel_array.dtype, np.integer) else -np.inf, 
                            np.iinfo(ds.pixel_array.dtype).max if np.issubdtype(ds.pixel_array.dtype, np.integer) else np.inf)
        pixel_array = pixel_array.astype(ds.pixel_array.dtype)
    
    # Update the pixel data
    ds.PixelData = pixel_array.tobytes()
    
    # Make the DICOM file unique by changing some UIDs
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    # Add processing description to the image
    if hasattr(ds, 'SeriesDescription'):
        ds.SeriesDescription = ds.SeriesDescription + description_suffix
    else:
        ds.SeriesDescription = 'Processed' + description_suffix
        
    if hasattr(ds, 'ImageComments'):
        ds.ImageComments = f"Processed with custom filters: {ds.ImageComments}"
    else:
        ds.ImageComments = "Processed with custom filters"
    
    # Save the new DICOM file
    ds.save_as(output_path)
    print(f"Saved processed DICOM to: {output_path}")


def apply_hu_windowing(image, window_center, window_width):
    """
    Apply windowing to CT scan using Hounsfield Units
    
    Parameters:
        image (ndarray): Original CT Scan
        window_center (integer): center of window
        window_width (integer): width of window
        
    Returns:
        windowed_image = new image in correct window
    """
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    
    windowed_image = np.clip(image, min_value, max_value)
    
    # Normalize to [0, 1] for display
    windowed_image = (windowed_image - min_value) / window_width
    return windowed_image

def load_ct_scan(file_path):
    """
    Load a CT scan from DICOM file
    
    Parameters:
        file_path (str): Path to CT scan DICOM file
        
    Returns:
        image, ds: Image converted to HU and DICOM dataset
    """
    # Load DICOM file
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array
    
    # Convert to Hounsfield Units
    if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept
    return image, ds



if __name__ == "__main__":
    """
    Test results for preprocessing.py
    Saves new filtered scan for visualization
    """
    # Path to your input DICOM file
    input_dicom_path = "test.dcm"
    
    # Load the DICOM file
    ds = pydicom.dcmread(input_dicom_path)
    
    # Get image data
    image = ds.pixel_array
    
    # If HU conversion is needed (for CT scans)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept
    
    # Apply filters
    processed_image = apply_all_filters(image)
    
    # Save the processed image as a new DICOM file
    output_dicom_path = "output.dcm"
    save_as_dicom(processed_image, ds, output_dicom_path)