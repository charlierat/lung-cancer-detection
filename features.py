import numpy as np
import scipy.ndimage as ndimage
from skimage import measure


def extract_area_feature(segmented_image):
    """
    Extract area-based features from segmented lung regions
    
    Parameters:
        segmented_image (binary image): segmented regions of scan
        
    Returns:
        Dictionary of area-based features and filtered image
    """
    # Label connected components
    labeled_image, num_features = ndimage.label(segmented_image)
    
    # Measure properties of labeled regions
    region_props = measure.regionprops(labeled_image)
    
    # Filter out small regions
    min_area_threshold = 10  # pixels
    filtered_image = np.zeros_like(segmented_image)
    
    valid_regions = []
    for region in region_props:
        if region.area >= min_area_threshold:
            valid_regions.append(region)
            # Reconstruct binary filtered image
            filtered_image[labeled_image == region.label] = 1
    
    return {
        'num_regions': len(valid_regions),
        'region_areas': [region.area for region in valid_regions],
        'filtered_image': filtered_image
    }

def extract_calcification_features(ct_image, segmented_regions):
    """
    Extract calcification patterns from CT image
    
    Parameters:
        ct_image (ndarray): Original CT image with HU values
        segmented_regions: Binary mask of regions of interest
        
    Returns:
        Dictionary of calcification features
    """
    # Define HU thresholds for calcification
    calcification_lower_threshold = 100  # HU value typical for calcification
    calcification_upper_threshold = 1100 # Hu value upperbound to eliminate unwanted regions
    
    calcification_features = {}
    labeled_regions, num_regions = ndimage.label(segmented_regions)
    
    for region_idx in range(1, num_regions + 1):
        # Create mask for current region
        region_mask = labeled_regions == region_idx
        
        # Extract intensity values within the region
        region_intensities = ct_image[region_mask]
        
        # Calculate calcification metrics
        calc_pixels = np.logical_and(
            region_intensities > calcification_lower_threshold,
            region_intensities < calcification_upper_threshold
        )
        calc_percentage = np.sum(calc_pixels) / len(region_intensities)
        
        # Determine calcification pattern 
        if calc_percentage > 0.8:
            pattern = "diffuse"
        elif calc_percentage > 0.5:
            # Check if calcification is central
            region_props = measure.regionprops((region_mask).astype(int))
            centroid = region_props[0].centroid
            
            # Create distance map from centroid
            y, x = np.ogrid[:region_mask.shape[0], :region_mask.shape[1]]
            distance_from_center = np.sqrt((x - centroid[1])**2 + (y - centroid[0])**2)
            
            # If calcification is primarily in center
            if np.mean(region_intensities[distance_from_center < distance_from_center.max()/3] > calcification_lower_threshold):
                pattern = "central"
            else:
                pattern = "laminated"
        else:
            pattern = "scattered"
            
        calcification_features[f'region_{region_idx}'] = {
            'calc_percentage': calc_percentage,
            'pattern': pattern,
            'mean_intensity': np.mean(region_intensities),
            'max_intensity': np.max(region_intensities)
        }
    
    return calcification_features

def extract_shape_features(segmented_regions):
    """
    Extract shape-based features from segmented regions
    
    Parameters:
        segmented_regions: Binary mask of regions of interest
        
    Returns:
        Dictionary of shape features
    """
    labeled_regions, num_regions = ndimage.label(segmented_regions)
    region_props = measure.regionprops(labeled_regions)
    
    shape_features = {}
    
    for idx, region in enumerate(region_props):
        # Calculate 3D ratio
        width = region.bbox[3] - region.bbox[1]
        height = region.bbox[2] - region.bbox[0]
        three_d_ratio = width / height if height > 0 else 0
        
        # Calculate shape metrics
        perimeter = region.perimeter
        area = region.area
        
        # Circularity/roundness (1 for perfect circle, less for irregular shapes)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        # Solidity (area / convex hull area)
        solidity = region.solidity
        
        # Eccentricity (0 for circle, 1 for line)
        eccentricity = region.eccentricity
        
        # Determine if shape is polygonal (many corners/edges)
        is_polygonal = circularity < 0.8 and solidity > 0.9
        
        shape_features[f'region_{idx+1}'] = {
            'three_d_ratio': three_d_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'is_polygonal': is_polygonal,
            'perimeter': perimeter,
            'area': area
        }
    
    return shape_features

def extract_size_features(segmented_regions, pixel_spacing_mm):
    """
    Extract size-based features from segmented regions
    
    Parameters:
        segmented_regions: Binary mask of regions of interest
        pixel_spacing_mm: Pixel spacing in millimeters (from DICOM metadata)
        
    Returns:
        Dictionary of size features and malignancy risk assessment
    """
    labeled_regions, num_regions = ndimage.label(segmented_regions)
    region_props = measure.regionprops(labeled_regions)
    
    size_features = {}
    
    for idx, region in enumerate(region_props):
        # Calculate physical dimensions
        width_pixels = region.bbox[3] - region.bbox[1]
        height_pixels = region.bbox[2] - region.bbox[0]
        
        # Convert to mm
        width_mm = width_pixels * pixel_spacing_mm
        height_mm = height_pixels * pixel_spacing_mm
        
        # Maximum diameter in mm
        max_diameter_mm = max(width_mm, height_mm)
        
        # Equivalent diameter (diameter of circle with same area)
        equiv_diameter_mm = region.equivalent_diameter * pixel_spacing_mm
        
        # Categorize based on size (SPN definition)
        if max_diameter_mm < 30:
            category = "nodule"
            if max_diameter_mm < 4:
                malignancy_risk = "very low"
            elif max_diameter_mm < 8:
                malignancy_risk = "low"
            elif max_diameter_mm < 20:
                malignancy_risk = "moderate"
            else:
                malignancy_risk = "high"
        else:
            category = "mass"
            malignancy_risk = "very high"
        
        size_features[f'region_{idx+1}'] = {
            'max_diameter_mm': max_diameter_mm,
            'equiv_diameter_mm': equiv_diameter_mm,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'category': category,
            'malignancy_risk_by_size': malignancy_risk
        }
    
    return size_features
