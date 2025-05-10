import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, binary_erosion
from scipy import ndimage
import sys

# Import functions from files
from preprocessing import (
    load_ct_scan, 
    apply_all_filters, 
    apply_hu_windowing,
)
from edges import (
    apply_sobel_then_canny
)
from features import (
    extract_area_feature,
    extract_calcification_features,
    extract_shape_features,
    extract_size_features
)

def segment_lung_regions(ct_image):
    """
    A robust combined approach for lung segmentation that handles both normal lung tissue
    and potential tumors/dense regions while excluding background
    
    Parameters:
        ct_image: CT image in Hounsfield Units
        
    Returns:
        Binary mask of segmented lung regions
    """
    
    # STEP 1: Create a body mask to separate patient from background
    # Body tissues are generally > -500 HU
    body_mask = ct_image > -500
    # Apply morphological operations to clean up the body mask
    body_mask = ndimage.binary_closing(body_mask, structure=np.ones((10, 10)))
    body_mask = ndimage.binary_fill_holes(body_mask)
    
    # Find largest connected component (the body)
    labeled_body, num_body = ndimage.label(body_mask)
    if num_body > 0:
        sizes = np.bincount(labeled_body.flatten())
        sizes[0] = 0  # Ignore background
        largest_body = np.argmax(sizes)
        # Keep only the largest component
        body_mask = labeled_body == largest_body
    
    # STEP 2: Standard lung tissue segmentation
    # Typical lung tissue is between -950 and -500 HU
    standard_mask = np.logical_and(ct_image > -950, ct_image < -500)
    
    # STEP 3: Wider threshold to include denser tissues
    wider_mask = np.logical_and(ct_image > -950, ct_image < -200)
    
    # STEP 4: Restrict lung segmentation to be inside the body only
    standard_mask = np.logical_and(standard_mask, body_mask)
    wider_mask = np.logical_and(wider_mask, body_mask)
    
    # Clean up both masks
    standard_mask = ndimage.binary_opening(standard_mask, structure=np.ones((3, 3)))
    wider_mask = ndimage.binary_opening(wider_mask, structure=np.ones((3, 3)))
    
    # Find the largest connected components in the standard mask (should be lungs)
    labeled, num_features = ndimage.label(standard_mask)
    
    if num_features > 0:
        sizes = np.bincount(labeled.flatten())
        sizes[0] = 0  # Ignore background
        
        # Get the largest few regions
        largest_regions = np.argsort(sizes)[-5:] if num_features >= 5 else np.argsort(sizes)[-(num_features+1):]
        largest_regions = [x for x in largest_regions if x > 0]  # Ensure we don't include background
        
        # Create a new mask with only these regions
        core_lung_mask = np.zeros_like(standard_mask)
        for label in largest_regions:
            core_lung_mask = np.logical_or(core_lung_mask, labeled == label)
        
        # Dilate the core lung mask to include surrounding areas
        expanded_core = ndimage.binary_dilation(core_lung_mask, structure=np.ones((5, 5)))
        
        # Combine with the wider mask but only in the vicinity of the core lungs
        combined_mask = np.logical_and(wider_mask, expanded_core)
        
        # Make sure we're still inside the body
        combined_mask = np.logical_and(combined_mask, body_mask)
        
        # Fill holes (this is crucial for including tumors within lung boundaries)
        lung_mask = ndimage.binary_fill_holes(combined_mask)
        
        # Final cleaning - ensure we're still within the body
        lung_mask = np.logical_and(lung_mask, body_mask)
        
        return lung_mask
    else:
        # Fallback if standard approach fails - still ensure we're within the body
        return np.logical_and(wider_mask, body_mask)

def segment_potential_nodules(ct_image, lung_mask):
    """
    Improved nodule segmentation with better HU range selection
    
    Parameters:
        ct_image: CT image in Hounsfield Units
        lung_mask: Binary mask of lung regions
        
    Returns:
        Binary mask of potential nodules
    """
    # 1. Solid nodules (typically -100 to 400 HU)
    solid_nodules = np.logical_and(ct_image > -100, ct_image < 400)
    
    # 2. Part-solid/ground glass nodules (typically -750 to -100 HU)
    ground_glass = np.logical_and(ct_image > -750, ct_image < -100)
    
    # 3. Calcified nodules (typically > 200 HU, but not too high)
    calcified = np.logical_and(ct_image > 200, ct_image < 1000)
    
    # Combine all potential nodule types
    all_nodules = np.logical_or(np.logical_or(solid_nodules, ground_glass), calcified)
    
    # Only consider candidates within the lung regions
    nodule_candidates = np.logical_and(all_nodules, lung_mask)
    
    # Label connected components
    labeled_array, num_features = label(nodule_candidates)
    
    # Remove very small components (likely noise)
    min_size = 10  # Minimum number of pixels for a valid nodule
    
    # Calculate size of each component
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Remove small components
    remove_small = np.isin(labeled_array, np.where(component_sizes < min_size)[0]).reshape(labeled_array.shape)
    nodule_candidates[remove_small] = 0
    
    # Apply morphological operations to clean up
    nodule_candidates = binary_erosion(nodule_candidates, iterations=1)
    nodule_candidates = binary_dilation(nodule_candidates, iterations=2)
    nodule_candidates = binary_erosion(nodule_candidates, iterations=1)
    
    return nodule_candidates

def analyze_ct_scan_with_improved_detection(image_path, output_dir='cancer_analysis_results'):
    """
    Analyze CT scan with improved nodule detection
        
    Parameters:
        image_path: Path to image file
        output_dir: File name to save output as
        
    Returns:
        Results: Cancer probability
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the CT scan
    ct_image, dicom_dataset = load_ct_scan(image_path)
    
    if dicom_dataset is None:
        return {"error": "Input file is not a DICOM file"}
    
    # Get pixel spacing from DICOM metadata
    pixel_spacing_mm = 1.0
    if hasattr(dicom_dataset, 'PixelSpacing') and len(dicom_dataset.PixelSpacing) >= 1:
        pixel_spacing_mm = float(dicom_dataset.PixelSpacing[0])
    
    # Store original image
    original_img = ct_image.copy()
    
    # Apply preprocessing filters
    filtered_img = apply_all_filters(ct_image)
    
    # Segment lung regions using the combined approach
    lung_mask = segment_lung_regions(filtered_img)
    
    # Apply improved nodule detection
    nodule_candidates = segment_potential_nodules(filtered_img, lung_mask)
    
    # Save intermediate results
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Feature extraction
    area_features = extract_area_feature(nodule_candidates)
    filtered_nodules = area_features['filtered_image']
    
    # If no significant nodules found
    if area_features['num_regions'] == 0:
        return {
            "has_cancer": False,
            "confidence": 0.95,
            "message": "No significant nodules found in the scan."
        }
    
    # Extract additional features
    calc_features = extract_calcification_features(filtered_img, filtered_nodules)
    shape_features = extract_shape_features(filtered_nodules)
    size_features = extract_size_features(filtered_nodules, pixel_spacing_mm)
    
    # Analyze features for cancer probability
    cancer_probability = 0.0
    cancer_regions = []
    
    for region_idx in range(1, area_features['num_regions'] + 1):
        region_key = f'region_{region_idx}'
        
        if (region_key not in calc_features or 
            region_key not in shape_features or 
            region_key not in size_features):
            continue
        
        calc_feature = calc_features[region_key]
        shape_feature = shape_features[region_key]
        size_feature = size_features[region_key]
        
        # Calculate region probability
        region_probability = 0.0
        
        # Size-based risk
        if size_feature['malignancy_risk_by_size'] == "very high":
            region_probability += 0.5
        elif size_feature['malignancy_risk_by_size'] == "high":
            region_probability += 0.35  
        elif size_feature['malignancy_risk_by_size'] == "moderate":
            region_probability += 0.2  
        elif size_feature['malignancy_risk_by_size'] == "low":
            region_probability += 0.1 
        else:  # very low
            region_probability += 0.05 

        # Shape-based risk
        if shape_feature['circularity'] < 0.6:  # Very irregular
            region_probability += 0.2
        elif shape_feature['circularity'] < 0.75:  # Moderately irregular
            region_probability += 0.1

        # Solidity risk
        if shape_feature['solidity'] < 0.75:  # Spiculated margins
            region_probability += 0.15

        # Calcification-based risk
        if 'pattern' in calc_feature:
            if calc_feature['pattern'] == 'scattered':
                # Scattered calcification can indicate malignancy
                region_probability += 0.1
            elif calc_feature['pattern'] == 'central':
                # Central calcification typically indicates benign granuloma
                region_probability -= 0.15
            elif calc_feature['pattern'] == 'diffuse':
                # Diffuse calcification typically indicates benign process
                region_probability -= 0.2
            elif calc_feature['pattern'] == 'laminated':
                # Laminated calcification typically indicates benign process
                region_probability -= 0.15

        # Density-based risk
        if region_key in calc_features:
            mean_hu = calc_features[region_key].get('mean_attenuation', 0)
            if mean_hu > 0:  # Solid nodule
                region_probability += 0.1
            elif mean_hu < -400:  # Ground glass
                region_probability += 0.05
                
        # Add a size sanity check
        if size_feature['max_diameter_mm'] > 40:  # Suspiciously large
            # Apply a penalty for extremely large regions (likely false positives)
            region_probability *= 0.1

        # Ensure probability stays in valid range
        region_probability = max(0.05, min(region_probability, 0.95))
        
        cancer_regions.append({
            "region_id": region_idx,
            "probability": region_probability,
            "size_mm": size_feature['max_diameter_mm'],
            "category": size_feature['category'],
            "shape": "irregular" if shape_feature['circularity'] < 0.75 else "regular",
            "calcification": calc_feature.get('pattern', 'none')
        })
        
        cancer_probability = max(cancer_probability, region_probability)
    
    # Create final visualization
    has_cancer = cancer_probability > 0.5
    result = {
        "has_cancer": has_cancer,
        "confidence": cancer_probability,
        "regions": cancer_regions,
        "message": generate_result_message(has_cancer, cancer_probability, cancer_regions)
    }
    
    # Generate the standard output visualization
    visualize_results(original_img, filtered_img, lung_mask, filtered_nodules, 
                     result, output_dir, base_filename)
    
    return result

def generate_result_message(has_cancer, probability, regions):
    """
    Generate a human-readable result message
    
    Parameters:
        has_cancer: True or false
        probability: Percent chance of cancer
        regions: Possible cancer regions
        
    Returns:
        message: Information message printed to screen
    """
    
    if not has_cancer:
        return "No evidence of cancerous tumors detected in this CT scan."
    
    # Sort regions by probability (highest first)
    sorted_regions = sorted(regions, key=lambda x: x['probability'], reverse=True)
    
    message = f"Potential cancerous tumor(s) detected with {probability:.1%} confidence.\n"
    
    for i, region in enumerate(sorted_regions[:3]):  # Show top 3 regions
        message += f"\nRegion {i+1}: {region['probability']:.1%} probability of malignancy\n"
        message += f"- Size: {region['size_mm']:.1f}mm ({region['category']})\n"
        message += f"- Shape: {region['shape']}\n"
        message += f"- Calcification: {region['calcification']}\n"
    
    if len(sorted_regions) > 3:
        message += f"\n{len(sorted_regions) - 3} additional suspicious regions found."
    
    return message

def visualize_results(original_img, filtered_img, lung_mask, nodule_mask, 
                     result, output_dir, base_filename):
    """
    Visualize and save analysis results
    
    Parameters:
        original_img: Original CT scan
        filtered_img: Filtered CT scan
        lung_mask: Mask of lung regions
        nodule_mask: Mask of nodules found
        result: Cancer result
        output_dir: Output filename
        base_filename: Original filename
    """
    
    # Apply windowing for better visualization
    display_original = apply_hu_windowing(original_img, -600, 1500)
    display_filtered = apply_hu_windowing(filtered_img, -600, 1500)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(display_original, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    # Filtered image
    axes[0, 1].imshow(display_filtered, cmap='gray')
    axes[0, 1].set_title('Filtered CT Image')
    axes[0, 1].axis('off')
    
    # Lung segmentation
    axes[1, 0].imshow(display_original, cmap='gray')
    lung_overlay = np.zeros_like(display_original)
    lung_overlay[lung_mask] = 1
    axes[1, 0].imshow(lung_overlay, cmap='Blues', alpha=0.3)
    axes[1, 0].set_title('Lung Segmentation')
    axes[1, 0].axis('off')
    
    # Nodule detection
    axes[1, 1].imshow(display_original, cmap='gray')
    nodule_overlay = np.zeros_like(display_original)
    nodule_overlay[nodule_mask] = 1
    axes[1, 1].imshow(nodule_overlay, cmap='Reds', alpha=0.5)
    
    # Add cancer probability to title
    if result["has_cancer"]:
        axes[1, 1].set_title(f'Potential Tumors (Cancer Probability: {result["confidence"]:.1%})')
    else:
        axes[1, 1].set_title('No Significant Tumors Detected')
    
    axes[1, 1].axis('off')
    
    # Add overall result as figure title
    if result["has_cancer"]:
        fig.suptitle("CANCER DETECTED", fontsize=16, color='red')
    else:
        fig.suptitle("No Cancer Detected", fontsize=16, color='green')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save visualization
    plt.savefig(os.path.join(output_dir, f"{base_filename}_analysis_results.png"), dpi=300)


if __name__ == "__main__":
    """
    Test results for results.py
    Saves cancer probability results and resulting message
    """
    # Check if a file path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Analyzing CT scan: {image_path}")
        
        # Run the analysis
        result = analyze_ct_scan_with_improved_detection(image_path)
        
        # Print the result message
        print("\n" + "="*50)
        print("ANALYSIS RESULTS:")
        print("="*50)
        print(result["message"])
        print("="*50)
        
        # Print the output directory
        print(f"\nDetailed results saved in 'cancer_analysis_results' directory")
    else:
        print("Please provide the path to your CT scan DICOM file.")
        print("Usage: python results.py path/to/your/ct_scan.dcm")