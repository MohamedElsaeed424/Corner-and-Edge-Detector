"""
SUSAN Corner Detection Algorithm
================================
SUSAN (Smallest Univalue Segment Assimilating Nucleus) is a corner detection algorithm
that identifies corners by analyzing the similarity of pixel intensities within a 
circular mask centered on each pixel.

The key idea: If most pixels in the mask are similar to the center pixel, it's likely
an edge or flat region. If few pixels are similar, it's likely a corner.
"""

import cv2
import numpy as np


def create_circular_mask(radius):
    """
    Create a circular mask for the SUSAN algorithm.
    
    Parameters:
    - radius: The radius of the circular mask (typically 3 or 4)
    
    Returns:
    - mask: A 2D boolean array where True indicates pixels inside the circle
    """
    # Create coordinate grids
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # Create circular mask: pixels where distance from center <= radius
    mask = (x**2 + y**2) <= radius**2
    
    return mask


def susan_corner_detection(image, threshold=27, geometric_threshold=0.5, radius=3):
    """
    Perform SUSAN corner detection on an input image.
    
    Parameters:
    - image: Grayscale input image (2D numpy array)
    - threshold: Brightness difference threshold (t) - determines what counts as "similar"
                 Lower values = more strict similarity requirement
    - geometric_threshold: Geometric threshold (g) - determines corner sensitivity
                          Typical value is 0.5, lower = more corners detected
    - radius: Radius of the circular mask (typically 3)
    
    Returns:
    - corners: Binary image where 255 indicates a corner, 0 otherwise
    """
    # Convert image to float for precise calculations
    img = image.astype(np.float32)
    
    # Get image dimensions
    height, width = img.shape
    
    # Create the circular mask
    mask = create_circular_mask(radius)
    mask_size = np.sum(mask)  # Total number of pixels in the mask
    
    # Initialize output image (all zeros = no corners initially)
    corners = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate through each pixel in the image
    # Skip border pixels (radius pixels from each edge) to avoid out-of-bounds
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            
            # Get the center pixel intensity
            center_intensity = img[i, j]
            
            # Extract the region of interest (ROI) around current pixel
            roi = img[i-radius:i+radius+1, j-radius:j+radius+1]
            
            # Apply circular mask to get only pixels inside the circle
            masked_pixels = roi[mask]
            
            # Calculate absolute difference between center pixel and all pixels in mask
            intensity_differences = np.abs(masked_pixels - center_intensity)
            
            # Apply comparison function: c(r) = exp(-(d/t)^6)
            # This gives a value close to 1 for similar pixels, close to 0 for different pixels
            comparison_values = np.exp(-(intensity_differences / threshold) ** 6)
            
            # Calculate USAN (Univalue Segment Assimilating Nucleus) area
            # This is the sum of comparison values - indicates how many pixels are "similar"
            usan_area = np.sum(comparison_values)
            
            # Calculate corner response: R = 1 - n/N
            # where n = USAN area, N = mask size
            # Low USAN area (few similar pixels) = high response = corner
            corner_response = 1 - (usan_area / mask_size)
            
            # If corner response exceeds geometric threshold, mark as corner
            if corner_response > geometric_threshold:
                corners[i, j] = 255
    
    return corners


def create_test_image():
    """
    Create a simple test image with corners for testing the SUSAN algorithm.
    This creates a white square on black background with clear corners.
    
    Returns:
    - test_image: A grayscale test image with corners
    """
    # Create a black image (all zeros)
    test_image = np.zeros((300, 300), dtype=np.uint8)
    
    # Draw a white square (this will have 4 corners)
    cv2.rectangle(test_image, (50, 50), (250, 250), 255, -1)
    
    # Draw some additional shapes to create more corners
    # Draw a triangle (3 corners)
    triangle_pts = np.array([[150, 100], [100, 200], [200, 200]], np.int32)
    cv2.fillPoly(test_image, [triangle_pts], 255)
    
    return test_image


def main():
    """
    Main function to demonstrate SUSAN corner detection.
    """
    # Option 1: Load an image from file
    # Replace 'input_image.jpg' with your image path
    image_path = 'input_image.jpg'
    
    try:
        # Read image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Option 2: If no image file, create a test image instead
        if image is None:
            print(f"Could not load image from {image_path}")
            print("Creating a test image with corners instead...")
            image = create_test_image()
            cv2.imwrite('test_image.png', image)
            print("Test image saved as 'test_image.png'")
        
        # Perform SUSAN corner detection
        print("Detecting corners using SUSAN algorithm...")
        corners = susan_corner_detection(
            image, 
            threshold=27,        # Brightness threshold
            geometric_threshold=0.5,  # Corner sensitivity
            radius=3             # Mask radius
        )
        
        # Create a visualization: overlay corners on original image
        # Make a copy of the original image
        result = image.copy()
        
        # Mark corners in red (if image is grayscale, convert to BGR first)
        if len(image.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Draw corners as red circles
        corner_coords = np.where(corners == 255)
        for y, x in zip(corner_coords[0], corner_coords[1]):
            cv2.circle(result, (x, y), 2, (0, 0, 255), -1)  # Red filled circle
        
        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Detected Corners', corners)
        cv2.imshow('Corners Overlay', result)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results
        cv2.imwrite('corners_detected.png', corners)
        cv2.imwrite('corners_overlay.png', result)
        print("Results saved as 'corners_detected.png' and 'corners_overlay.png'")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

