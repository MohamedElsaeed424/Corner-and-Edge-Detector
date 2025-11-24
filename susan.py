
import numpy as np
import cv2
import matplotlib.pyplot as plt

# def calculate_t(image, method='std', k=0.5):
#     """
#     Calculate the brightness difference threshold 't' for SUSAN algorithm.
    
#     Parameters:
#     - image: Grayscale input image (2D numpy array)
#     - method: Method to calculate t
#         - 'std': Based on standard deviation (t = k * std)
#         - 'gradient': Based on gradient magnitude
#         - 'noise': Based on noise estimation
#     - k: Scaling factor (default: 0.5)
    
#     Returns:
#     - t: Calculated threshold value
#     """
#     img = image.astype(np.float32)
    
#     if method == 'std':
#         # Method 1: Based on standard deviation
#         # t = k * standard_deviation of image
#         std_dev = np.std(img)
#         t = k * std_dev
#         # Manual max: return larger value
#         if t > 1.0:
#             return t
#         else:
#             return 1.0
    
#     elif method == 'gradient':
#         # Method 2: Based on gradient magnitude
#         # Calculate gradients
#         grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#         grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#         gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
#         # Use median gradient magnitude
#         t = k * np.median(gradient_magnitude)
#         # Manual max: return larger value
#         if t > 1.0:
#             return t
#         else:
#             return 1.0
    
#     elif method == 'noise':
#         # Method 3: Estimate noise level
#         # Use median absolute deviation (MAD) as noise estimate
#         median = np.median(img)
#         mad = np.median(np.abs(img - median))
#         # MAD to standard deviation approximation: σ ≈ 1.4826 * MAD
#         noise_std = 1.4826 * mad
#         t = k * noise_std
#         # Manual max: return larger value
#         if t > 1.0:
#             return t
#         else:
#             return 1.0
    
#     else:
#         # Default: return a reasonable value
#         return 27.0

# Manual append function
def manual_append(mask_list, element):
    # Create new list with one more element
    new_list = [0] * (len(mask_list) + 1)
    # Copy old elements
    for i in range(len(mask_list)):
        new_list[i] = mask_list[i]
    # Add new element at the end
    new_list[len(mask_list)] = element
    return new_list

# //get me the pixels pos. inside the mask
def circular_mask(radius):
    mask = []
    #how we move up,down from neucleus
    for a in range(-radius, radius+1):
        #let,right
        for b in range(-radius, radius+1):
            #x^2+y^2<=radius^2
            if b*b + a*a <= radius*radius:
                # Manual append: add (b, a) to mask
                mask = manual_append(mask, (b, a))
    return mask


def susan(image, t, radius=3):
    #height,width of input img
    h, w = image.shape
    # Convert image to float for precise calculations
    img = image.astype(np.float32)    
    # Pad the image to handle border pixels
    pad = radius
    padded = np.pad(img, pad, mode='edge')    
    #get pixels pos in mask
    mask = circular_mask(radius)   
    #create 2d array of zeros same size as img 
    scores = np.zeros((h, w), dtype=np.float32)    
    # loop through each pixel
    for y in range(h):
        for x in range(w):
            # pixel instens.
            center_intensity = padded[y+pad, x+pad]

            usan_size = 0.0
            # Iterate through all pixels in the circular mask
            for b, a in mask:
                #  neighbor pixel ,up ,left,right,dw
                ny = y + a + pad
                nx = x + b + pad
                #  neighbor pixel instens.
                neighbor_intensity = padded[ny, nx]
                # c(m) = e^(-((I(m)-I(m₀))/t)⁶)
                intensity_diff = neighbor_intensity - center_intensity
                c_m = np.exp(-((intensity_diff / t) ** 6))
                # Sum up all c(m) values to get n(m₀)
                usan_size += c_m
            scores[y, x] = usan_size
    #2d array same size of img but has usan size for that pixel
    return scores

# manual min
def manual_min(arr):
    # Flatten array to 1D
    flat_arr = arr.flatten()
    # Start with first value
    min_val = flat_arr[0]
    # Check each value
    for i in range(1, len(flat_arr)):
        if flat_arr[i] < min_val:
            min_val = flat_arr[i]
    return min_val

# # manual max
def manual_max(arr):
    # Flatten array to 1D
    flat_arr = arr.flatten()
    # Start with first value
    max_val = flat_arr[0]
    # Check each value
    for i in range(1, len(flat_arr)):
        if flat_arr[i] > max_val:
            max_val = flat_arr[i]
    return max_val

#create histogram from image scores after sawsan
def susan_histogram(scores, bins=50):
    minv = manual_min(scores)
    maxv = manual_max(scores)  
    # Create histogram using numpy
    # Flatten the scores array and create histogram
    hist, bin_edges = np.histogram(scores.flatten(), bins=bins, range=(minv, maxv))
    
    return hist, bin_edges


def main():

    image_path = 'Test Image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return
    # Print the 2D array of the test image
    print("original img array")
    print(image)
    # Test case 1
    threshold = 27 
    radius = 3      
    # Print value of t
    print(f"\nt = {threshold}")

    scores = susan(image, t=threshold, radius=radius)
    print("\n2. result img array")
    print(scores)

    hist, bin_edges = susan_histogram(scores)

    print("\n3. result histogram")
    print(hist)
    
    # Create and save histogram visualization plot
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0])*0.8, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('USAN Size (n(m₀))', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Pixels (Frequency)', fontsize=12, fontweight='bold')
    plt.title('Histogram of SUSAN Corner Detection Result\n(USAN Size Distribution)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Use manual max function
    max_count = manual_max(hist)
    for i, (x, y) in enumerate(zip(bin_centers, hist)):
        if y > max_count * 0.01:  # Only label bars with >1% of max count
            plt.text(x, y, str(int(y)), ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig('histogram_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find corners (pixels with low USAN size - few similar neighbors)
    mask_size = len(circular_mask(radius))
    geometric_threshold = 0.5
    corner_threshold = mask_size * geometric_threshold
    
    # Create binary corner map: 255 for corners, 0 for non-corners
    corners = (scores < corner_threshold).astype(np.uint8) * 255
    
    # Save corner detection result
    cv2.imwrite('corners_detected.png', corners)


if __name__ == "__main__":
    main()
