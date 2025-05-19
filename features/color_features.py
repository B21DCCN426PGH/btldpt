import cv2
import numpy as np
from config import COLOR_BINS

def extract_color_histogram(frame):
    """Extract color histogram features from a frame."""
    # Convert to HSV color space (better for color analysis)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Calculate histogram for each channel
    hist_h = cv2.calcHist([hsv_frame], [0], None, [COLOR_BINS], [0, 180])
    hist_s = cv2.calcHist([hsv_frame], [1], None, [COLOR_BINS], [0, 256])
    hist_v = cv2.calcHist([hsv_frame], [2], None, [COLOR_BINS], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
    
    # Concatenate histograms
    hist_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    return hist_features

def extract_dominant_colors(frame, k=3):
    """Extract dominant colors from a frame using K-means clustering."""
    # Reshape the frame to be a list of pixels
    pixels = frame.reshape((-1, 3))
    
    # Convert to float32
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Count occurrences of each label
    counts = np.bincount(labels.flatten())
    
    # Sort colors by frequency
    sorted_indices = np.argsort(counts)[::-1]
    sorted_centers = centers[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Calculate percentage of each color
    total_pixels = frame.shape[0] * frame.shape[1]
    color_percentages = sorted_counts / total_pixels
    
    # Combine colors and their percentages
    dominant_colors = []
    for i in range(min(k, len(sorted_centers))):
        dominant_colors.extend(sorted_centers[i])
        dominant_colors.append(color_percentages[i])
    
    return np.array(dominant_colors)

def calculate_color_statistics(frame):
    """Calculate statistical features of color channels."""
    # Split into channels
    r, g, b = cv2.split(frame)
    
    # Calculate statistics for each channel
    stats = []
    for channel in [r, g, b]:
        mean = np.mean(channel)
        std = np.std(channel)
        median = np.median(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        stats.extend([mean, std, median, min_val, max_val])
    
    return np.array(stats)