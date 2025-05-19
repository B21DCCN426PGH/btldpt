import cv2
import numpy as np
from sklearn.decomposition import PCA

def extract_edge_features(frame):
    """Extract edge features using Canny edge detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Calculate edge density (percentage of edge pixels)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Divide the image into a 4x4 grid and calculate edge density for each cell
    h, w = edges.shape
    cell_h, cell_w = h // 4, w // 4
    
    grid_features = []
    for i in range(4):
        for j in range(4):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_density = np.sum(cell > 0) / (cell.shape[0] * cell.shape[1])
            grid_features.append(cell_density)
    
    # Combine global and grid features
    edge_features = np.array([edge_density] + grid_features)
    
    return edge_features

def extract_texture_features(frame):
    """Extract texture features using Haralick texture features."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = np.zeros((8, 8))
    h, w = gray.shape
    
    # Quantize to 8 levels
    gray_quantized = (gray // 32).astype(np.uint8)
    
    # Calculate co-occurrence matrix
    for i in range(h-1):
        for j in range(w-1):
            glcm[gray_quantized[i, j], gray_quantized[i+1, j+1]] += 1
    
    # Normalize GLCM
    glcm = glcm / np.sum(glcm)
    
    # Calculate texture properties
    contrast = 0
    homogeneity = 0
    energy = 0
    correlation = 0
    
    # Calculate mean and standard deviation
    mean_i = 0
    mean_j = 0
    for i in range(8):
        for j in range(8):
            mean_i += i * np.sum(glcm[i, :])
            mean_j += j * np.sum(glcm[:, j])
    
    std_i = 0
    std_j = 0
    for i in range(8):
        for j in range(8):
            std_i += (i - mean_i)**2 * np.sum(glcm[i, :])
            std_j += (j - mean_j)**2 * np.sum(glcm[:, j])
    
    std_i = np.sqrt(std_i)
    std_j = np.sqrt(std_j)
    
    # Calculate texture features
    for i in range(8):
        for j in range(8):
            contrast += (i - j)**2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
            energy += glcm[i, j]**2
            if std_i > 0 and std_j > 0:
                correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)
    
    return np.array([contrast, homogeneity, energy, correlation])

def extract_pca_features(frames, n_components=50):
    """Extract PCA features from a set of frames."""
    if len(frames) == 0:
        return np.zeros(n_components)
    
    # Convert frames to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).flatten() for frame in frames]
    
    # Stack frames
    X = np.vstack(gray_frames)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    pca_features = pca.fit_transform(X)
    
    # Take the mean of PCA features across frames
    mean_pca_features = np.mean(pca_features, axis=0)
    
    # Pad with zeros if necessary
    if mean_pca_features.shape[0] < n_components:
        padding = np.zeros(n_components - mean_pca_features.shape[0])
        mean_pca_features = np.concatenate([mean_pca_features, padding])
    
    return mean_pca_features