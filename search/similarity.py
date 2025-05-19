import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_color_histogram_similarity(hist1, hist2):
    """Calculate similarity between two color histograms."""
    return cosine_similarity([hist1], [hist2])[0][0]

def calculate_dominant_color_similarity(colors1, colors2):
    """Calculate similarity between dominant colors."""
    return cosine_similarity([colors1], [colors2])[0][0]

def calculate_edge_similarity(edge1, edge2):
    """Calculate similarity between edge features."""
    return cosine_similarity([edge1], [edge2])[0][0]

def calculate_texture_similarity(texture1, texture2):
    """Calculate similarity between texture features."""
    return cosine_similarity([texture1], [texture2])[0][0]

def calculate_pca_similarity(pca1, pca2):
    """Calculate similarity between PCA features."""
    return cosine_similarity([pca1], [pca2])[0][0]

def calculate_object_similarity(objects1, objects2):
    """Calculate similarity between detected objects."""
    # Get all unique object labels
    all_objects = set(objects1.keys()) | set(objects2.keys())
    
    if not all_objects:
        return 0.0
    
    # Create vectors for each object set
    vec1 = np.zeros(len(all_objects))
    vec2 = np.zeros(len(all_objects))
    
    for i, obj in enumerate(all_objects):
        vec1[i] = objects1.get(obj, 0)
        vec2[i] = objects2.get(obj, 0)
    
    # Calculate cosine similarity
    return cosine_similarity([vec1], [vec2])[0][0]

def calculate_overall_similarity(features1, features2, weights=None):
    """Calculate overall similarity between two videos based on their features."""
    # Default weights if not provided
    if weights is None:
        weights = {
            "color_histogram": 0.2,
            "dominant_colors": 0.1,
            "color_statistics": 0.1,
            "edge_features": 0.15,
            "texture_features": 0.15,
            "pca_features": 0.2,
            "objects": 0.1
        }
    
    # Extract global features
    global1 = features1["global_features"]
    global2 = features2["global_features"]
    
    # Calculate similarities for each feature type
    similarities = {}
    
    # Color histogram similarity
    similarities["color_histogram"] = calculate_color_histogram_similarity(
        global1["avg_color_histogram"], 
        global2["avg_color_histogram"]
    )
    
    # Dominant colors similarity
    similarities["dominant_colors"] = calculate_dominant_color_similarity(
        global1["avg_dominant_colors"], 
        global2["avg_dominant_colors"]
    )
    
    # Color statistics similarity
    similarities["color_statistics"] = calculate_color_histogram_similarity(
        global1["avg_color_statistics"], 
        global2["avg_color_statistics"]
    )
    
    # Edge features similarity
    similarities["edge_features"] = calculate_edge_similarity(
        global1["avg_edge_features"], 
        global2["avg_edge_features"]
    )
    
    # Texture features similarity
    similarities["texture_features"] = calculate_texture_similarity(
        global1["avg_texture_features"], 
        global2["avg_texture_features"]
    )
    
    # PCA features similarity
    similarities["pca_features"] = calculate_pca_similarity(
        global1["pca_features"], 
        global2["pca_features"]
    )
    
    # Object similarity
    if "detected_objects" in global1 and "detected_objects" in global2:
        similarities["objects"] = calculate_object_similarity(
            global1["detected_objects"], 
            global2["detected_objects"]
        )
    else:
        similarities["objects"] = 0.0
        weights["objects"] = 0.0
        # Normalize remaining weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
    
    # Calculate weighted average
    overall_similarity = sum(similarities[key] * weights[key] for key in similarities)
    
    return overall_similarity, similarities