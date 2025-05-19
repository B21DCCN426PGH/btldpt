import numpy as np
import json
from utils.video_utils import extract_frames, get_video_metadata
from features.color_features import extract_color_histogram, extract_dominant_colors, calculate_color_statistics
from features.frame_features import extract_edge_features, extract_texture_features, extract_pca_features
from features.object_features import extract_object_features, detect_objects

def extract_all_features(video_path):
    """Extract all features from a video."""
    # Get video metadata
    metadata = get_video_metadata(video_path)
    
    # Extract frames
    frames = extract_frames(video_path)
    
    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")
    
    # Initialize feature dictionary
    features = {
        "metadata": metadata,
        "frame_features": [],
        "global_features": {}
    }
    
    # Process each frame
    for i, frame in enumerate(frames):
        frame_features = {}
        
        # Extract color features
        frame_features["color_histogram"] = extract_color_histogram(frame).tolist()
        frame_features["dominant_colors"] = extract_dominant_colors(frame).tolist()
        frame_features["color_statistics"] = calculate_color_statistics(frame).tolist()
        
        # Extract frame features
        frame_features["edge_features"] = extract_edge_features(frame).tolist()
        frame_features["texture_features"] = extract_texture_features(frame).tolist()
        
        # Extract object features (computationally expensive, so we'll do it for fewer frames)
        if i % 5 == 0:  # Only process every 5th frame for object detection
            frame_features["object_features"] = extract_object_features(frame).tolist()
            frame_features["detected_objects"] = detect_objects(frame)
        
        features["frame_features"].append(frame_features)
    
    # Extract global features
    features["global_features"]["pca_features"] = extract_pca_features(frames).tolist()
    
    # Calculate average features across all frames
    avg_color_hist = np.mean([f["color_histogram"] for f in features["frame_features"]], axis=0).tolist()
    avg_dominant_colors = np.mean([f["dominant_colors"] for f in features["frame_features"]], axis=0).tolist()
    avg_color_stats = np.mean([f["color_statistics"] for f in features["frame_features"]], axis=0).tolist()
    avg_edge_features = np.mean([f["edge_features"] for f in features["frame_features"]], axis=0).tolist()
    avg_texture_features = np.mean([f["texture_features"] for f in features["frame_features"]], axis=0).tolist()
    
    # Combine object detections across frames
    all_objects = {}
    object_count = 0
    for f in features["frame_features"]:
        if "detected_objects" in f:
            object_count += 1
            for obj, score in f["detected_objects"].items():
                if obj in all_objects:
                    all_objects[obj] += score
                else:
                    all_objects[obj] = score
    
    # Average object scores
    if object_count > 0:
        for obj in all_objects:
            all_objects[obj] /= object_count
    
    # Add average features to global features
    features["global_features"]["avg_color_histogram"] = avg_color_hist
    features["global_features"]["avg_dominant_colors"] = avg_dominant_colors
    features["global_features"]["avg_color_statistics"] = avg_color_stats
    features["global_features"]["avg_edge_features"] = avg_edge_features
    features["global_features"]["avg_texture_features"] = avg_texture_features
    features["global_features"]["detected_objects"] = all_objects
    
    return features

def serialize_features(features):
    """Serialize features to JSON format for storage."""
    # Create a custom JSON encoder that handles NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Convert numpy arrays to lists
    serialized = json.dumps(features, cls=NumpyEncoder)
    return serialized

def deserialize_features(serialized_features):
    """Deserialize features from JSON format."""
    features = json.loads(serialized_features)
    return features