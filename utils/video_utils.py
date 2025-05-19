import cv2
import numpy as np
import os
import tempfile
from config import FRAME_SAMPLE_RATE, RESIZE_DIMENSIONS

def download_video(supabase, bucket_name, file_path):
    """Download a video from Supabase storage to a temporary file."""
    try:
        response = supabase.storage.from_(bucket_name).download(file_path)
        
        # Create a temporary file to store the downloaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(response)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def extract_frames(video_path, sample_rate=FRAME_SAMPLE_RATE):
    """Extract frames from a video at the specified sample rate."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)
    
    if frame_interval <= 0:
        frame_interval = 1
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Resize frame for consistent processing
            resized_frame = cv2.resize(frame, RESIZE_DIMENSIONS)
            # Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def get_video_metadata(video_path):
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return metadata

def cleanup_temp_file(file_path):
    """Remove a temporary file."""
    if os.path.exists(file_path):
        os.unlink(file_path)