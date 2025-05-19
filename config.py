import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Database table names
VIDEOS_TABLE = "videos"
FEATURES_TABLE = "video_features"

# Storage bucket name
VIDEOS_BUCKET = "videos"

# Feature extraction settings
FRAME_SAMPLE_RATE = 1  # Extract 1 frame per second
COLOR_BINS = 8  # Number of bins for color histogram
RESIZE_DIMENSIONS = (224, 224)  # Resize frames for consistent processing

# Search settings
TOP_RESULTS = 3  # Number of similar videos to return