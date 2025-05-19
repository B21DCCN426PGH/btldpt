from supabase import create_client
import os
from config import SUPABASE_URL, SUPABASE_KEY

def get_supabase_client():
    """Create and return a Supabase client instance."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL and key must be provided in .env file")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def initialize_database():
    """Initialize the database tables if they don't exist."""
    supabase = get_supabase_client()
    
    # Create videos table
    supabase.table("videos").select("*").limit(1).execute()
    print("Connected to videos table")
    
    # Create features table
    supabase.table("video_features").select("*").limit(1).execute()
    print("Connected to video_features table")
    
    return supabase

def initialize_storage():
    """Initialize the storage bucket connection."""
    supabase = get_supabase_client()
    
    try:
        # Only check if bucket exists, don't try to create it
        supabase.storage.from_("videos").get_public_url("test.txt")
        print("Connected to videos bucket")
    except Exception as e:
        print(f"Error connecting to videos bucket: {e}")
        print("Please make sure the 'videos' bucket exists in your Supabase project")
        print("Create it manually through the Supabase dashboard")
        import sys
        sys.exit(1)
    
    return supabase