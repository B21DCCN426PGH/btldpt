# test_supabase.py
from utils.supabase_client import get_supabase_client

def test_supabase_connection():
    """Test Supabase connection and storage access."""
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
        
        # Test database access
        response = supabase.table("videos").select("*").limit(1).execute()
        print("✅ Database connection successful")
        
        # Test storage access
        try:
            # Try to list files in the bucket
            response = supabase.storage.from_("videos").list()
            print("✅ Storage bucket 'videos' exists and is accessible")
            print(f"Files in bucket: {len(response)}")
        except Exception as e:
            print(f"❌ Error accessing storage bucket: {e}")
            print("Please create the 'videos' bucket in the Supabase dashboard")
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")

if __name__ == "__main__":
    test_supabase_connection()