from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename

from utils.supabase_client import get_supabase_client, initialize_database, initialize_storage
from utils.video_utils import extract_frames, get_video_metadata, download_video, cleanup_temp_file
from features.extractor import extract_all_features, serialize_features, deserialize_features
from search.ranking import rank_videos_by_similarity
from config import VIDEOS_BUCKET, VIDEOS_TABLE, FEATURES_TABLE, TOP_RESULTS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size

# Initialize Supabase
supabase = get_supabase_client()
initialize_database()
initialize_storage()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle video upload."""
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_file.name)
        temp_file.close()
        
        try:
            # Generate a unique ID for the video
            video_id = str(uuid.uuid4())
            
            # Extract features
            features = extract_all_features(temp_file.name)
            
            # Get video metadata
            metadata = get_video_metadata(temp_file.name)
            
            # Upload video to Supabase Storage
            filename = secure_filename(video_file.filename)
            storage_path = f"{video_id}/{filename}"
            
            with open(temp_file.name, 'rb') as f:
                supabase.storage.from_(VIDEOS_BUCKET).upload(storage_path, f)
            
            # Get public URL
            file_url = supabase.storage.from_(VIDEOS_BUCKET).get_public_url(storage_path)
            
            # Store video metadata in database
            video_data = {
                "id": video_id,
                "filename": filename,
                "storage_path": storage_path,
                "url": file_url,
                "duration": metadata["duration"],
                "width": metadata["width"],
                "height": metadata["height"],
                "fps": metadata["fps"]
            }
            
            supabase.table(VIDEOS_TABLE).insert(video_data).execute()
            
            # Store features in database
            feature_data = {
                "video_id": video_id,
                "features": serialize_features(features)
            }
            
            supabase.table(FEATURES_TABLE).insert(feature_data).execute()
            
            return jsonify({
                "success": True,
                "video_id": video_id,
                "message": "Video uploaded and processed successfully"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            # Clean up temporary file
            cleanup_temp_file(temp_file.name)
    
    return render_template('upload.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle video search."""
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_file.name)
        temp_file.close()
        
        try:
            # Extract features from the query video
            query_features = extract_all_features(temp_file.name)
            
            # Get all video features from the database
            response = supabase.table(FEATURES_TABLE).select("*").execute()
            
            if not response.data:
                return jsonify({"error": "No videos found in the database"}), 404
            
            # Prepare video features list
            video_features_list = []
            for item in response.data:
                video_id = item["video_id"]
                features = deserialize_features(item["features"])
                video_features_list.append((video_id, features))
            
            # Rank videos by similarity
            ranked_videos = rank_videos_by_similarity(query_features, video_features_list, TOP_RESULTS)
            
            # Get video details for the top results
            result_videos = []
            for item in ranked_videos:
                video_id = item["video_id"]
                video_response = supabase.table(VIDEOS_TABLE).select("*").eq("id", video_id).execute()
                
                if video_response.data:
                    video_data = video_response.data[0]
                    result_videos.append({
                        "id": video_data["id"],
                        "filename": video_data["filename"],
                        "url": video_data["url"],
                        "similarity": item["similarity"],
                        "detailed_similarities": item["detailed_similarities"]
                    })
            
            return jsonify({
                "success": True,
                "results": result_videos
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            # Clean up temporary file
            cleanup_temp_file(temp_file.name)
    
    return render_template('search.html')

@app.route('/videos')
def videos():
    """Display all videos."""
    response = supabase.table(VIDEOS_TABLE).select("*").execute()
    
    if not response.data:
        return render_template('videos.html', videos=[])
    
    return render_template('videos.html', videos=response.data)

@app.route('/video/<video_id>')
def video_details(video_id):
    """Display details for a specific video."""
    video_response = supabase.table(VIDEOS_TABLE).select("*").eq("id", video_id).execute()
    
    if not video_response.data:
        return redirect(url_for('videos'))
    
    features_response = supabase.table(FEATURES_TABLE).select("*").eq("video_id", video_id).execute()
    
    features = None
    if features_response.data:
        features = deserialize_features(features_response.data[0]["features"])
    
    return render_template('video_details.html', video=video_response.data[0], features=features)

if __name__ == '__main__':
    app.run(debug=True)