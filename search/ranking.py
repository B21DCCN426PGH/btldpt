from search.similarity import calculate_overall_similarity

def rank_videos_by_similarity(query_features, video_features_list, top_n=3):
    """Rank videos by similarity to the query video."""
    similarities = []
    
    for video_id, features in video_features_list:
        similarity, detailed_similarities = calculate_overall_similarity(query_features, features)
        similarities.append({
            "video_id": video_id,
            "similarity": similarity,
            "detailed_similarities": detailed_similarities
        })
    
    # Sort by similarity (descending)
    ranked_videos = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    # Return top N results
    return ranked_videos[:top_n]