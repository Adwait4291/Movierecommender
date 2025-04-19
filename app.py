# [source: 3] Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

# [source: 3] Standard library imports
import os
import sys
import logging
import random
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Any

# [source: 3] Third-party imports
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import click # For CLI commands

# [source: 3] Flask and extensions
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, flash, session, g
)

# --- Configuration ---

# [source: 3] Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

# --- Constants ---
# [source: 3] Load keys/secrets from environment variables (e.g., from .env file)
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')

# --- Check for Essential Configuration ---
# [source: 3] Exit if critical environment variables are missing
if not TMDB_API_KEY:
    logging.error("FATAL ERROR: TMDB_API_KEY environment variable not set.")
    sys.exit("TMDB_API_KEY is required. Please set the environment variable (e.g., in .env file).")

# [source: 3] TMDB Base URL and Data File Path
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
# [source: 3] Ensure this file is in the same directory as app.py
DATA_FILE = 'Hydra-Movie-Scrape.csv'

# --- Flask App Initialization ---
# [source: 3] Initialize Flask app
app = Flask(__name__)

# --- Data Loading and Preprocessing ---
# [source: 3] Use LRU cache to avoid reloading data on every request
@lru_cache(maxsize=1)
def load_and_preprocess_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    [source: 65] Loads the movie dataset from CSV, cleans, preprocesses it, and adds an 'id' column based on the final index.
    [source: 65] Returns the processed DataFrame or None if loading fails.
    """
    try:
        # [source: 65] Load data using pandas
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from {filepath}. Initial shape: {df.shape}")

        # [source: 65] --- Data Cleaning ---
        # [source: 65] Define essential columns needed for the application
        essential_cols = ['Title', 'Summary', 'Cast', 'Director', 'Writers', 'Year', 'Runtime', 'Rating', 'Movie Poster']
        # [source: 65] Check if any essential columns are missing
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            logging.error(f"Missing essential columns in CSV: {missing_essential}")
            return None

        original_rows = len(df)
        # [source: 65] Drop rows with missing values in essential columns
        df.dropna(subset=essential_cols, how='any', inplace=True)
        logging.info(f"Shape after dropping NA in essential columns: {df.shape} ({original_rows - len(df)} rows dropped)")

        # [source: 65] Convert Year safely to numeric, coercing errors, then drop rows where conversion failed
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year'], inplace=True)
        df['Year'] = df['Year'].astype(int)

        # [source: 66] Convert Runtime safely, extracting digits first
        # Ensure Runtime is treated as string before extraction
        df['Runtime'] = df['Runtime'].astype(str).str.extract(r'(\d+)', expand=False)
        df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
        df.dropna(subset=['Runtime'], inplace=True)
        df['Runtime'] = df['Runtime'].astype(int)

        # [source: 66] Convert Rating safely
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df.dropna(subset=['Rating'], inplace=True) # Drop rows where rating conversion failed

        # [source: 66] Fill remaining NAs in specified text fields with empty strings
        text_cols_to_fill = ['Summary', 'Short Summary', 'Cast', 'Director', 'Writers', 'IMDB ID', 'YouTube Trailer', 'Movie Poster']
        for col in text_cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                # [source: 67] If a non-essential text column doesn't exist, create it empty to avoid errors later
                logging.warning(f"Column '{col}' not found in CSV. Creating empty column.")
                df[col] = ''

        # [source: 67] Reset index AFTER all filtering/dropping to ensure it's contiguous and starts from 0
        df.reset_index(drop=True, inplace=True)
        # [source: 67] Add 'id' column based on the final DataFrame index for consistent lookup
        df['id'] = df.index
        logging.info(f"Data preprocessing complete. Final shape: {df.shape}")
        return df
    # [source: 67] Handle file not found error specifically
    except FileNotFoundError:
        logging.error(f"FATAL ERROR: Data file not found at {filepath}")
        return None
    # [source: 67] Handle any other exceptions during loading/preprocessing
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}", exc_info=True)
        return None

# --- Feature Matrix Creation ---
def create_feature_matrix(data: pd.DataFrame) -> Optional[csr_matrix]:
    """
    [source: 68] Creates a combined feature matrix from text and numerical data using TF-IDF, CountVectorizer, and Scaling.
    [source: 69] Returns the sparse matrix or None if data is invalid.
    """
    # [source: 69] Check if input data is valid
    if data is None or data.empty:
        logging.error("Cannot create feature matrix: Input data is None or empty.")
        return None
    try:
        # [source: 69] TF-IDF Vectorizer for movie summaries (captures importance of words)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['Summary']) # Uses Summary column

        # [source: 69] Combine Cast, Director, Writers into a single 'tags' string per movie
        # [source: 70] Using fillna('') here again just in case preprocessing missed something
        data['tags'] = data['Cast'].fillna('') + ' ' + data['Director'].fillna('') + ' ' + data['Writers'].fillna('')

        # [source: 70] Count Vectorizer for tags (captures frequency of names/keywords)
        # [source: 70] Adjusted token pattern to better capture multi-word names/phrases
        count_vectorizer = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r"(?u)\b\w[\w\s-]+\b")
        tags_matrix = count_vectorizer.fit_transform(data['tags'])

        # [source: 96] Scale Numerical Features (Year, Runtime, Rating) using MinMaxScaler
        scaler = MinMaxScaler()
        numerical_features = data[['Year', 'Runtime', 'Rating']].copy() # Ensure copy

        # [source: 96] Fill potential NaNs in numerical features before scaling (should be handled, but safe)
        numerical_features.fillna(numerical_features.mean(), inplace=True)
        scaled_numerical = scaler.fit_transform(numerical_features)
        # [source: 96] Convert scaled numerical data to a sparse matrix format
        numerical_matrix = csr_matrix(scaled_numerical)

        # [source: 96] Combine matrices with weights (adjust weights based on importance)
        w_text = 0.5 # Weight for summary TF-IDF
        w_tags = 0.3 # Weight for cast/crew tags
        w_numeric = 0.2 # Weight for numerical features

        # [source: 96] Horizontally stack the weighted sparse matrices
        feature_matrix_combined = hstack([
            tfidf_matrix * w_text,
            tags_matrix * w_tags,
            numerical_matrix * w_numeric
        ]).tocsr() # Convert to CSR format for efficient calculations

        logging.info(f"Feature matrix created successfully. Shape: {feature_matrix_combined.shape}")
        # [source: 97] Return the combined sparse matrix
        return feature_matrix_combined
    # [source: 97] Catch any exceptions during matrix creation
    except Exception as e:
        logging.error(f"Error creating feature matrix: {e}", exc_info=True)
        return None

# --- Load Data and Create Matrix on Startup ---
# [source: 97] Load data and create the matrix when the app starts
movie_data = load_and_preprocess_data(DATA_FILE)
feature_matrix = None # Initialize feature_matrix
if movie_data is not None:
    # [source: 97] Set the 'id' column as the index for easy .loc lookups
    # Keep 'id' also as a column if needed elsewhere (e.g., in JSON responses)
    movie_data.set_index('id', inplace=True, drop=False)
    # Call create_feature_matrix directly (no caching needed here)
    feature_matrix = create_feature_matrix(movie_data)
else:
    # [source: 97] Log critical error if data loading fails
    logging.error("CRITICAL: Failed to load movie data. Recommendation features will be unavailable.")

# --- TMDB API Helper ---
# [source: 98] Function to get detailed movie info from TMDB API
@lru_cache(maxsize=100) # Cache results for TMDB API calls
def get_movie_details_from_tmdb(movie_title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    [source: 98] Fetches detailed movie info (poster, trailer, cast, genres, etc.) from TMDB.
    """
    # [source: 98] Check if TMDB API key is configured
    if not TMDB_API_KEY:
        logging.warning("TMDB_API_KEY not configured. Cannot fetch TMDB details.")
        return None
    try:
        # [source: 98] Search for the movie by title and optionally year
        search_url = f"{TMDB_BASE_URL}/search/movie"
        search_params = {'api_key': TMDB_API_KEY, 'query': movie_title}
        if year:
            try:
                search_params['year'] = int(year)
            except ValueError:
                logging.warning(f"Invalid year format for TMDB search: {year}")

        # [source: 98] Make the search request with a timeout
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        search_results = search_response.json()

        # [source: 98] Check if any results were found
        if not search_results.get('results'):
            logging.warning(f"No results found on TMDB for '{movie_title}' (Year: {year})")
            return None

        # [source: 98] Get the TMDB ID of the first search result
        tmdb_id = search_results['results'][0]['id']

        # [source: 99] Fetch details, videos, and credits using the movie ID in one call
        details_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        details_params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'append_to_response': 'videos,credits' # Append videos and credits to the main details request
        }
        details_response = requests.get(details_url, params=details_params, timeout=15)
        details_response.raise_for_status()
        details_data = details_response.json()

        # --- Extract required information ---
        # [source: 173] Extract poster path
        poster_path = details_data.get('poster_path')
        # [source: 173] Find the first YouTube trailer key if available
        trailer_key = next((v.get('key') for v in details_data.get('videos', {}).get('results', []) if v.get('type') == 'Trailer' and v.get('site') == 'YouTube'), None)
        # [source: 173] Extract genre names
        genres = [g.get('name') for g in details_data.get('genres', []) if g.get('name')]

        # [source: 173] Extract top 10 cast members with their profile pictures
        cast_list = []
        if 'credits' in details_data:
            for actor in details_data['credits'].get('cast', [])[:10]: # Limit to top 10
                profile_path = actor.get('profile_path')
                cast_list.append({
                    'name': actor.get('name'),
                    'character': actor.get('character'),
                    # [source: 173] Construct full profile image URL
                    'profile_path': f"https://image.tmdb.org/t/p/w185{profile_path}" if profile_path else None
                })

        # [source: 173] Return a dictionary with extracted details
        return {
            'tmdb_poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
            'trailer_key': trailer_key,
            'genres': genres,
            'tmdb_id': tmdb_id,
            'overview': details_data.get('overview'),
            'release_date': details_data.get('release_date'),
            'status': details_data.get('status'),
            'original_language': details_data.get('original_language'),
            # [source: 173] Use vote_average and vote_count from TMDB
            'tmdb_rating': details_data.get('vote_average'),
            'tmdb_votes': details_data.get('vote_count'),
            'cast': cast_list
        }
    # [source: 173] Handle network-related errors
    except requests.exceptions.RequestException as e:
        logging.error(f"TMDB API request failed for '{movie_title}': {e}")
        return None
    # [source: 173] Handle any other exceptions during TMDB data processing
    except Exception as e:
        logging.error(f"Error processing TMDB data for '{movie_title}': {e}", exc_info=True)
        return None

# --- Recommendation Logic ---
# [source: 174] Function to get recommendations and fetch details
def get_recommendations_with_details(
    title: str,
    data: pd.DataFrame,
    feature_matrix_combined: csr_matrix,
    top_n: int = 10
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    [source: 174] Finds similar movies using cosine similarity and fetches their details.
    Args:
        title: The title of the movie to get recommendations for.
    [source: 175]    data: The original DataFrame (indexed by 'id').
    [source: 176]    feature_matrix_combined: The precomputed combined feature matrix.
    [source: 176]    top_n: The number of recommendations to return.
    Returns:
    [source: 177]    A tuple containing (list of recommendation dicts, error message string or None).
    """
    # [source: 178] Check if recommendation engine components are ready
    if data is None or feature_matrix_combined is None:
        return None, "Recommendation engine not ready. Please try again later."
    try:
        # [source: 178] Find the index (id) of the input movie in the DataFrame (case-insensitive)
        matching_indices = data[data['Title'].str.lower() == title.lower()].index
        if matching_indices.empty:
            logging.warning(f"Movie '{title}' not found in the dataset for recommendation.")
            # [source: 178] Return empty list and specific error message if movie not found
            return [], f"Movie '{title}' not found in our database."
        # [source: 178] Use the first match's index (which should be the movie's 'id')
        idx = matching_indices[0]

        # [source: 179] Crucial Check: Ensure the found index is valid for the feature matrix shape
        if idx >= feature_matrix_combined.shape[0]:
            # This indicates a serious mismatch between the DataFrame and the feature matrix
            logging.error(f"Index mismatch: Movie index {idx} out of bounds for feature matrix shape {feature_matrix_combined.shape[0]}.")
            return None, "Internal error: Movie index mismatch. Cannot generate recommendations."

        # [source: 179] Calculate cosine similarity between the input movie's vector and all others
        # .flatten() converts the result to a 1D array
        cosine_sim = cosine_similarity(feature_matrix_combined[idx], feature_matrix_combined).flatten()
        # [source: 179] Create a list of (index, similarity_score) tuples
        sim_scores = list(enumerate(cosine_sim))
        # [source: 179] Sort movies based on similarity score in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # [source: 180] Get indices (ids) of top N similar movies
        # Note: sim_scores uses positional index (0 to n-1), data is indexed by 'id'
        # We need to map positional index back to 'id' from the original DataFrame's index
        # [source: 180] Get the list of 'id's (original index values) from the DataFrame
        df_indices = data.index.tolist()
        recommended_ids = []
        # [source: 180] Iterate through sorted similarity scores, starting from the second element (to skip self)
        for i, score in sim_scores[1:]:
            # [source: 180] Ensure the positional index 'i' is valid within the bounds of df_indices
            if i < len(df_indices):
                movie_id = df_indices[i] # Get the actual movie 'id' corresponding to positional index 'i'
                # [source: 188] Double-check it's not the input movie itself (shouldn't be if we start from sim_scores[1])
                if movie_id != idx:
                    recommended_ids.append(movie_id)
            # [source: 188] Get a few extra recommendations initially in case some fail TMDB lookup later
            if len(recommended_ids) >= top_n + 5:
                break
        # [source: 188] Trim the list to the desired top_n count
        recommended_ids = recommended_ids[:top_n]

        if not recommended_ids:
            logging.warning(f"No similar movies found for '{title}' (excluding itself).")
            # [source: 188] Return empty list, no error message if no similar movies found
            return [], None

        # [source: 189] Get movie data (including TMDB details) for the recommended IDs
        recommendations = []
        for rec_id in recommended_ids:
            # [source: 189] Use the helper function to get combined details
            details = get_movie_data_with_details(rec_id)
            if details:
                # [source: 189] Select only the fields needed for the recommendation card display
                recommendations.append({
                    'id': int(details.get('id', -1)), # Ensure ID is int, default to -1 if missing
                    'Title': details.get('Title', 'N/A'),
                    'Year': details.get('Year', 'N/A'),
                    'Rating': details.get('Rating', 'N/A'), # Use CSV rating for cards
                    'poster_url': details.get('poster_url') # Gets combined poster URL
                })
        # [source: 189] Return the list of recommendation details and no error message
        return recommendations, None
    # [source: 189] Catch any exceptions during recommendation generation
    except Exception as e:
        logging.error(f"Error getting recommendations for '{title}': {e}", exc_info=True)
        return None, "An error occurred while generating recommendations."

# --- Helper Function to get combined movie data + TMDB details ---
# [source: 190] This function centralizes fetching movie data by ID and enriching it with TMDB details
def get_movie_data_with_details(movie_id: int) -> Optional[Dict[str, Any]]:
    """
    [source: 190] Safely retrieves movie data by its ID (DataFrame index) and fetches/merges TMDB details.
    """
    if movie_data is None:
        logging.error("Movie data not loaded, cannot get details.")
        return None
    try:
        # [source: 190] Check if the movie ID exists in the DataFrame index
        if movie_id not in movie_data.index:
            logging.warning(f"Invalid movie ID requested: {movie_id}")
            return None

        # [source: 190] Retrieve the movie row using the ID (which is the index) and convert to dictionary
        movie = movie_data.loc[movie_id].to_dict()

        # [source: 190] Fetch details from TMDB using the movie title and year
        tmdb_details = get_movie_details_from_tmdb(movie['Title'], movie.get('Year'))

        # [source: 191] Combine data, prioritizing TMDB details where available
        combined_details = movie.copy() # Start with data from CSV

        if tmdb_details:
            # Override or add TMDB fields if they exist
            combined_details['overview'] = tmdb_details.get('overview', movie.get('Summary')) # Use TMDB overview if present
            combined_details['genres'] = tmdb_details.get('genres') # TMDB genres (list or None)
            combined_details['release_date'] = tmdb_details.get('release_date')
            combined_details['status'] = tmdb_details.get('status')
            combined_details['original_language'] = tmdb_details.get('original_language')
            combined_details['cast'] = tmdb_details.get('cast') # TMDB cast (list or None)
            combined_details['trailer_key'] = tmdb_details.get('trailer_key') # YouTube key or None
            combined_details['tmdb_rating'] = tmdb_details.get('tmdb_rating')
            combined_details['tmdb_votes'] = tmdb_details.get('tmdb_votes')
            # [source: 191] Use TMDB poster if available, otherwise keep CSV poster
            combined_details['poster_url'] = tmdb_details.get('tmdb_poster_url', movie.get('Movie Poster'))
        else:
            # [source: 191] Ensure essential fields exist even if TMDB lookup fails
            combined_details['poster_url'] = movie.get('Movie Poster') # Use CSV poster
            combined_details.setdefault('overview', movie.get('Summary'))
            combined_details.setdefault('genres', []) # Default to empty list if not found
            combined_details.setdefault('cast', []) # Default to empty list
            combined_details.setdefault('trailer_key', None) # Default to None

        # [source: 192] Always include the raw YouTube Trailer URL from the CSV if it exists
        # This provides a fallback if TMDB trailer_key parsing fails or isn't present
        youtube_url = movie.get('YouTube Trailer')
        # Attempt to extract video ID from various YouTube URL formats
        video_id = None
        if youtube_url:
             # Example extraction logic (adjust based on actual URL formats in CSV)
            if "youtube.com/watch?v=" in youtube_url:
                video_id = youtube_url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in youtube_url:
                 video_id = youtube_url.split("youtu.be/")[-1].split("?")[0]
            # Add more conditions if other URL formats exist in your CSV
        combined_details['csv_trailer_id'] = video_id # Store extracted ID from CSV URL

        # [source: 192] Ensure 'id' is present (it's the index value passed in)
        combined_details['id'] = movie_id
        return combined_details
    # [source: 192] Handle case where ID is valid but a column is missing (shouldn't happen with preprocessing)
    except KeyError as e:
        logging.warning(f"Movie ID {movie_id} found, but expected column missing: {e}")
        return None
    # [source: 192] Handle any other exceptions during data retrieval
    except Exception as e:
        logging.error(f"Error retrieving/combining movie data for ID {movie_id}: {e}", exc_info=True)
        return None

# --- Flask Routes ---

# [source: 192] Main index route
@app.route('/')
def index():
    """Renders the main page (index.html)."""
    return render_template('index.html')

# [source: 193] Route for movie title suggestions (autocomplete)
@app.route('/suggest_titles')
def suggest_titles():
    """Provides movie title suggestions for autocomplete."""
    # [source: 193] Get query parameters safely
    query = request.args.get('query', '').strip().lower()
    try:
        limit = int(request.args.get('limit', 8)) # Default limit of 8 suggestions
    except ValueError:
        limit = 8 # Fallback to default if limit is not a valid integer

    # [source: 193] Basic validation for the query
    if not query or len(query) < 2:
        return jsonify([]) # Return empty list for short/empty queries

    # [source: 193] Check if movie data is loaded
    if movie_data is None:
        logging.warning("Suggest titles called but movie data is not loaded.")
        return jsonify({"error": "Movie data not available"}), 503 # Service Unavailable
    try:
        # [source: 193] Perform case-insensitive 'contains' search on the 'Title' column
        # Use .unique().tolist() to get unique titles as a list
        matches = movie_data[movie_data['Title'].str.lower().str.contains(query, na=False)]['Title'].unique().tolist()
        # [source: 193] Sort matches by length (shorter often more relevant) and apply limit
        suggestions = sorted(matches, key=len)[:limit]
        return jsonify(suggestions)
    # [source: 193] Handle exceptions during suggestion generation
    except Exception as e:
        logging.error(f"Error generating suggestions for query '{query}': {e}", exc_info=True)
        return jsonify({"error": "Could not fetch suggestions"}), 500 # Internal Server Error

# [source: 194] Route to handle recommendation requests (POST from index page)
@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles the recommendation request from the main page form."""
    # [source: 194] Get movie title from form data safely
    movie_title = request.form.get('movie_title', '').strip()
    if not movie_title:
        return jsonify({"error": "Movie title is required."}), 400 # Bad Request

    logging.info(f"Recommendation requested for: {movie_title}")

    # [source: 194] Check if recommendation engine components are ready
    if movie_data is None or feature_matrix is None:
        logging.error("Recommendation requested but engine not ready.")
        return jsonify({"error": "Recommendation engine not ready. Please try again later."}), 503

    try:
        # [source: 194] --- Find the input movie's data first to display it ---
        input_movie_display_data = None
        # [source: 194] Find matches for the input title (case-insensitive)
        input_movie_matches = movie_data[movie_data['Title'].str.lower() == movie_title.lower()]
        if not input_movie_matches.empty:
            input_movie_id = input_movie_matches.index[0] # Get the 'id' (which is the DataFrame index)
            # [source: 194] Fetch combined details for the input movie
            input_movie_full_data = get_movie_data_with_details(input_movie_id)
            if input_movie_full_data:
                # [source: 194] Select specific fields needed for the brief display on index.html
                input_movie_display_data = {
                    'id': int(input_movie_full_data.get('id', -1)),
                    'title': input_movie_full_data.get('Title', 'N/A'),
                    'summary': input_movie_full_data.get('overview'), # Prefer TMDB overview
                    'year': input_movie_full_data.get('Year', 'N/A'),
                    'rating': input_movie_full_data.get('Rating', 'N/A'), # Use CSV rating here
                    'poster_url': input_movie_full_data.get('poster_url') # Use combined poster URL
                }
            else:
                logging.warning(f"Could not retrieve full details for input movie ID: {input_movie_id}")
        else:
             logging.warning(f"Input movie title '{movie_title}' not found in dataset.")


        # [source: 195] --- Get recommendations based on the title ---
        recommendations, error = get_recommendations_with_details(
            movie_title, movie_data, feature_matrix, top_n=15 # Get 15 recommendations
        )

        # [source: 195] Handle errors during recommendation generation
        if error and recommendations is None:
            # [source: 195] Critical error during recommendation
            logging.error(f"Critical error generating recommendations for '{movie_title}': {error}")
            return jsonify({"error": error, "input_movie": input_movie_display_data}), 500
        elif error:
            # [source: 195] Non-critical error (e.g., movie not found), recommendations might be empty
            logging.warning(f"Non-critical error/issue generating recommendations for '{movie_title}': {error}")
            # Return the specific error message along with empty recommendations
            return jsonify({"error": error, "recommendations": [], "input_movie": input_movie_display_data})

        # [source: 195] Return the input movie details and the recommendations
        return jsonify({"recommendations": recommendations, "input_movie": input_movie_display_data})

    # [source: 195] Catch any unexpected exceptions during the recommendation process
    except Exception as e:
        logging.error(f"Unexpected error in /recommend route for '{movie_title}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# [source: 195] Route for displaying movie details
@app.route('/movie/<int:movie_id>')
def movie_details(movie_id: int):
    """Renders the movie details page."""
    logging.info(f"Requesting details for movie ID: {movie_id}")
    # [source: 196] Check if movie data is available
    if movie_data is None:
        flash("Movie data is currently unavailable. Please try again later.", "danger")
        return redirect(url_for('index'))

    # [source: 196] Fetch combined details for the selected movie
    movie = get_movie_data_with_details(movie_id)

    # [source: 196] Handle case where the movie ID is not found or details couldn't be fetched
    if movie is None:
        flash(f"Movie with ID {movie_id} not found or could not be loaded.", "warning")
        logging.warning(f"Movie details not found for ID: {movie_id}")
        return redirect(url_for('index')) # Redirect back to the index page

    # [source: 196] Generate recommendations for the movie being viewed
    recommendations, error = get_recommendations_with_details(
        movie['Title'], movie_data, feature_matrix, top_n=10 # Get 10 recommendations for details page
    )

    # [source: 196] Log any errors during recommendation generation for the details page
    if error:
        logging.warning(f"Error getting recommendations on details page for '{movie['Title']}': {error}")
        flash(f"Could not load recommendations: {error}", "warning") # Show a message but still load the page
        recommendations = [] # Provide an empty list to the template

    # [source: 197] Render the movie_details.html template with movie data and recommendations
    # Pass the combined movie details and the list of recommendation dicts
    return render_template('movie_details.html', movie=movie, recommendations=recommendations or [])


# --- API Endpoints for Dynamic Content ---

# [source: 197] API endpoint to get popular movies
@app.route('/api/movies/popular')
def get_popular_movies():
    """API endpoint to get popular movies (e.g., based on rating)."""
    # [source: 197] Check if movie data is loaded
    if movie_data is None:
        return jsonify({"error": "Movie data not available"}), 503

    try:
        # [source: 197] Get limit from query args, default to 20, max 50
        limit = min(int(request.args.get('limit', 20)), 50)
        # [source: 197] Sort movies by 'Rating' (descending) and take the top 'limit'
        # FIX: Removed .reset_index()
        popular = movie_data.nlargest(limit, 'Rating')

        # [source: 197] Fetch details (mainly poster) for each popular movie
        results = []
        # FIX: Iterate through the index and rows of the 'popular' DataFrame
        for movie_id, movie_row in popular.iterrows():
            # Use get_movie_data_with_details to get the best poster URL
            # Pass the movie_id (which is the index)
            details = get_movie_data_with_details(movie_id)
            if details:
                 # Select fields needed for display cards
                results.append({
                    'id': int(details.get('id', -1)),
                    'Title': details.get('Title', 'N/A'),
                    'Year': details.get('Year', 'N/A'),
                    'Rating': details.get('Rating', 'N/A'), # Use CSV rating
                    'poster_url': details.get('poster_url')
                })
        # [source: 197] Return the list of popular movies as JSON
        return jsonify(results)
    # [source: 197] Handle potential errors during fetching popular movies
    except Exception as e:
        logging.error(f"Error fetching popular movies: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch popular movies"}), 500

# [source: 198] API endpoint to get trending movies (can be similar to popular or use another metric)
@app.route('/api/movies/trending')
def get_trending_movies():
    """API endpoint to get trending movies (using recent high ratings or random selection)."""
    # [source: 198] Check if movie data is loaded
    if movie_data is None:
        return jsonify({"error": "Movie data not available"}), 503

    try:
        # [source: 198] Get limit, default 20, max 50
        limit = min(int(request.args.get('limit', 20)), 50)
        # [source: 198] Simple trending: Filter recent years (e.g., last 10 years) and sort by rating
        current_year = pd.Timestamp.now().year
        recent_movies = movie_data[movie_data['Year'] >= current_year - 10]
        # [source: 198] If too few recent movies, fall back to random sampling
        if len(recent_movies) < limit:
             # FIX: Removed .reset_index()
            trending = movie_data.sample(n=min(limit, len(movie_data))) # Sample from all data
        else:
             # FIX: Removed .reset_index()
            trending = recent_movies.nlargest(limit, 'Rating')

        # [source: 198] Fetch details (mainly poster) for each trending movie
        results = []
        # FIX: Iterate through the index and rows of the 'trending' DataFrame
        for movie_id, movie_row in trending.iterrows():
            # Pass the movie_id (which is the index)
            details = get_movie_data_with_details(movie_id)
            if details:
                 results.append({
                    'id': int(details.get('id', -1)),
                    'Title': details.get('Title', 'N/A'),
                    'Year': details.get('Year', 'N/A'),
                    'Rating': details.get('Rating', 'N/A'),
                    'poster_url': details.get('poster_url')
                })
        # [source: 198] Return the list of trending movies
        return jsonify(results)
    # [source: 198] Handle errors
    except Exception as e:
        logging.error(f"Error fetching trending movies: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch trending movies"}), 500

# [source: 199] API endpoint to get latest movie trailers (using TMDB or fallback)
@app.route('/api/movies/trailers')
def get_latest_trailers():
    """API endpoint to get movies with trailers."""
    # [source: 199] Check if movie data is loaded
    if movie_data is None:
        return jsonify({"error": "Movie data not available"}), 503

    try:
        # [source: 199] Get limit, default 10, max 30
        limit = min(int(request.args.get('limit', 10)), 30)
        # [source: 199] Simple approach: Get recent, highly rated movies and check if they have a trailer key
        candidates = movie_data.nlargest(limit * 3, 'Rating') # Get more candidates

        results = []
        for movie_id in candidates.index:
            details = get_movie_data_with_details(movie_id)
            # [source: 199] Check if either TMDB trailer key or extracted CSV trailer ID exists
            if details and (details.get('trailer_key') or details.get('csv_trailer_id')):
                 # Select fields needed for display
                 results.append({
                    'id': int(details.get('id', -1)),
                    'Title': details.get('Title', 'N/A'),
                    'Year': details.get('Year', 'N/A'),
                    'poster_url': details.get('poster_url'),
                    # [source: 199] Prefer TMDB key, fallback to CSV extracted ID
                    'trailer_key': details.get('trailer_key') or details.get('csv_trailer_id')
                })
            # [source: 199] Stop once we have enough results
            if len(results) >= limit:
                break
        # [source: 199] Return movies with trailers
        return jsonify(results)
    # [source: 199] Handle errors
    except Exception as e:
        logging.error(f"Error fetching latest trailers: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch trailers"}), 500

# --- Error Handlers ---
# [source: 202] Custom error handler for 404 Not Found errors
@app.errorhandler(404)
def not_found_error(error):
    """Handles 404 errors."""
    logging.warning(f"404 Not Found error at {request.url}: {error}")
    # FIX: Return JSON response instead of rendering a template
    return jsonify(error=f"Not Found: {request.url}"), 404

# [source: 202] Custom error handler for 500 Internal Server errors
@app.errorhandler(500)
def internal_error(error):
    """Handles 500 internal server errors."""
    # [source: 202] Log the error traceback
    logging.error(f"500 Internal Server Error: {error}", exc_info=True)
    # FIX: Return JSON response instead of rendering a template
    return jsonify(error="Internal Server Error"), 500


# --- Main Execution ---
# [source: 202] Entry point for running the Flask application
if __name__ == '__main__':
    # [source: 202] Run the Flask development server
    # Debug=True should ONLY be used in development, not production!
    # Use host='0.0.0.0' to make the server accessible externally (e.g., within a Docker container or local network)
    # Use port=int(...) to ensure port is an integer if read from env var
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(host='0.0.0.0', port=port, debug=debug_mode)