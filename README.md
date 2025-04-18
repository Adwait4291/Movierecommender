# Movie Recommender System (Flask)

A web application built with Flask that recommends movies based on content similarity (summary, cast, director, etc.) using data from `Hydra-Movie-Scrape.csv` and enhances the experience by fetching posters, trailers, and additional details from The Movie Database (TMDB) API.

## Features

* **Content-Based Recommendations:** Suggests movies similar to a user's input based on TF-IDF of summaries, Count Vectorization of crew, and scaled numerical features (year, runtime, rating).
* **TMDB Integration:** Fetches movie posters (with fallback to CSV 'Movie Poster' column), trailers, genres, cast details, and more from TMDB.
* **Interactive UI:**
    * Autocomplete suggestions for movie titles.
    * Displays selected movie details and recommendations on the main page.
    * Horizontal scrolling rows for Top Rated, Trending (placeholder), Latest Releases, and People Also Watch (placeholder) movies.
    * Detailed movie page with poster, summary, cast, trailer background (if available), and further recommendations.
* **Basic User Authentication:** Simple login/register functionality (in-memory store - **NOT secure for production**).

## Technology Stack

* **Backend:** Python, Flask
* **Data Science:** Pandas, Scikit-learn, SciPy
* **Frontend:** HTML, CSS, JavaScript
* **API:** The Movie Database (TMDB)

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables:**
    * **TMDB API Key:** Obtain an API key from [TMDB](https://www.themoviedb.org/documentation/api). Set it as an environment variable `TMDB_API_KEY`.
        ```bash
        # Example (macOS/Linux):
        export TMDB_API_KEY='YOUR_ACTUAL_TMDB_API_KEY'
        # Example (Windows CMD):
        set TMDB_API_KEY=YOUR_ACTUAL_TMDB_API_KEY
        # Example (Windows PowerShell):
        $env:TMDB_API_KEY = 'YOUR_ACTUAL_TMDB_API_KEY'
        ```
        *(Alternatively, replace the placeholder in `app.py`, but environment variables are strongly recommended)*
    * **Flask Secret Key:** Set a strong, random secret key for session management as `FLASK_SECRET_KEY`.
        ```bash
        # Example (macOS/Linux):
        export FLASK_SECRET_KEY='generate_a_strong_random_key_here'
        # Example (Windows CMD):
        set FLASK_SECRET_KEY=generate_a_strong_random_key_here
        # Example (Windows PowerShell):
        $env:FLASK_SECRET_KEY = 'generate_a_strong_random_key_here'
        ```
        *(You can generate one in Python using `import os; os.urandom(24).hex()`)*

5.  **Ensure Data File:** Place the `Hydra-Movie-Scrape.csv` file in the same directory as `app.py`.

6.  **Static Files:** Ensure you have a `static` folder containing `style.css`, `script.js`, and potentially `img` or `fonts` subdirectories if your HTML/CSS references them (e.g., `placeholder.png`, `placeholder_person.png`, background images, custom fonts).

7.  **Run the Application (Development):**
    ```bash
    flask run
    # Or: python app.py
    ```
    The application will likely be running at `http://127.0.0.1:5001` (or the port specified in `app.py`).

8.  **Run the Application (Production):**
    Use a production-ready WSGI server like Gunicorn or Waitress.
    ```bash
    # Example using Waitress (cross-platform):
    pip install waitress
    waitress-serve --host=0.0.0.0 --port=5001 app:app

    # Example using Gunicorn (Linux/macOS):
    pip install gunicorn
    gunicorn --bind 0.0.0.0:5001 app:app
    ```

## How Recommendations Work

1.  **Data Loading & Preprocessing:** `Hydra-Movie-Scrape.csv` is loaded into a Pandas DataFrame. Data is cleaned (handling missing values, converting types), and an 'id' column (based on index) is added.
2.  **Feature Engineering:** Text (summary, crew) and numerical (year, runtime, rating) features are extracted and converted into a combined sparse matrix using TF-IDF, CountVectorizer, and MinMaxScaler. Weights are applied during combination.
3.  **Similarity Calculation:** Cosine similarity is calculated between the feature vector of the user's input movie and all other movies.
4.  **Details & Poster Fetching:** Top similar movies are identified. Details (including poster URLs) are fetched from TMDB API for both the input movie and recommendations. The system uses the TMDB poster if available, otherwise falls back to the URL in the 'Movie Poster' column of the CSV.
5.  **Display:** The input movie, recommendations, and other lists (Top Rated, etc.) are displayed on the frontend with posters and basic details. Clicking a movie leads to a detailed view.

## Important Notes

* **API Keys:** **Never commit your `TMDB_API_KEY` or `FLASK_SECRET_KEY` directly into your code or version control.** Use environment variables or a secure configuration method.
* **User Authentication:** The included login/register system is **for demonstration only and is insecure**. For any real application, implement proper authentication using a database (like PostgreSQL or MySQL) and secure libraries (e.g., Flask-Login, Flask-SQLAlchemy, Flask-Bcrypt).
* **Error Handling:** The application includes basic error handling and logging, but production applications require more comprehensive strategies.
* **Placeholders:** Ensure placeholder images (`placeholder.png`, `placeholder_person.png`) exist in your `static/img/` directory or update the paths in the code/CSS.
* **Scalability:** For larger datasets or high traffic, consider database integration, optimizing feature calculation, caching strategies, and potentially asynchronous task queues for API calls.
