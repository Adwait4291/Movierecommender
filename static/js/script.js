// [source: 19] Wait for the DOM to be fully loaded before executing scripts
document.addEventListener('DOMContentLoaded', () => {
    // [source: 19] --- Element Selectors ---
    const recommendationForm = document.getElementById('recommendation-form');
    const movieSearchInput = document.getElementById('movie-search-input');
    const suggestionsBox = document.getElementById('suggestions-box');
    const popularList = document.getElementById('popular-movies-list');
    const trendingList = document.getElementById('trending-movies-list');
    const trailersList = document.getElementById('latest-trailers-list');
    const recommendationsSection = document.getElementById('recommendations-section');
    const recommendationsList = document.getElementById('recommendations-list');
    const inputMovieDisplay = document.getElementById('input-movie-display');
    const defaultMoviesSection = document.getElementById('default-movies-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');

    // [source: 19] --- Constants ---
    const API_BASE = '/api/movies'; // Base URL for movie API endpoints
    const SUGGEST_URL = '/suggest_titles'; // URL for title suggestions
    const RECOMMEND_URL = '/recommend'; // URL for getting recommendations

    // [source: 19] Debounce timer variable for suggestions
    let debounceTimer;

    // [source: 19] --- Helper Functions ---

    /**
     * [source: 20] Shows the loading indicator.
     */
    const showLoading = () => {
        if (loadingIndicator) loadingIndicator.style.display = 'block';
    };

    /**
     * [source: 20] Hides the loading indicator.
     */
    const hideLoading = () => {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
    };

    /**
    * [source: 20] Displays an error message to the user.
    * @param {string} message - The error message to display.
    */
    const displayError = (message) => {
        if (errorMessageDiv) {
            errorMessageDiv.textContent = message;
            errorMessageDiv.style.display = 'block';
            // Optionally hide after a few seconds
            setTimeout(() => {
                errorMessageDiv.style.display = 'none';
            }, 5000);
        }
        console.error('Error:', message); // Also log to console
    };

    /**
     * [source: 20] Fetches data from a given API endpoint and handles loading/errors.
     * @param {string} endpoint - The API endpoint URL.
     * @returns {Promise<any>} - A promise that resolves with the JSON data or rejects on error.
     */
    const fetchData = async (endpoint) => {
        showLoading();
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: `HTTP error! Status: ${response.status}` }));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            displayError(`Failed to fetch data from ${endpoint}: ${error.message}`);
            throw error; // Re-throw error to be caught by caller if needed
        } finally {
            hideLoading();
        }
    };

    /**
     * [source: 21] Creates an HTML movie card element.
     * @param {object} movie - The movie data object.
     * @param {boolean} isTrailer - Whether the card is for the trailer section.
     * @returns {HTMLElement} - The created movie item element.
     */
    const createMovieCard = (movie, isTrailer = false) => {
        const movieItem = document.createElement('div');
        movieItem.classList.add('movie-item');

        // Default placeholder image URL
        const placeholderImage = '/static/images/placeholder.png'; // Adjust path if needed

        // Use poster_url if available, otherwise use placeholder
        const posterSrc = movie.poster_url || placeholderImage;

        // Create link to movie details page
        const movieLink = document.createElement('a');
        // Check if movie.id is valid before creating the link
        if (movie.id && movie.id > 0) {
             movieLink.href = `/movie/${movie.id}`;
        } else {
            movieLink.href = '#'; // Fallback link if ID is invalid/missing
            console.warn(`Invalid or missing ID for movie: ${movie.Title}`);
        }


        // Image element
        const img = document.createElement('img');
        img.src = posterSrc;
        img.alt = `Poster for ${movie.Title || 'Unknown Movie'}`;
        img.loading = 'lazy'; // Lazy load images
        // Handle image loading errors by setting placeholder
        img.onerror = () => { img.src = placeholderImage; };
        movieLink.appendChild(img);

        // Movie info container
        const infoDiv = document.createElement('div');
        infoDiv.classList.add('movie-item-info');

        // Movie Title
        const title = document.createElement('p');
        title.classList.add('movie-title');
        title.textContent = movie.Title || 'N/A';
        infoDiv.appendChild(title);

        // Year and Rating (only if not a trailer card)
        if (!isTrailer) {
            const yearRating = document.createElement('p');
            yearRating.classList.add('movie-year-rating');
            let yearRatingText = movie.Year ? `<span>${movie.Year}</span>` : '<span>N/A</span>';
            if (movie.Rating) {
                // Ensure rating is a number and format it
                const ratingValue = typeof movie.Rating === 'number' ? movie.Rating.toFixed(1) : movie.Rating;
                yearRatingText += `<span> &bull; ${ratingValue}/10</span>`;
            }
            yearRating.innerHTML = yearRatingText;
            infoDiv.appendChild(yearRating);
        }

        movieLink.appendChild(infoDiv);
        movieItem.appendChild(movieLink);

        // Add "Watch Trailer" button for trailer cards
        if (isTrailer && movie.trailer_key) {
            const trailerButton = document.createElement('button');
            trailerButton.classList.add('watch-trailer-button');
            trailerButton.textContent = 'Watch Trailer';
            // Open trailer in a new tab or modal
            trailerButton.onclick = () => window.open(`https://www.youtube.com/watch?v=${movie.trailer_key}`, '_blank');
            movieItem.appendChild(trailerButton);
        }

        return movieItem;
    };

    /**
    * [source: 22] Creates HTML for displaying the input movie in the recommendations section.
    * @param {object} movie - The input movie data object.
    * @returns {string} - The HTML string for the input movie display.
    */
    const createInputMovieDisplay = (movie) => {
        if (!movie) return ''; // Return empty string if no movie data
        const placeholderImage = '/static/images/placeholder.png';
        const posterSrc = movie.poster_url || placeholderImage;
        // Format rating if it exists
        const ratingText = movie.rating ? `${Number(movie.rating).toFixed(1)}/10` : 'N/A';

        return `
            <h3>Recommendations based on:</h3>
            <div class="movie-item">
                <a href="/movie/${movie.id}" title="View details for ${movie.title}">
                    <img src="${posterSrc}" alt="Poster for ${movie.title}" loading="lazy" onerror="this.onerror=null;this.src='${placeholderImage}';">
                    <div class="movie-item-info">
                        <p class="movie-title">${movie.title || 'N/A'}</p>
                        <p class="movie-year-rating">
                            <span>${movie.year || 'N/A'}</span>
                            <span>&bull; ${ratingText}</span>
                        </p>
                    </div>
                </a>
                 ${movie.summary ? `<p class="input-movie-summary">${movie.summary.substring(0, 100)}...</p>` : ''}
            </div>
        `;
    };


    /**
     * [source: 22] Populates a container element with movie cards.
     * @param {HTMLElement} container - The container element to populate.
     * @param {Array<object>} movies - An array of movie data objects.
     * @param {boolean} isTrailerList - Indicates if the list is for trailers.
     */
    const populateMovieList = (container, movies, isTrailerList = false) => {
        if (!container) {
            console.error("Target container not found for populating movie list.");
            return;
        }
        container.innerHTML = ''; // Clear previous content
        if (!movies || movies.length === 0) {
            container.innerHTML = '<p>No movies found.</p>';
            return;
        }
        movies.forEach(movie => {
            const movieCard = createMovieCard(movie, isTrailerList);
            container.appendChild(movieCard);
        });
    };

    /**
     * [source: 23] Fetches and displays default movie lists (Popular, Trending, Trailers).
     */
    const loadDefaultMovies = async () => {
        try {
            // Fetch all in parallel
            const [popular, trending, trailers] = await Promise.all([
                fetchData(`${API_BASE}/popular?limit=15`),
                fetchData(`${API_BASE}/trending?limit=15`),
                fetchData(`${API_BASE}/trailers?limit=10`)
            ]);

            populateMovieList(popularList, popular);
            populateMovieList(trendingList, trending);
            populateMovieList(trailersList, trailers, true); // Mark as trailer list

        } catch (error) {
            console.error("Failed to load one or more default movie sections.", error);
            // Optionally display a more specific error to the user
            if (popularList) popularList.innerHTML = '<p>Could not load popular movies.</p>';
            if (trendingList) trendingList.innerHTML = '<p>Could not load trending movies.</p>';
            if (trailersList) trailersList.innerHTML = '<p>Could not load trailers.</p>';
        }
    };

    /**
     * [source: 23] Fetches movie title suggestions based on user input.
     * @param {string} query - The search query.
     */
    const fetchSuggestions = async (query) => {
        if (query.length < 2) { // Minimum length for suggestions
            suggestionsBox.innerHTML = '';
            suggestionsBox.style.display = 'none';
            return;
        }

        try {
            const response = await fetch(`${SUGGEST_URL}?query=${encodeURIComponent(query)}&limit=8`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const suggestions = await response.json();

            suggestionsBox.innerHTML = ''; // Clear previous suggestions
            if (suggestions.length > 0) {
                suggestions.forEach(title => {
                    const div = document.createElement('div');
                    div.textContent = title;
                    // When a suggestion is clicked, fill the input and submit the form
                    div.addEventListener('click', () => {
                        movieSearchInput.value = title;
                        suggestionsBox.style.display = 'none';
                        recommendationForm.requestSubmit(); // Programmatically submit the form
                    });
                    suggestionsBox.appendChild(div);
                });
                suggestionsBox.style.display = 'block';
            } else {
                suggestionsBox.style.display = 'none';
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            suggestionsBox.style.display = 'none'; // Hide on error
        }
    };

    /**
    * [source: 24] Handles the submission of the recommendation form.
    * @param {Event} event - The form submission event.
    */
    const handleRecommendationSubmit = async (event) => {
        event.preventDefault(); // Prevent default form submission
        const movieTitle = movieSearchInput.value.trim();
        if (!movieTitle) {
            displayError("Please enter a movie title.");
            return;
        }

        // Hide suggestions box
        suggestionsBox.style.display = 'none';
        // Clear previous error messages
        if (errorMessageDiv) errorMessageDiv.style.display = 'none';

        showLoading(); // Show loading indicator

        try {
            // Prepare form data for POST request
            const formData = new FormData();
            formData.append('movie_title', movieTitle);

            // Make the POST request to the backend /recommend endpoint
            const response = await fetch(RECOMMEND_URL, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle errors returned from the backend (e.g., movie not found, internal error)
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            // --- Process successful response ---

            // Hide default sections, show recommendations section
            if (defaultMoviesSection) defaultMoviesSection.style.display = 'none';
            if (recommendationsSection) recommendationsSection.style.display = 'block';

            // Display the input movie details
            if (inputMovieDisplay && data.input_movie) {
                inputMovieDisplay.innerHTML = createInputMovieDisplay(data.input_movie);
            } else if (inputMovieDisplay) {
                 inputMovieDisplay.innerHTML = ''; // Clear if no input movie data
            }

            // Populate the recommendations list
            populateMovieList(recommendationsList, data.recommendations);

        } catch (error) {
             console.error("Error getting recommendations:", error);
             displayError(`Could not get recommendations: ${error.message}`);
             // Optionally: Hide recommendations section and show defaults again on error
             if (recommendationsSection) recommendationsSection.style.display = 'none';
             if (defaultMoviesSection) defaultMoviesSection.style.display = 'block';
             if (inputMovieDisplay) inputMovieDisplay.innerHTML = ''; // Clear input movie display

        } finally {
            hideLoading(); // Hide loading indicator regardless of outcome
        }
    };


    // [source: 25] --- Event Listeners ---

    // Handle recommendation form submission
    if (recommendationForm) {
        recommendationForm.addEventListener('submit', handleRecommendationSubmit);
    }

    // Handle input in the search field for suggestions (with debouncing)
    if (movieSearchInput && suggestionsBox) {
        movieSearchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            clearTimeout(debounceTimer); // Clear existing timer
            // Set a new timer to fetch suggestions after 300ms
            debounceTimer = setTimeout(() => {
                fetchSuggestions(query);
            }, 300);
        });

        // Hide suggestions when clicking outside the input/suggestions box
        document.addEventListener('click', (e) => {
            if (!movieSearchInput.contains(e.target) && !suggestionsBox.contains(e.target)) {
                suggestionsBox.style.display = 'none';
            }
        });
         // Hide suggestions when input loses focus (unless clicking a suggestion)
         movieSearchInput.addEventListener('blur', () => {
             // Use a short delay to allow click event on suggestion to register first
             setTimeout(() => {
                 if (!suggestionsBox.matches(':hover')) { // Check if mouse is over suggestions
                     suggestionsBox.style.display = 'none';
                 }
             }, 150);
         });
    }

    // [source: 25] --- Initial Load ---
    // Load popular, trending movies and trailers when the page loads
    loadDefaultMovies();

}); // End DOMContentLoaded