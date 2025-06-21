import pandas as pd
import gradio as gr
import json
import random
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# LLM Configuration
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key='###',  # Replace with your actual API key
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024,
    temperature=0.9
)


class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.load_dataset()
        self.prepare_content_based_features()

    def create_sample_dataset(self):
        """Create a comprehensive sample movie dataset"""
        movies_data = [
            {
                "movie_id": 1,
                "title": "The Shawshank Redemption",
                "year": 1994,
                "genre": "Drama",
                "director": "Frank Darabont",
                "cast": "Tim Robbins, Morgan Freeman",
                "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "rating": 9.3,
                "runtime": 142,
                "language": "English",
                "country": "USA",
                "keywords": "prison, friendship, hope, redemption, drama"
            },
            {
                "movie_id": 2,
                "title": "The Godfather",
                "year": 1972,
                "genre": "Crime, Drama",
                "director": "Francis Ford Coppola",
                "cast": "Marlon Brando, Al Pacino",
                "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "rating": 9.2,
                "runtime": 175,
                "language": "English",
                "country": "USA",
                "keywords": "mafia, family, crime, power, loyalty"
            },
            {
                "movie_id": 3,
                "title": "The Dark Knight",
                "year": 2008,
                "genre": "Action, Crime, Drama",
                "director": "Christopher Nolan",
                "cast": "Christian Bale, Heath Ledger",
                "plot": "Batman begins his war on crime with his first major enemy being the clownishly homicidal Joker.",
                "rating": 9.0,
                "runtime": 152,
                "language": "English",
                "country": "USA",
                "keywords": "superhero, batman, joker, crime, action"
            },
            {
                "movie_id": 4,
                "title": "Pulp Fiction",
                "year": 1994,
                "genre": "Crime, Drama",
                "director": "Quentin Tarantino",
                "cast": "John Travolta, Uma Thurman, Samuel L. Jackson",
                "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
                "rating": 8.9,
                "runtime": 154,
                "language": "English",
                "country": "USA",
                "keywords": "crime, violence, non-linear, dialogue, cool"
            },
            {
                "movie_id": 5,
                "title": "Inception",
                "year": 2010,
                "genre": "Action, Sci-Fi, Thriller",
                "director": "Christopher Nolan",
                "cast": "Leonardo DiCaprio, Marion Cotillard",
                "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
                "rating": 8.8,
                "runtime": 148,
                "language": "English",
                "country": "USA",
                "keywords": "dreams, reality, heist, mind-bending, sci-fi"
            },
            {
                "movie_id": 6,
                "title": "Forrest Gump",
                "year": 1994,
                "genre": "Drama, Romance",
                "director": "Robert Zemeckis",
                "cast": "Tom Hanks, Robin Wright",
                "plot": "The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75.",
                "rating": 8.8,
                "runtime": 142,
                "language": "English",
                "country": "USA",
                "keywords": "life, love, history, simple, heartwarming"
            },
            {
                "movie_id": 7,
                "title": "The Matrix",
                "year": 1999,
                "genre": "Action, Sci-Fi",
                "director": "The Wachowskis",
                "cast": "Keanu Reeves, Laurence Fishburne",
                "plot": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
                "rating": 8.7,
                "runtime": 136,
                "language": "English",
                "country": "USA",
                "keywords": "virtual reality, hacker, philosophy, action, cyberpunk"
            },
            {
                "movie_id": 8,
                "title": "Goodfellas",
                "year": 1990,
                "genre": "Biography, Crime, Drama",
                "director": "Martin Scorsese",
                "cast": "Robert De Niro, Ray Liotta, Joe Pesci",
                "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife and his partners.",
                "rating": 8.7,
                "runtime": 146,
                "language": "English",
                "country": "USA",
                "keywords": "mafia, crime, biography, violence, loyalty"
            },
            {
                "movie_id": 9,
                "title": "Interstellar",
                "year": 2014,
                "genre": "Adventure, Drama, Sci-Fi",
                "director": "Christopher Nolan",
                "cast": "Matthew McConaughey, Anne Hathaway",
                "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
                "rating": 8.6,
                "runtime": 169,
                "language": "English",
                "country": "USA",
                "keywords": "space, time, love, science, survival"
            },
            {
                "movie_id": 10,
                "title": "The Lion King",
                "year": 1994,
                "genre": "Animation, Adventure, Drama",
                "director": "Roger Allers, Rob Minkoff",
                "cast": "Matthew Broderick, Jeremy Irons, James Earl Jones",
                "plot": "A Lion cub crown prince is tricked by a treacherous uncle into thinking he caused his father's death.",
                "rating": 8.5,
                "runtime": 88,
                "language": "English",
                "country": "USA",
                "keywords": "disney, animation, family, coming-of-age, africa"
            },
            {
                "movie_id": 11,
                "title": "Parasite",
                "year": 2019,
                "genre": "Comedy, Drama, Thriller",
                "director": "Bong Joon Ho",
                "cast": "Song Kang-ho, Lee Sun-kyun",
                "plot": "A poor family schemes to become employed by a wealthy family by posing as unrelated, highly qualified individuals.",
                "rating": 8.6,
                "runtime": 132,
                "language": "Korean",
                "country": "South Korea",
                "keywords": "class, society, thriller, dark comedy, korean"
            },
            {
                "movie_id": 12,
                "title": "Spirited Away",
                "year": 2001,
                "genre": "Animation, Adventure, Family",
                "director": "Hayao Miyazaki",
                "cast": "Rumi Hiiragi, Miyu Irino",
                "plot": "A girl enters a world ruled by gods, witches, and spirits, where humans are changed into beasts.",
                "rating": 8.6,
                "runtime": 125,
                "language": "Japanese",
                "country": "Japan",
                "keywords": "anime, fantasy, spirits, adventure, miyazaki"
            },
            {
                "movie_id": 13,
                "title": "Avengers: Endgame",
                "year": 2019,
                "genre": "Action, Adventure, Drama",
                "director": "Anthony Russo, Joe Russo",
                "cast": "Robert Downey Jr., Chris Evans, Mark Ruffalo",
                "plot": "The Avengers assemble once more to reverse Thanos' actions and restore balance to the universe.",
                "rating": 8.4,
                "runtime": 181,
                "language": "English",
                "country": "USA",
                "keywords": "marvel, superhero, epic, time travel, finale"
            },
            {
                "movie_id": 14,
                "title": "Casablanca",
                "year": 1942,
                "genre": "Drama, Romance, War",
                "director": "Michael Curtiz",
                "cast": "Humphrey Bogart, Ingrid Bergman",
                "plot": "A cynical nightclub owner protects an old flame and her husband from Nazis in Morocco.",
                "rating": 8.5,
                "runtime": 102,
                "language": "English",
                "country": "USA",
                "keywords": "classic, romance, war, sacrifice, timeless"
            },
            {
                "movie_id": 15,
                "title": "Titanic",
                "year": 1997,
                "genre": "Drama, Romance",
                "director": "James Cameron",
                "cast": "Leonardo DiCaprio, Kate Winslet",
                "plot": "A seventeen-year-old aristocrat falls in love with a poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
                "rating": 7.8,
                "runtime": 194,
                "language": "English",
                "country": "USA",
                "keywords": "romance, disaster, class, love, tragedy"
            }
        ]
        return pd.DataFrame(movies_data)

    def load_dataset(self):
        """Load or create the movie dataset"""
        try:
            # Try to load from CSV if exists
            self.movies_df = pd.read_csv('movies_dataset.csv')
        except FileNotFoundError:
            # Create sample dataset
            self.movies_df = self.create_sample_dataset()
            # Save to CSV for future use
            self.movies_df.to_csv('movies_dataset.csv', index=False)

    def prepare_content_based_features(self):
        """Prepare features for content-based recommendations"""
        # Combine relevant text features
        self.movies_df['combined_features'] = (
                self.movies_df['genre'] + ' ' +
                self.movies_df['director'] + ' ' +
                self.movies_df['cast'] + ' ' +
                self.movies_df['keywords'] + ' ' +
                self.movies_df['plot']
        )

        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['combined_features'])

    def get_content_based_recommendations(self, movie_title: str, num_recommendations: int = 5) -> List[Dict]:
        """Get content-based recommendations for a given movie"""
        try:
            # Find the movie index
            movie_idx = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index[0]

            # Calculate cosine similarity
            movie_vector = self.tfidf_matrix[movie_idx]
            similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()

            # Get top similar movies (excluding the input movie)
            similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]

            recommendations = []
            for idx in similar_indices:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie['title'],
                    'year': movie['year'],
                    'genre': movie['genre'],
                    'rating': movie['rating'],
                    'similarity_score': similarity_scores[idx]
                })

            return recommendations
        except IndexError:
            return []

    def get_llm_recommendations(self, user_preferences: str, num_recommendations: int = 5) -> str:
        """Get LLM-based recommendations based on user preferences"""

        # Create movie database string for context
        movies_context = ""
        for _, movie in self.movies_df.iterrows():
            movies_context += f"Title: {movie['title']} ({movie['year']}) | Genre: {movie['genre']} | Rating: {movie['rating']} | Plot: {movie['plot'][:100]}...\n"

        system_message = SystemMessage(content=f"""
        You are an expert movie recommendation system. Based on the user's preferences and the following movie database, 
        recommend {num_recommendations} movies that best match their interests.

        Movie Database:
        {movies_context}

        Provide recommendations in this exact JSON format:
        {{
            "recommendations": [
                {{
                    "title": "Movie Title",
                    "reason": "Detailed explanation why this movie matches user preferences"
                }}
            ]
        }}
        """)

        human_message = HumanMessage(content=f"User preferences: {user_preferences}")

        try:
            response = llm([system_message, human_message])
            return response.content
        except Exception as e:
            return f"Error getting LLM recommendations: {str(e)}"

    def get_genre_based_recommendations(self, preferred_genres: str, num_recommendations: int = 5) -> List[Dict]:
        """Get recommendations based on preferred genres"""
        genre_list = [g.strip().lower() for g in preferred_genres.split(',')]

        # Filter movies that match any of the preferred genres
        filtered_movies = self.movies_df[
            self.movies_df['genre'].str.lower().str.contains('|'.join(genre_list), na=False)
        ]

        # Sort by rating and get top recommendations
        top_movies = filtered_movies.nlargest(num_recommendations, 'rating')

        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append({
                'title': movie['title'],
                'year': movie['year'],
                'genre': movie['genre'],
                'rating': movie['rating'],
                'plot': movie['plot']
            })

        return recommendations

    def search_movies(self, query: str) -> List[Dict]:
        """Search movies by title, genre, director, or cast"""
        query_lower = query.lower()

        # Search in multiple columns
        mask = (
                self.movies_df['title'].str.lower().str.contains(query_lower, na=False) |
                self.movies_df['genre'].str.lower().str.contains(query_lower, na=False) |
                self.movies_df['director'].str.lower().str.contains(query_lower, na=False) |
                self.movies_df['cast'].str.lower().str.contains(query_lower, na=False)
        )

        results = self.movies_df[mask]

        search_results = []
        for _, movie in results.iterrows():
            search_results.append({
                'title': movie['title'],
                'year': movie['year'],
                'genre': movie['genre'],
                'director': movie['director'],
                'rating': movie['rating'],
                'plot': movie['plot']
            })

        return search_results


# Initialize the recommendation system
rec_system = MovieRecommendationSystem()


def format_recommendations(recommendations: List[Dict], rec_type: str) -> str:
    """Format recommendations for display"""
    if not recommendations:
        return "No recommendations found."

    formatted = f"## {rec_type} Recommendations\n\n"

    for i, rec in enumerate(recommendations, 1):
        formatted += f"**{i}. {rec['title']} ({rec.get('year', 'N/A')})**\n"
        formatted += f"- Genre: {rec.get('genre', 'N/A')}\n"
        formatted += f"- Rating: {rec.get('rating', 'N/A')}/10\n"

        if 'similarity_score' in rec:
            formatted += f"- Similarity Score: {rec['similarity_score']:.2f}\n"

        if 'reason' in rec:
            formatted += f"- Why recommended: {rec['reason']}\n"

        if 'plot' in rec:
            formatted += f"- Plot: {rec['plot'][:150]}...\n"

        formatted += "\n"

    return formatted


def get_content_recommendations(movie_title: str, num_recs: int) -> str:
    """Gradio interface function for content-based recommendations"""
    if not movie_title.strip():
        return "Please enter a movie title."

    recommendations = rec_system.get_content_based_recommendations(movie_title, num_recs)

    if not recommendations:
        available_movies = rec_system.movies_df['title'].tolist()
        return f"Movie '{movie_title}' not found in database.\n\nAvailable movies:\n" + "\n".join(available_movies)

    return format_recommendations(recommendations, "Content-Based")


def get_llm_recommendations_interface(user_preferences: str, num_recs: int) -> str:
    """Gradio interface function for LLM-based recommendations"""
    if not user_preferences.strip():
        return "Please enter your preferences."

    try:
        llm_response = rec_system.get_llm_recommendations(user_preferences, num_recs)

        # Try to parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                recommendations = json_data.get('recommendations', [])
                return format_recommendations(recommendations, "AI-Powered")
            else:
                return f"## AI-Powered Recommendations\n\n{llm_response}"
        except json.JSONDecodeError:
            return f"## AI-Powered Recommendations\n\n{llm_response}"

    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your API key configuration."


def get_genre_recommendations_interface(genres: str, num_recs: int) -> str:
    """Gradio interface function for genre-based recommendations"""
    if not genres.strip():
        return "Please enter preferred genres (comma-separated)."

    recommendations = rec_system.get_genre_based_recommendations(genres, num_recs)
    return format_recommendations(recommendations, "Genre-Based")


def search_movies_interface(query: str) -> str:
    """Gradio interface function for movie search"""
    if not query.strip():
        return "Please enter a search query."

    results = rec_system.search_movies(query)
    return format_recommendations(results, "Search Results")


def get_random_recommendations(num_recs: int) -> str:
    """Get random movie recommendations"""
    random_movies = rec_system.movies_df.sample(n=min(num_recs, len(rec_system.movies_df)))

    recommendations = []
    for _, movie in random_movies.iterrows():
        recommendations.append({
            'title': movie['title'],
            'year': movie['year'],
            'genre': movie['genre'],
            'rating': movie['rating'],
            'plot': movie['plot']
        })

    return format_recommendations(recommendations, "Random")


# Create Gradio Interface
def create_gradio_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title="🎬 Movie Recommendation System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎬 Movie Recommendation System

        Welcome to your personalized movie recommendation system! Choose from different recommendation methods below.
        """)

        with gr.Tabs():
            # Content-Based Recommendations Tab
            with gr.Tab("🎯 Similar Movies"):
                gr.Markdown("Find movies similar to one you already like!")

                with gr.Row():
                    movie_input = gr.Textbox(
                        label="Enter a movie title",
                        placeholder="e.g., The Shawshank Redemption",
                        lines=1
                    )
                    content_num_recs = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Number of recommendations"
                    )

                content_button = gr.Button("Get Similar Movies", variant="primary")
                content_output = gr.Markdown()

                content_button.click(
                    get_content_recommendations,
                    inputs=[movie_input, content_num_recs],
                    outputs=content_output
                )

            # LLM-Based Recommendations Tab
            with gr.Tab("🤖 AI Recommendations"):
                gr.Markdown("Get personalized recommendations based on your preferences using AI!")

                with gr.Row():
                    preferences_input = gr.Textbox(
                        label="Describe your movie preferences",
                        placeholder="e.g., I like action movies with great cinematography and complex plots",
                        lines=3
                    )
                    llm_num_recs = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Number of recommendations"
                    )

                llm_button = gr.Button("Get AI Recommendations", variant="primary")
                llm_output = gr.Markdown()

                llm_button.click(
                    get_llm_recommendations_interface,
                    inputs=[preferences_input, llm_num_recs],
                    outputs=llm_output
                )

            # Genre-Based Recommendations Tab
            with gr.Tab("🎭 Genre Favorites"):
                gr.Markdown("Discover top-rated movies in your favorite genres!")

                with gr.Row():
                    genre_input = gr.Textbox(
                        label="Enter preferred genres (comma-separated)",
                        placeholder="e.g., Action, Drama, Sci-Fi",
                        lines=1
                    )
                    genre_num_recs = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Number of recommendations"
                    )

                genre_button = gr.Button("Get Genre Recommendations", variant="primary")
                genre_output = gr.Markdown()

                genre_button.click(
                    get_genre_recommendations_interface,
                    inputs=[genre_input, genre_num_recs],
                    outputs=genre_output
                )

            # Search Tab
            with gr.Tab("🔍 Search Movies"):
                gr.Markdown("Search for movies by title, director, genre, or cast!")

                search_input = gr.Textbox(
                    label="Search query",
                    placeholder="e.g., Christopher Nolan, Action, Tom Hanks",
                    lines=1
                )

                search_button = gr.Button("Search Movies", variant="primary")
                search_output = gr.Markdown()

                search_button.click(
                    search_movies_interface,
                    inputs=search_input,
                    outputs=search_output
                )

            # Random Recommendations Tab
            with gr.Tab("🎲 Surprise Me"):
                gr.Markdown("Get random movie recommendations for those adventurous movie nights!")

                random_num_recs = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of random recommendations"
                )

                random_button = gr.Button("Get Random Movies", variant="primary")
                random_output = gr.Markdown()

                random_button.click(
                    get_random_recommendations,
                    inputs=random_num_recs,
                    outputs=random_output
                )

            # Dataset Info Tab
            with gr.Tab("📊 Dataset Info"):
                gr.Markdown("Information about the movie database")

                def show_dataset_info():
                    info = f"""
                    ## Dataset Information

                    **Total Movies:** {len(rec_system.movies_df)}

                    **Available Genres:** {', '.join(set([genre.strip() for genres in rec_system.movies_df['genre'] for genre in genres.split(',')]))}

                    **Year Range:** {rec_system.movies_df['year'].min()} - {rec_system.movies_df['year'].max()}

                    **Average Rating:** {rec_system.movies_df['rating'].mean():.1f}/10

                    **Sample Movies:**
                    """

                    sample_movies = rec_system.movies_df.sample(n=5)
                    for _, movie in sample_movies.iterrows():
                        info += f"\n- **{movie['title']}** ({movie['year']}) - {movie['genre']} - {movie['rating']}/10"

                    return info

                dataset_button = gr.Button("Show Dataset Info")
                dataset_output = gr.Markdown()

                dataset_button.click(
                    show_dataset_info,
                    outputs=dataset_output
                )

        gr.Markdown("""
        ---
        **Note:** 
        - Make sure to add your DeepSeek API key in the code
        - The system uses a sample dataset, but you can easily replace it with a larger dataset
        - For production use, consider using datasets from TMDB, IMDB, or MovieLens
        """)

    return interface


if __name__ == "__main__":
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )