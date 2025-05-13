import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import ast

# Custom UI Config
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    # Read raw CSVs
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')

    # Merge both dataframes on title
    df = movies_df.merge(credits_df, on='title')

    # Extract genres from JSON string
    def extract_genres(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return '|'.join([g['name'] for g in genres])
        except:
            return ''

    # Extract director from crew JSON string
    def extract_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name']
        except:
            return ''
        return ''

    # Extract top 3 cast members
    def extract_cast(cast_str):
        try:
            cast = ast.literal_eval(cast_str)
            return '|'.join([c['name'] for c in cast[:3]])
        except:
            return ''

    # Apply transformations
    df['genres'] = df['genres'].apply(extract_genres)
    df['director'] = df['crew'].apply(extract_director)
    df['cast'] = df['cast'].apply(extract_cast)

    # Keep only required columns
    df = df[['title', 'genres', 'director', 'cast', 'vote_average']].dropna()

    # Create combined metadata column for recommendations
    df['metadata'] = df['genres'] + ' ' + df['director'] + ' ' + df['cast']
    return df

df = load_data()

# Initialize TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

# Recommendation engine
def get_recommendations(selected_movies, selected_genres):
    indices = [df[df['title'] == movie].index[0] for movie in selected_movies]
    sim_scores = np.mean(cosine_similarity(tfidf_matrix[indices], tfidf_matrix), axis=0)
    top_indices = np.argsort(sim_scores)[-15:][::-1]
    recommendations = df.iloc[top_indices]

    if selected_genres:
        mask = recommendations['genres'].apply(lambda x: any(genre in x for genre in selected_genres))
        recommendations = recommendations[mask].head(10)

    return recommendations.head(10)

# Genre match visualization
def plot_genre_match(recommendations, selected_genres):
    matches = []
    for _, movie in recommendations.iterrows():
        movie_genres = set(movie['genres'].split('|'))
        overlap = len(movie_genres.intersection(selected_genres))
        match_percent = (overlap / len(selected_genres)) * 100
        matches.append(match_percent)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(recommendations['title'], matches, color='#4CAF50')
    ax.set_xlabel('Genre Match (%)', fontsize=12)
    ax.set_title('How Well These Match Your Taste', fontsize=14)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.0f}%', va='center')

    plt.tight_layout()
    st.pyplot(fig)

# UI Components
def main():
    st.markdown("""
    <style>
    .header {
        font-size: 40px !important;
        color: #FF4B4B !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 20px !important;
        color: #1F77B4 !important;
        margin-top: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="header">🍿 CineMatch AI</p>', unsafe_allow_html=True)
    st.markdown("### Discover Your Next Favorite Movie")

    # Step 1: Genre Selection
    st.markdown('<p class="subheader">Step 1: Choose Your Favorite Genres</p>', unsafe_allow_html=True)
    all_genres = sorted(set(g for genres in df['genres'].str.split('|') for g in genres))
    selected_genres = st.multiselect(
        "Select up to 3 genres (we'll find similar films):",
        all_genres,
        max_selections=3,
        key="genres"
    )

    # Step 2: Movie Selection
    if selected_genres:
        genre_movies = df[df['genres'].apply(lambda x: any(g in x for g in selected_genres))]
        st.markdown('<p class="subheader">Step 2: Pick 5 Movies You Love</p>', unsafe_allow_html=True)
        selected_movies = st.multiselect(
            "Select movies that match your taste (exactly 5 for best results):",
            genre_movies['title'].unique(),
            max_selections=5,
            key="movies"
        )

        if len(selected_movies) == 5:
            if st.button("🎯 Get Personalized Recommendations", type="primary"):
                with st.spinner('Finding your perfect matches...'):
                    recommendations = get_recommendations(selected_movies, selected_genres)

                    st.success("Here are your personalized recommendations!")
                    st.markdown("---")

                    # Top 3 movies with emphasis
                    cols = st.columns(3)
                    for i in range(3):
                        with cols[i]:
                            st.markdown(f"### #{i+1}: {recommendations.iloc[i]['title']}")
                            st.caption(f"**Genres:** {recommendations.iloc[i]['genres']}")
                            st.caption(f"**Director:** {recommendations.iloc[i]['director']}")
                            st.caption(f"**Rating:** ⭐ {recommendations.iloc[i]['vote_average']}/10")

                    st.markdown("---")
                    st.dataframe(
                        recommendations[['title', 'genres', 'director', 'vote_average']],
                        column_config={
                            "title": "Movie Title",
                            "genres": "Genres",
                            "director": "Director",
                            "vote_average": "Rating"
                        },
                        hide_index=True,
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.markdown("### How These Match Your Taste")
                    plot_genre_match(recommendations, set(selected_genres))
        elif len(selected_movies) > 0:
            st.warning("Please select exactly 5 movies for optimal recommendations")

if __name__ == "__main__":
    main()
