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
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')

    df = movies_df.merge(credits_df, on='title')

    def extract_genres(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return '|'.join([g['name'] for g in genres])
        except:
            return ''

    def extract_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name']
        except:
            return ''
        return ''

    def extract_cast(cast_str):
        try:
            cast = ast.literal_eval(cast_str)
            return '|'.join([c['name'] for c in cast[:3]])
        except:
            return ''

    df['genres'] = df['genres'].apply(extract_genres)
    df['director'] = df['crew'].apply(extract_director)
    df['cast'] = df['cast'].apply(extract_cast)

    df = df[['title', 'genres', 'director', 'cast', 'vote_average']].dropna()
    df['metadata'] = df['genres'] + ' ' + df['director'] + ' ' + df['cast']

    return df

df = load_data()

# TF-IDF and similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

# Recommendation engine
def get_recommendations(selected_movies, selected_genres):
    indices = [df[df['title'] == movie].index[0] for movie in selected_movies]
    sim_scores = np.mean(cosine_similarity(tfidf_matrix[indices], tfidf_matrix), axis=0)
    top_indices = np.argsort(sim_scores)[-30:][::-1]
    recommendations = df.iloc[top_indices]

    if selected_genres:
        selected_genres = set([g.strip().lower() for g in selected_genres])
        recommendations = recommendations[recommendations['genres'].apply(
            lambda x: any(g in x.lower() for g in selected_genres)
        )]

    return recommendations.drop_duplicates(subset='title').head(10)

# Genre match visualization
def plot_genre_match(recommendations, selected_genres):
    matches = []
    selected_genres_set = set([g.lower() for g in selected_genres])
    for _, movie in recommendations.iterrows():
        movie_genres = set(g.strip().lower() for g in movie['genres'].split('|'))
        overlap = len(movie_genres.intersection(selected_genres_set))
        match_percent = (overlap / len(selected_genres_set)) * 100 if selected_genres_set else 0
        matches.append(match_percent)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(recommendations['title'], matches, color='#4CAF50')
    ax.set_xlabel('Genre Match (%)', fontsize=12)
    ax.set_title('How Well These Match Your Taste', fontsize=14)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2, f'{width:.0f}%', va='center')

    plt.tight_layout()
    st.pyplot(fig)

# UI Layout
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
    all_genres = sorted(set(g for genre_str in df['genres'] for g in genre_str.split('|')))

    # Create a multiselect with restriction
    selected_genres = st.multiselect("Select up to 3 genres:", all_genres)

    # Genre warning
    if len(selected_genres) > 3:
        st.error("You can select a maximum of 3 genres.")
        selected_genres = selected_genres[:3]

    if 0 < len(selected_genres) <= 3:
        # Step 2: Movie Selection
        st.markdown('<p class="subheader">Step 2: Pick 5 Movies You Love</p>', unsafe_allow_html=True)
        genre_filtered_movies = df[df['genres'].apply(lambda x: any(g in x for g in selected_genres))]
        selected_movies = st.multiselect(
            "Select exactly 5 movies:",
            sorted(genre_filtered_movies['title'].unique())
        )

        if len(selected_movies) == 5:
            if st.button("🎯 Get Personalized Recommendations"):
                with st.spinner("Finding your perfect matches..."):
                    recommendations = get_recommendations(selected_movies, selected_genres)

                    if recommendations.empty:
                        st.warning("No matching movies found. Try changing genres or movies.")
                    else:
                        st.success("Here are your personalized recommendations!")
                        st.markdown("---")

                        # Top 3 Highlighted
                        cols = st.columns(3)
                        for i in range(min(3, len(recommendations))):
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
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown("### How These Match Your Taste")
                        plot_genre_match(recommendations, selected_genres)

        elif len(selected_movies) > 0:
            st.warning("Please select exactly 5 movies to get the best recommendations.")

if __name__ == "__main__":
    main()
