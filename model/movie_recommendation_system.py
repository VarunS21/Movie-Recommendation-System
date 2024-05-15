from flask import request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.options.display.max_columns = 50
df = pd.read_csv('./model/movies.csv')

features = ['genres','keywords','title','tagline','cast','director']
for values in features:
    df[values] = df[values].fillna('')
combined_features = df['genres'] + " " + df['keywords'] + " "  +df['title'] + " "  +df['tagline'] + " " +df['cast'] + " "+ df['crew']

vector = TfidfVectorizer()
feature_vectors = vector.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)
movies_list = df['title'].to_list()


def recommend_movies(input_movie, no_movies):
    close_match = difflib.get_close_matches(input_movie,movies_list)
    close_match = close_match[0]
    movie_index = df[df['title']==close_match]["index"].values[0]
    similarity_score = list(enumerate(similarity[movie_index]))
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)
    i = 1
    movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df[df['index']==index]["title"].values[0]
        if (i < no_movies+1):
            # print(i+":"+title_from_index)
            movies.append(title_from_index)
            i +=1
    return movies
