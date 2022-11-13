
import pandas as pd
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from google.colab import drive
drive.mount('/content/drive')

anime = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/irm-final-dataset/anime.csv")
rating = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/irm-final-dataset/rating.csv")


anime_ratings = pd.DataFrame(rating.groupby('anime_id')['rating'].count())
user_ratings = pd.DataFrame(rating.groupby('user_id')['rating'].count())

anime_ratings = anime_ratings[anime_ratings.rating >= 1000]
user_ratings = user_ratings[user_ratings.rating >= 200]

rating = rating[rating.anime_id.isin(anime_ratings.index.tolist())]
rating = rating[rating.anime_id.isin(user_ratings.index.tolist())]
rating = rating.replace(-1, np.nan)

anime['rating'] = anime['rating'].replace(np.nan, 0.0)

missing = anime.loc[(anime['episodes']=="Unknown") & (anime['type'].isnull())].copy()
missing_indexes = missing.index.tolist()
for i in missing_indexes:
  anime = anime[anime.index != i]

anime['genre'].fillna('Unknown', inplace=True)


v = anime['members'].count()
c = anime.rating.mean()
r = anime.rating
m = anime['members'].quantile(0.75)
w = (r*v + c*m)/(v+m)
anime['weighted_rating'] = anime.apply(w, axis=1)

anime_new = anime
anime_new.drop(['anime_id', 'rating', 'members', 'episodes'], axis=1,
               inplace=True)
anime_new = pd.concat([anime_new, anime_new['type'].str.get_dummies(),
                       anime_new['genre'].str.get_dummies(sep=',')], axis=1)

anime_genres = anime_new.loc[:, "Movie":].copy()

anime_genres = pd.DataFrame(anime_genres)

knn = NearestNeighbors(n_neighbors=12, algorithm='ball_tree').fit(anime_genres)
distances, indices = knn.kneighbors(anime_genres)

def get_index_from_name(name):
    return anime[anime["name"]==name].index.tolist()[0]

def get_recommendation(query):
  listed = []
  found_id = get_index_from_name(query)
  for id in indices[found_id][1:]:
    listed.append(anime.iloc[id]['name'])
  return listed

def print_similar_animes(listed):
  for i,element in enumerate(listed):
    print('{}. {}'.format(i+1, element))

print_similar_animes(get_recommendation(query))
