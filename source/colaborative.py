
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


anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
m = anime['members'].quantile(0.5)
anime = anime[(anime['members'] >= m)]

joined = anime.merge(rating, how='inner', on='anime_id')
joined = joined[['user_id', 'name', 'rating_y']]
joined = joined[(joined['user_id'] <= 30000)]

pivot = joined.pivot_table(index='user_id', columns='name', values='rating_y')
pivot.dropna(axis=0, how='all', inplace=True)

pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.fillna(0, inplace=True)


user_similarity = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm),
                               index=pivot_norm.index, columns=pivot_norm.index)
user_similarity.head()

user_knn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(pivot_norm)
user_distances, user_indices = user_knn.kneighbors(pivot_norm)


def get_recommendation(user_id):
    users, similarity = get_similar_user(user_id)
    similarity, users = user_distances[user_id], user_indices[user_id]
    
    if users is None or similarity is None:
        return None
    
    user_arr = np.array([x for x in users[1:]])
    sim_arr = np.array([x for x in similarity[1:]])
    predicted_rating = np.array([])

    
    for anime_name in pivot_norm.columns:
        filtering = pivot_norm.iloc[user_arr][anime_name] != 0.0  
        temp = np.dot(pivot.iloc[user_arr[filtering]][anime_name],
                      sim_arr[filtering]) / np.sum(sim_arr[filtering])
        predicted_rating = np.append(predicted_rating, temp)
    
    temp = pd.DataFrame({'predicted':predicted_rating,
                         'name':pivot_norm.columns})
    
    filtering = (pivot_norm.loc[user_id] == 0.0)

    temp = temp.loc[filtering.values].sort_values(by='predicted',
                                                  ascending=False)

    result = pd.DataFrame(anime.loc[anime_index.loc[temp.name[:10]]])
    return result['name'].tolist()


def print_similar_animes(anime_list):
  for i,element in enumerate(anime_list):
    print('{}. {}'.format(i+1, element))


print_similar_animes(get_recommendation(10))
