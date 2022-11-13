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

m = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= m)]

joined = anime.merge(reduced_frame, how='inner', on='anime_id')
joined = joined[['user_id', 'name', 'rating_y']]

pivot = joined.pivot_table(index='name', columns='user_id', values='rating_y')

pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.fillna(0, inplace=True)

pivot_similarity = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm),
                                index=pivot_norm.index,
                                columns=pivot_norm.index)

def get_similar_anime(anime_name):
    if anime_name not in pivot_norm.index:
        return None, None
    else:
        similar_animes = pivot_similarity.sort_values(by=anime_name,
                                                      ascending=False).index[1:]
        similar_score = pivot_similarity.sort_values(by=anime_name,
                                                     ascending=False).loc[:,
                                                        anime_name].tolist()[1:]
        return similar_animes, similar_score

def predict_rating(user_id, anime_name):
    animes, scores = get_similar_anime(anime_name)
    anime_arr = np.array([x for x in animes])
    sim_arr = np.array([x for x in scores])
    
    filtering = pivot_norm[user_id].loc[anime_arr] != 0
    
    temp = np.dot(sim_arr[filtering][:10],
               pivot[user_id].loc[anime_arr[filtering][:10]]) \
            / np.sum(sim_arr[filtering][:10])
    
    return temp

def get_recommendation(user_id):
    predicted_rating = np.array([])
    
    for _anime in pivot_norm.index:
        predicted_rating = np.append(predicted_rating,
                                     predict_rating(user_id, _anime))
    
    temp = pd.DataFrame({'predicted':predicted_rating, 'name':pivot_norm.index})
    filtering = (pivot_norm[user_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted',
                                                  ascending=False)

    result = pd.DataFrame(anime.loc[anime_index.loc[temp.name[:10]]])
    return result['name'].tolist()

print_similar_animes(get_recommendation(10))
