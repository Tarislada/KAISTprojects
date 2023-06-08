import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics.pairwise
from uszipcode import SearchEngine
import plotly.express as px
from scipy.spatial.distance import pdist, dice
import scipy.sparse as sparse
from sklearn.metrics import pairwise_distances
import gower
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, OPTICS
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import hdbscan
import prince
import altair as alt
import altair_viewer as alt_viewer
import plotly.express as px
from sklearn import metrics as mt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn

search = SearchEngine()

ratings = pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/2023/Movielens/ratings.dat', delimiter='::', header=None, encoding='latin-1',
          names=['user_id', 'movie_id', 'rating', 'timestamp'],
          usecols=['user_id', 'movie_id', 'rating'], engine='python')

movies = pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/2023/Movielens/movies.dat', delimiter='::', header=None, encoding='latin-1',
          names=['movie_id', 'title', 'genre'], engine='python')

users = pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/2023/Movielens/users.dat', delimiter='::', header=None, encoding='latin-1',
        names = ['user_id','gender','age','occupation','zipcode'], engine = 'python')

statepop= pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/2023/DIS501/statepopulation.txt',delimiter='\t')
statecode = pd.read_csv('C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/2023/DIS501/statecode.txt',delimiter='\t')
statecodepop = pd.merge(statecode.rename(columns={'State Name': 'State'}),statepop,on='State')
stateinfo = statecodepop[['State','State Code','2003']]

year = movies['title'].apply(lambda x : x[-5:-1])
movies['year'] = year
decade = movies['year'].apply(lambda x: x[:-1]+'0')
movies['decade'] = decade


unique_genre_dict = {}

for row in movies['genre']:
    parsed_genre = row.split('|')
    for genre_name in parsed_genre:
        if (genre_name in unique_genre_dict) == False :
            unique_genre_dict[genre_name] = 1
        else:
            unique_genre_dict[genre_name] = unique_genre_dict[genre_name] +1

# genre_mat = np.empty((movies.shape[0],6),dtype=object)
# for row in range(len(movies['genre'])):
#     parsed_genre = movies['genre'][row].split('|')
#     padded_genre = parsed_genre + ['NaN'] * (6 - len(parsed_genre))
#     genre_mat[row,:] = padded_genre
#
# genres = ['genre1','genre2','genre3','genre4','genre5','genre6']
# genre_df = pd.DataFrame(genre_mat,columns=genres)
# pd.concat([movies,genre_df],axis=1)

movies['genre'] = movies['genre'].apply(lambda x: x.split('|'))
movies_exploded = movies.explode('genre')
movies_exploded.reset_index(drop=True, inplace=True)

def get_state(zipcode):
    place = search.by_zipcode(zipcode)
    try:
        statename = place.state
        return statename
    except:
        return 'NaN'

users['zipcode_simple'] = users['zipcode'].apply(lambda x: x[0:5])
users['States'] = users['zipcode_simple'].apply(get_state)
statecount = users.groupby('States').count()
state_relative = statecount.merge(stateinfo,left_index=True,right_on='State Code')
state_relative['percapita'] = state_relative['user_id'] / state_relative['2003'].str.replace(',','').astype(int) * 10000
state_to_region = {
    'AL': 'Southeast',
    'AK': 'Northwest',
    'AZ': 'West',
    'AR': 'Southeast',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'Mid-Atlantic',
    'FL': 'Southeast',
    'GA': 'Southeast',
    'HI': 'West',
    'ID': 'Northwest',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'Southeast',
    'LA': 'Southeast',
    'ME': 'Northeast',
    'MD': 'Mid-Atlantic',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'Southeast',
    'MO': 'Midwest',
    'MT': 'Northwest',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Mid-Atlantic',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'Southeast',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'Southwest',
    'OR': 'Northwest',
    'PA': 'Mid-Atlantic',
    'RI': 'Northeast',
    'SC': 'Southeast',
    'SD': 'Midwest',
    'TN': 'Southeast',
    'TX': 'Southwest',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'Mid-Atlantic',
    'WA': 'Northwest',
    'WV': 'Mid-Atlantic',
    'WI': 'Midwest',
    'WY': 'West'
}

fig = px.choropleth(statecount, locations=statecount.index, locationmode='USA-states', color='user_id', scope='usa')

# fig.show()

mean_rating = ratings.groupby('movie_id')['rating'].mean()
movies.merge(mean_rating,on='movie_id',how='left')

test = pd.merge(left=movies_exploded,right=ratings,how='inner',on='movie_id')
rating_by_genre_df = test.groupby('genre').agg({'rating': ['mean', 'count']}).sort_values(('rating', 'mean')).reset_index()
rating_by_genre_df.columns = ['_'.join(col).strip() for col in rating_by_genre_df.columns.values]
px.bar(rating_by_genre_df, x='genre_', y='rating_mean', height=300)
px.bar(rating_by_genre_df, x='genre_', y='rating_count', height=300)

combined_ratings_df = pd.merge(pd.merge(movies_exploded, ratings, on='movie_id'), users, on='user_id')
combined_ratings_data = combined_ratings_df.groupby(['genre', 'gender']).agg({'rating': ['mean', 'count']}).reset_index()

state_to_region = {
    'AL': 'Southeast',
    'AK': 'Northwest',
    'AZ': 'West',
    'AR': 'Southeast',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'Mid-Atlantic',
    'FL': 'Southeast',
    'GA': 'Southeast',
    'HI': 'West',
    'ID': 'Northwest',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'Southeast',
    'LA': 'Southeast',
    'ME': 'Northeast',
    'MD': 'Mid-Atlantic',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'Southeast',
    'MO': 'Midwest',
    'MT': 'Northwest',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Mid-Atlantic',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'Southeast',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'Southwest',
    'OR': 'Northwest',
    'PA': 'Mid-Atlantic',
    'RI': 'Northeast',
    'SC': 'Southeast',
    'SD': 'Midwest',
    'TN': 'Southeast',
    'TX': 'Southwest',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'Mid-Atlantic',
    'WA': 'Northwest',
    'WV': 'Mid-Atlantic',
    'WI': 'Midwest',
    'WY': 'West'
}

users['Region'] = users['States'].map(state_to_region)
new_users = users.drop(columns=['zipcode','zipcode_simple','States'])
new_users['occupation']=pd.Categorical(new_users.occupation)
users_encoded = pd.get_dummies(new_users)
users_encoded_noid = users_encoded.drop('user_id',axis=1)

# Creating distance matrix of user data: Gower's, Dice
dice_dist = pairwise_distances(users_encoded.values, metric='dice')
float_cols = users_encoded.select_dtypes(include=[int]).columns
users_encoded[float_cols] = users_encoded[float_cols].astype(np.float64)
gower_dist = gower.gower_matrix(users_encoded.values)
user_cos = sklearn.metrics.pairwise.cosine_similarity(users_encoded.values)

# Clustering using user distance matrices
opticlabels = OPTICS(metric='precomputed').fit_predict(dice_dist)
hdbscanlabels = hdbscan.HDBSCAN(metric='precomputed').fit_predict(dice_dist)

go_opticlabels = OPTICS(metric='precomputed').fit_predict(gower_dist)
go_hdbscanlabels = hdbscan.HDBSCAN(metric='precomputed').fit_predict(gower_dist)

# dimension reduction on user data
mca = prince.MCA(n_components=3,n_iter=100,random_state=101)
usersmca = mca.fit(users_encoded_noid)
users_mca = usersmca.transform(users_encoded_noid)
usersmca.eigenvalues_summary
alt.data_transformers.disable_max_rows()
mca.row_coordinates(users_encoded_noid)
fig = px.scatter_3d(users_encoded_noid, x=0, y=1, z=2)
fig.show()

contribution = usersmca.column_contributions_

# Clustering again with dr'ed user data
mca_optic = OPTICS(metric='mahalanobis').fit_predict()
mca_hdbscan =  hdbscan.HDBSCAN(metric='mahalanobis').fit_predict()

# Preparing Movie information for calculation
genre_dummy = pd.get_dummies(pd.DataFrame(movies['genre'].tolist()).stack()).sum(level=0)
new_movies = pd.concat([movies[['movie_id','title','decade']],genre_dummy],axis=1)
movie_encoded = new_movies.drop(['movie_id','title'],axis=1)

# Movie distance
movie_dice = pairwise_distances(movie_encoded.values,metric='dice')
movie_gower = gower.gower_matrix(movie_encoded.values)
movie_cos = sklearn.metrics.pairwise.cosine_similarity(movie_encoded.values)
#

moviesmca = mca.fit(movie_encoded)
moviesmca.eigenvalues_summary
movies_mca = moviesmca.transform(movie_encoded)
fig = px.scatter_3d(mca.row_coordinates(movie_encoded))
# fig.show()




clustermca = prince.MCA(n_components=10,n_iter=100,random_state=101)

meansil = np.zeros((19,10))
clusteruserdata = clustermca.fit_transform(users_encoded_noid)
for iter in range(10):
    for ii in range(2,21):
        kmeanscluster = KMeans(n_clusters=ii)
        meansil[ii-2,iter] = mt.silhouette_score(clusteruserdata, kmeanscluster.fit_predict(clusteruserdata))
meanmeansil = np.mean(meansil,axis=1)
semmeansil = np.std(meansil,axis=1) / np.sqrt(meansil.shape[1])
fig, ax = plt.subplots()
ax.errorbar(np.arange(meansil.shape[0]), meanmeansil, yerr=semmeansil, fmt='o', capsize=5,ls='-')
plt.show()


moviemeansil = np.zeros((19,10))
clustermoviedata = clustermca.fit_transform(movie_encoded)
for iter in range(10):
    for ii in range(2,21):
        kmeanscluster = KMeans(n_clusters=ii)
        moviemeansil[ii-2,iter] = mt.silhouette_score(clustermoviedata, kmeanscluster.fit_predict(clustermoviedata))
moviemeanmeansil = np.mean(moviemeansil,axis=1)
moviesemmeansil = np.std(moviemeansil,axis=1) / np.sqrt(moviemeansil.shape[1])
fig, ax = plt.subplots()
ax.errorbar(np.arange(moviemeansil.shape[0]), moviemeanmeansil, yerr=moviesemmeansil, fmt='o', capsize=5,ls='-')
# plt.show()

# the results of this shows that k-12 students are special cases - 93% contribution on 1st axis
# the second axis is also made up of occ 19 97% - unemployed
# third axis is ambiguous - 31% max, farmers contributing 31% of the axis
# calculating 5 most similar users (can use any distance matrix in place of dice_dist)

row_indices = ratings['user_id'].astype('category').cat.codes
# col_indices = ratings['movie_id'].astype('category').cat.codes
col_indices = ratings['movie_id']
values = ratings['rating']
csr_matrix_ratings = sparse.csr_matrix((values, (row_indices, col_indices)))


movie_key = new_movies['movie_id']
no_rating = np.setdiff1d(movie_key.unique(),ratings['movie_id'].unique()) # movies that are in the list, but not rated

mask = ~movie_key.isin(no_rating)
filtered = movie_key[mask].reset_index()

result_user = np.empty((ratings.shape[0],5))
result_movie = np.empty((ratings.shape[0],5))

for index,row in ratings.iterrows():
    target_movie = row['movie_id']
    target_user = row['user_id']-1

    sim_users = user_cos[target_user,:].argsort()
    sim_users = sim_users[sim_users != target_user]

    topsimratings = csr_matrix_ratings[sim_users, target_movie].data[:5]
    # topsimratings = targetmat[targetmat != 0][0,:5]

    result_user[index,:topsimratings.shape[0]] = topsimratings

    movie_index = movie_key[movie_key == target_movie].index.values[0] # convert ratings_movie id to a index in movie similarity matrix

    sim_movies = movie_cos[movie_index,:].argsort()
    sim_movies = sim_movies[sim_movies != movie_index] # content of sim_movies is based on # of movies in new_movies. csr is based on rated.
    # sim_movies.max() = 3882, not fitting into the movie_dice
    # sim_movies = sim_movies[~ np.isin(sim_movies, no_rating)]

    movie_mat = csr_matrix_ratings[target_user,sim_movies].data[:5]
    # topmovieratings = movie_mat[movie_mat != 0][0,:5]

    # result_movie[index,:] = topmovieratings
    result_movie[index, :movie_mat.shape[0]] = movie_mat

new_rating = pd.concat([ratings,pd.DataFrame(result_user,columns=["simu1", "simu2", "simu3", "simu4", "simu5"])],axis=1)
new_rating = pd.concat([new_rating,pd.DataFrame(result_movie,columns=["simm1", "simm2", "simm3", "simm4", "simm5"])],axis=1)

usr_mean_rating = ratings.groupby('user_id')['rating'].mean()
new_rating = new_rating.merge(usr_mean_rating.rename('Umean'),left_on='user_id',right_index=True,how='left')
new_rating = new_rating.merge(mean_rating.rename('Mmean'),left_on='movie_id',right_index=True,how='left')

concatuser = users_encoded['user_id'].astype('int64')
new_rating = new_rating.merge(concatuser,on='user_id',how='left')
concatmovie = new_movies.drop(columns=['title'])
new_rating = new_rating.merge(concatmovie,on='movie_id',how='left')
#labels = SpectralClustering(n_clusters=5, affinity='precomputed',verbose=True,n_jobs=4, eigen_solver='amg',assign_labels='cluster_qr').fit_predict(gower_dist)

coltochan = new_rating.columns[16:64]
new_rating[coltochan] = new_rating[coltochan].astype('bool')
new_rating['decade'] = new_rating['decade'].astype('int64')
new_rating.loc[new_rating['rating']<=3, 'rating'] = 0
new_rating.loc[new_rating['rating']>=4, 'rating'] = 1
X_train, X_test, y_train, y_test = train_test_split(new_rating.drop(columns=['movie_id','user_id','rating']),new_rating['rating'],test_size=0.2,random_state=42)

# regress_rating['rating'] = ratings['rating']

xgb_regress = xgb.XGBRegressor(n_jobs=13,random_state=15,n_estimators=100)
xgb_regress.fit(X_train, y_train-1, eval_metric = 'rmse')
test_results = dict()
# from the trained model, get the predictions
y_test_pred = xgb_regress.predict(X_test)

xgb_model = xgb.XGBClassifier(n_jobs=13,random_state=15,n_estimators=100)
xgb_model.fit(X_train, y_train, eval_metric = 'auc')

#dictionaries for storing train and test results
test_results = dict()
regress_results = dict()

# from the trained model, get the predictions
y_test_pred = xgb_model.predict(X_test)
fpr,tpr,thres = sklearn.metrics.roc_curve(y_test,y_test_pred)
auc = sklearn.metrics.auc(fpr,tpr)
accuracy = sklearn.metrics.accuracy_score(y_test,y_test_pred)


test_results = {'accuracy': accuracy, 'AUC' : auc, 'predictions' : y_test_pred}
regress_results = {'rmse': rmse, 'mape':mape}

feature_importance = xgb_model.get_score(importance_type='gain')
features = list(feature_importance.keys())
importance = list(feature_importance.values())

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance)), importance, align='center')
plt.yticks(range(len(features)), features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance')
plt.show()

# Distance matrix - Dice or Gowers V
# K-medoids, hieracrchical, spectral clustering, hdbscan, optic

# plan for genre - 18 binaries or 4 genre variables that contain 1st~4th genre of the movie V
# add statecode or division based on zipcode and http://www.structnet.com/instructions/zip_min_max_by_state.html V
# add movie overall rating V
# search for tags in larger dataset? use wiki for plot

# combined_ratings_data.loc[combined_ratings_data['gender'] == 'F', 'rating count'] /= len(combined_ratings_df[combined_ratings_df['gender'] == 'F'])
# combined_ratings_data.loc[combined_ratings_data['gender'] == 'M', 'rating count'] /= len(combined_ratings_df[combined_ratings_df['gender'] == 'M'])
# px.bar(combined_ratings_data, x='genre', y='rating count', color='gender', barmode='group')

# sns.barplot(decade.index,)
# sns.barplot(unique_genre_dict,)

# Cluster movies and users with given information (Occupation, Age, Large regions, gender) / (Genres, year, mean ratings)
# see if there is a correlation in User choices of movie clusters - MCA, Kmode, DBscan, / To use this, Use Dice similarity coefficient
# user 3598, 4486 is a terrorist - rated all movies as 1
# SVD recommendation system
# DNN recommendation system

# without rating information, user cluster is demographically similar users.
# see if they have a trend (most likely none)
#
# next, see if you can improve the clusters with the movie ratings they have made
# i.e) supervised clustering methods
#
# also, see if movies form a cluster.
# (unlikely)
#
# SVD rec - XGBoost rec - DNN rec
#
# reg_encoded = pd.get_dummies(new_users['occupation'],prefix='job')
# Collaborative filtering recommendation algorithm
# based on KNN and Xgboost hybrid

