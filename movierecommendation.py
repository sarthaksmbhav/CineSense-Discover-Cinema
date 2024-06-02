pip install IMDbPY

import numpy as np
import pandas as pd
import imdb
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import files

uploaded = files.upload()

data=pd.read_csv('IMDB-Movie-Data.csv')
data.head()

data.shape
data.info()

data.isnull().sum()

sns.heatmap(data.isnull())

data.dropna(axis=0)

data.duplicated().values
data.drop_duplicates()

data.describe()

data[data['Runtime (Minutes)']>=180]

sns.barplot(data.head(50))

data=data[['Rank','Title','Genre','Description','Actors','Rating','Director']]
data.head()

def con(obj):
  l=[]
  counter=0
  for i in range(len(obj)):
    if counter<3:
      l.append(obj[i])
    else :
      break
  return l;

data['Actors']=data['Actors'].apply(lambda x:x.split(","))

data.head()

data['Description']=data['Description'].apply(lambda x:x.split())

data.head()
data['Genre']=data['Genre'].apply(lambda x:x.split(","))
data['Director']=data['Director'].apply(lambda x:x.split())

data.tail()

data.head(10)

data['tags']=data['Description']+data['Genre']+data['Actors']+data['Director']

data.head()

movi=data[['Rank','Title','tags']]
movi

movi.loc[:, 'tags'] = movi['tags'].apply(lambda x: " ".join(x))
movi

ps=PorterStemmer()
def stem(text):
  n=[]
  for i in text.split():
    n.append(ps.stem(i))
  return " ".join(n)

cv=CountVectorizer(max_features=1000,stop_words="english")

vector=cv.fit_transform(movi['tags']).toarray()

movi['tags']=movi['tags'].apply(stem)
cv.get_feature_names_out()

similarity=cosine_similarity(vector)

def recommend(movie):
  movie_index=movi[movi['Title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(movi.iloc[i[0]].Title)

recommend('The Dark Knight')