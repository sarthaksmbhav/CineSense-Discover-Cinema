#app.py
import streamlit as st
import pickle
import pandas as pd
similarity=pickle.load(open('similarity.pkl', 'rb'))
def recommend(movie):
  movie_index=movi[movi['Title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  x=[]
  for i in movies_list:
    movi_id=i[0]
    x.append(movi.iloc[i[0]].Title)
  return x
dict = pickle.load(open('Movies.pkl', 'rb'))
movi=pd.DataFrame(dict)

st.title('Sarthak Recommendations: ')

option=st.selectbox('Select Option',movi['Title'])

def pos(movie):
  movie_index=movi[movi['Title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  x=[]
  for i in movies_list:
    movi_id=i[0]
    x.append(movi.iloc[i[0]].Poster)
  return x
def desc(movie):
  movie_index=movi[movi['Title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  x=[]
  for i in movies_list:
    movi_id=i[0]
    x.append(movi.iloc[i[0]].tags)
  return x
if st.button('Recommend'):
  recommendation=recommend(option)
  poster=pos(option)
  description=desc(option)
  for i in range(len(recommendation)):
    st.write(recommendation[i])
    st.image(poster[i])
    st.write(description[i])