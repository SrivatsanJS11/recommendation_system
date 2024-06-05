import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

pd.set_option('display.max_columns', None)

encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
for encoding in encodings:
    try:
        df1 = pd.read_csv(r'C:\Users\jssri\OneDrive\Documents\Recommendation System\NetflixDataset.csv', encoding=encoding)
        print(f"Successfully read the file with {encoding} encoding")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file with {encoding} encoding")

selected_columns = ['Title', 'Genre','Key Words','Languages','Series or Movie','Runtime','Director','Actors','Summary','Release Date', 'IMDb Score', 'Image']

# Creating a copy with selected columns
df = df1[selected_columns].copy()


df['IMDb Score'] = df['IMDb Score'].fillna(df['IMDb Score'].median)

df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Release Year'] = df['Release Date'].dt.year
df.drop('Release Date', axis=1, inplace=True)

df['Release Year'].fillna(df['Release Year'].mode , inplace = True)
#null value in release year

df['Runtime'] = df['Runtime'].fillna('<30 minutes')

df['Director'].fillna('', inplace=True)
df['Languages'].fillna('' , inplace = True)
df['Actors'].fillna('' , inplace = True)
df['Genre'].fillna('' , inplace = True)
df['Key Words'].fillna('' , inplace = True)
df['Summary'].dropna(inplace = True)

#removing punctuation
import string
string.punctuation
def remove_punctuation(text):
    if type(text) == str:
        punctuation_free = "".join([i for i in text if i not in string.punctuation])
        return punctuation_free
    else:
        return str(text)
#storing the puntuation free text

#remove spaces between names
df['Genre'] = df['Genre'].str.replace(' ', '')
df['Actors'] = df['Actors'].str.replace(' ', '')
df['Key Words'] = df['Key Words'].str.replace(' ', '')
df['Director'] = df['Director'].str.replace(' ', '')


df['Title'] = df['Title'].apply(lambda x:remove_punctuation(x))
df['Director'] = df['Director'].apply(lambda x:remove_punctuation(x))
df['Summary'] = df['Summary'].apply(lambda x:remove_punctuation(x))
df['Languages'] = df['Languages'].apply(lambda x:remove_punctuation(x))



def lowercase(text):
    if type(text) == str:
        return text.lower()
    else:
        return str(text)


df['Title'] = df['Title'].apply(lowercase)
df['Genre'] = df['Genre'].apply(lowercase)
df['Key Words'] = df['Key Words'].apply(lowercase)
df['Languages'] = df['Languages'].apply(lowercase)
df['Director'] = df['Director'].apply(lowercase)
df['Actors'] = df['Actors'].apply(lowercase)
df['Summary'] = df['Summary'].apply(lowercase)
df['Series or Movie'] = df['Series or Movie'].apply(lowercase)

df['Summary'] = df['Summary'].astype(str)
df['Summary'] = df['Summary'].apply(lambda x: x.split())

df['Series or Movie'] = df['Series or Movie'].astype(str)
df['Series or Movie'] = df['Series or Movie'].apply(lambda x: x.split())

df['Director'] = df['Director'].astype(str)
df['Director'] = df['Director'].apply(lambda x: x.split())


df['Actors'] = df['Actors'].astype(str)
df['Actors'] = df['Actors'].apply(lambda x: x.split())

df['Genre'] = df['Genre'].astype(str)
df['Genre'] = df['Genre'].apply(lambda x: x.split())

df['Key Words'] = df['Key Words'].astype(str)
df['Key Words'] = df['Key Words'].apply(lambda x: x.split())

df['Languages'] = df['Languages'].astype(str)
df['Languages'] = df['Languages'].apply(lambda x: x.split())

#remove stopwords
import nltk
nltk.download('stopwords')
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

df['Summary']= df['Summary'].apply(lambda x:remove_stopwords(x))


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# Defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# Applying the lemmatizer function to the 'Summary' column
df['Summary'] = df['Summary'].apply(lambda x: lemmatizer(x))

df['Tags']=df['Genre'] + df['Key Words'] + df['Director'] + df['Actors'] + df['Languages'] + df['Summary'] + df['Series or Movie']

new_df= df[['Title','Tags','Runtime','IMDb Score', 'Release Year' ,'Image']]

new_df['Tags'] = new_df['Tags'].apply(lambda x: " ".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=6000,stop_words='english')

vectors=cv.fit_transform(new_df['Tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()  # Make sure the movie title comparison is case-insensitive
    if movie not in new_df['Title'].values:
        return ["Movie not found. Please check the movie title and try again."]

    movie_index = new_df[new_df['Title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
    candidates = []
    for i in movies_list:
        candidates.append(list(new_df.iloc[i[0]]))
    return candidates

st.title('Netflix Recommendation System')
movie = st.text_input('Enter a movie title')
if st.button('Recommend'):
    if movie:
        recommendations = recommend(movie)
        if isinstance(recommendations, list) and len(recommendations) > 0 and isinstance(recommendations[0], str):
            st.write(recommendations[0])  # Display the error message
        else:
            st.write('Recommendations:')
            for rec in recommendations:
                st.write(f"Title: {rec[0]}")
                st.write(f"Runtime: {rec[2]}")
                st.write(f"IMDb Score: {rec[3]}")
                st.write(f"Release Year: {rec[4]}")
                st.image(rec[5])  # Assuming rec[5] is the image URL
    else:
        st.write('Please enter a movie title.')
