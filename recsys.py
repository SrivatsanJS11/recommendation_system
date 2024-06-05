import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
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

df1.shape

df1.info()

df1.describe()

df1.isnull().sum()

df1.duplicated().sum()

selected_columns = ['Title', 'Genre','Key Words','Languages','Series or Movie','Runtime','Director','Actors','Summary','Release Date', 'IMDb Score', 'Image']

# Creating a copy with selected columns
df = df1[selected_columns].copy()
df.head()

# prompt: change the colour of the below histogram from blue to red, sns

sns.histplot(data=df, x="Runtime", color="red")

sns.histplot(x= df['IMDb Score'] , data = df , kde = True , color = 'red')
plt.show()
sns.boxplot(x= df['IMDb Score'] , data = df,color='red')
plt.show()
# i guess we can use median to impute missing data in the IMDb score column
df['IMDb Score'] = df['IMDb Score'].fillna(df['IMDb Score'].median)

df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Release Year'] = df['Release Date'].dt.year
df.drop('Release Date', axis=1, inplace=True)

df['Release Year'].fillna(df['Release Year'].mode , inplace = True)
#null value in release year

#null value in runtime
df.iloc[df[df['Runtime'].isnull()].index.item()]
#it is a series, so we will replace the nan value in runtime as <30 minutes

df['Runtime'] = df['Runtime'].fillna('<30 minutes')

df['Director'].fillna('', inplace=True)
df['Languages'].fillna('' , inplace = True)
df['Actors'].fillna('' , inplace = True)
df['Genre'].fillna('' , inplace = True)
df['Key Words'].fillna('' , inplace = True)
df['Summary'].dropna(inplace = True)

df.isnull().sum()

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
df.head()

#converting to lower case
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
df.head()

df['Summary'][0]

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
df.head()

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
df.head()

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
df.head()

"""EDA"""

encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
for encoding in encodings:
    try:
        df2 = pd.read_csv(r'C:\Users\jssri\OneDrive\Documents\Recommendation System\NetflixDataset.csv', encoding=encoding)
        print(f"Successfully read the file with {encoding} encoding")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file with {encoding} encoding")

df2['Genre'] = df2['Genre'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
df2['Genre'] = df2['Genre'].apply(lambda x: word_tokenize(str(x)))
genres = list(df2.loc[0:9424, 'Genre' ])
genres_separate = []
for i in genres:
  for j in i:
    genres_separate.append(j)
genres_separate_new = []
for i in genres_separate:
  if i != 'nan':
    genres_separate_new.append(i)
  else:
    continue
genresdf = pd.DataFrame(genres_separate_new)
genresdf.columns = ['Genre']
print(genresdf['Genre'].unique())
ax = sns.countplot(x = 'Genre'  , data = genresdf , palette = 'Set3',width = 0.8 )
for i in ax.containers:
    ax.bar_label(i,)
plt.xticks(rotation = 90)
plt.show()

sns.countplot(x = 'Series or Movie' , data = df2  , palette = 'Set2' )
ax = sns.countplot(x='Series or Movie', data=df2, palette='Set2')

# Display counts on top of each bar
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1,
            f'{int(height)}', ha="center")

runtimes = list(df2.loc[0:9425 , 'Runtime'])
runtimes_new = []
for i in runtimes:
  if i != '< 30 minutes':
    runtimes_new.append(i)
  else:
    continue
runtimedf = pd.DataFrame(runtimes_new)
runtimedf.columns = ['Runtime of Movies']
sns.countplot(x  = 'Runtime of Movies' , data = runtimedf , palette = 'Set2')

plt.show()

df2.head()

netflix_date = df2[['Netflix Release Date']].dropna()

netflix_date['year'] = netflix_date['Netflix Release Date'].apply(lambda x: x.split('-')[0])
netflix_date['month'] = netflix_date['Netflix Release Date'].apply(lambda x: x.split('-')[1])

netflix_date = netflix_date[netflix_date['year'] != '2015']
netflix_date = netflix_date[netflix_date['year'] != '2021']

netflix_date.sort_values(by =['year', 'month'])

drt = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0).T
drt

plt.figure(figsize = (10,8), dpi = 100)
plt.pcolor(drt, cmap = 'Reds',edgecolors = 'white', linewidths = 2)
plt.xticks(np.arange(0.5, len(drt.columns), 1),drt.columns, fontsize = 7)
plt.yticks(np.arange(0.5, len(drt.index), 1), drt.index, fontsize = 7)

plt.title('Netflix Content Update', fontsize = 12, fontweight = 'bold')
cbar = plt.colorbar()

cbar.ax.tick_params(labelsize = 8)
cbar.ax.minorticks_on()
plt.show()

"""Tagging"""

df['Tags']=df['Genre'] + df['Key Words'] + df['Director'] + df['Actors'] + df['Languages'] + df['Summary'] + df['Series or Movie']
df.head()

df.iloc[0].Tags

df.isnull().sum()

new_df= df[['Title','Tags','Runtime','IMDb Score', 'Release Year' ,'Image']]
new_df.head()

new_df['Tags'] = new_df['Tags'].apply(lambda x: " ".join(x))

new_df['Tags'][0]

new_df.shape

"""Vectors"""

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=6000,stop_words='english')

vectors=cv.fit_transform(new_df['Tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

"""Recommending

recommendation based on title
"""

def recommend(movie):
  candidates = []
  movie_index = new_df[new_df['Title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(similarity[movie_index])) , reverse = True , key = lambda x: x[1])[1:6]
  for i in movies_list:
    candidates.append(list(new_df.iloc[i[0]]))
  for j in candidates:
    print(j[0])

recommend('the dark knight')