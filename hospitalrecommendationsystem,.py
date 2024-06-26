# -*- coding: utf-8 -*-
"""HospitalRecommendationSystem,.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jh3T_d7X5utrJWpsCvvdWX84TRnljFsB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('/content/centre.csv')
df

df['Specialization'].value_counts()

df['Specialization'].unique()

print(df['Index'].dtype)

df['Index'].unique()

df['Ratings'] = df['Ratings'].astype(str)

df['Cost (INR)'] = df['Cost (INR)'].astype(str)

df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df['Index'] = df['Index'].fillna(-1).astype(int)

print(df['Index'].dtype)

selected_features = ['Location','Specialization','Cost (INR)','Ratings']
print(selected_features)

for feature in selected_features:
  df[feature] = df[feature].fillna('')

combined_features = df['Location']+' '+df['Specialization']+' '+df['Cost (INR)']+' '+df['Ratings']

print(combined_features)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

similarity = cosine_similarity(feature_vectors)

print(similarity)

print(similarity.shape)

name = input('Enter your addiction among these (\'Substance Abuse Rehabilitation\', \'Alcohol Rehabilitation\', \'Drug Addiction Rehabilitation\', \'Gambling Addiction Rehabilitation\', \'Smoking Cessation Program\'): ')

list_of_all_titles = df['Specialization'].tolist()
print(list_of_all_titles)

find_close_match = difflib.get_close_matches(name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

index_of_the_movie = df[df.Specialization == close_match]['Index'].values[0]
print(index_of_the_movie)

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

sorted_similar_hospital = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print(sorted_similar_hospital)

print('Rehabilitation suggested for you:\n')

i = 1

for x in sorted_similar_hospital:
    index = x[0]
    movie_data = df[df['Index'] == index]
    if not movie_data.empty:
        title_from_index = movie_data['Name'].values[0] + ',  ' + \
                          movie_data['Ratings'].values[0] + ',  ' + \
                          movie_data['Location'].values[0]
        if i < 20:
            print(i, '.', title_from_index)
            i += 1

print('Rehabilitation suggested on the basis of your rating and the place:\n')


min_rating = float(input("Enter the minimum rating: "))
max_rating = float(input("Enter the maximum rating(not write greater than the 5): "))
city = input("Enter the city: ")
print('\n')

i = 1
for x in sorted_similar_hospital:
    index = x[0]
    data = df[df['Index'] == index]
    if not data.empty:

        rating = float(data['Ratings'].values[0])


        if min_rating <= rating <= max_rating and city in data['Location'].values[0]:
            title_from_index = data['Name'].values[0] + ',  ' + \
                              str(rating) + ',  ' + \
                              data['Location'].values[0]
            if i < 10:
                print(i, '.', title_from_index)
                i += 1

