import pandas as pd

tv=pd.read_csv(r"C:\Users\Manisha\Downloads\TV_Final.csv")
print(tv.isna().sum())

tv=tv["Operating System"].dropna()
print(tv.isna().sum())

from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser=TfidfVectorizer(stop_words="english")
vector_matrix=vectoriser.fit_transform(tv)


from sklearn.metrics.pairwise import linear_kernel

similarity_score=linear_kernel(vector_matrix,vector_matrix)
print(similarity_score.shape)
print(similarity_score[:10,:10])