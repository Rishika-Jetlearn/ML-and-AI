import pandas as pd

movies=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\movies_metadata.csv")
print(movies.isna().sum())
movies=movies[["overview","title","vote_average"]].dropna()
print(movies.isna().sum())
#vercoriser=converting large text to numbers

from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser=TfidfVectorizer(stop_words="english")
vector_matrix=vectoriser.fit_transform(movies["overview"])
print(vector_matrix.shape)
print(vectoriser.get_feature_names_out()[5000:5015])

from sklearn.metrics.pairwise import linear_kernel

similarity_score=linear_kernel(vector_matrix,vector_matrix)
print(similarity_score.shape)
print(similarity_score[:10,:10])