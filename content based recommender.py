import pandas as pd

movies=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\movies_metadata.csv")
#making it smaller
movies=movies.iloc[:20000]
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

mapping=pd.Series(movies.index,index=movies["title"])
print(mapping)

def best_movie(title):
    index=mapping[title]
    scores=list(enumerate(similarity_score[index]))
    print(scores[:10])
    sorted_movies=sorted(scores,key=lambda x:x[1], reverse=True)
    top_ten=sorted_movies[:10]
    top_ten_index=[i[0]for i in top_ten]
    print(top_ten_index)
    final=movies["title"].iloc[top_ten_index]
    print(final)

best_movie("Toy Story")
