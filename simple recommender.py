import pandas as pd

movies=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\movies_metadata.csv")
movies.info()
#Weighted Rating -   (v/(v+m))*R + (m/(v+m))*C
#v=number of votes(vote_count)
#m=minimum number of votes needed to be put in the chart(you choose)
#r= average rating of the movie(vote_average)
#c=mean of all the vote ratings

v=movies["vote_count"]
m=movies["vote_count"].quantile(0.9)
print(m)
r=movies["vote_average"]
c=movies["vote_average"].mean()

movies["weighted_rating"]=(v/(v+m))*r+(m/(v+m))*c

print(movies.head())

print(movies.sort_values(by="weighted_rating",ascending=False)[["title","weighted_rating"]].head(10))
