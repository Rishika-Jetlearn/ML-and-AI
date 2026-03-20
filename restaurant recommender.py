import pandas as pd

restaurants=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\indian_restaurants.csv")
restaurants.info()

v=restaurants["rating"].count()
m=50
print(m)
r=restaurants["rating"]
c=restaurants["rating"].mean()

restaurants["weighted_rating"]=(v/(v+m))*r+(m/(v+m))*c

print(restaurants.sort_values(by="weighted_rating",ascending=False)[["restaurant_name","location","weighted_rating"]].head(10))
 