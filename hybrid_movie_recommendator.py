
import pandas as pd

"""Veri Hazırlama"""

pd.set_option ('display.max_columns', 20)

pd.set_option ('display.width', None)

movie = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')

rating = pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")

df.head()

#1 film kaç kez geçmiş?
comment_counts = pd.DataFrame(df["title"].value_counts())

comment_counts.head()

#title'ı binden az olanlara nadir filmler dedim
rare_movies = comment_counts[comment_counts['title'] < 1000].index

rare_movies[0:10]

# binden fazla olanlara da yaygın filmler dedim
common_movies = df[~df['title'].isin(rare_movies)]

common_movies.head()

#user ve film ismine göre bir tablo çıkardık değerler ratinglerdir.
user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values = 'rating')

user_movie_df.head()

user_movie_df.shape

"""2) Öneri Yapılacak Kullanıcıların İzlediği Filmlerin Belirlenmesi"""

# kendimize hedef user belirledik. her zaman hedef aynı çıksın diye random state atadık.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state = 45).values)

random_user

#user ve filmlerle oluşturduğumuz dfin içerisinden random userımızı çektik = random_user_df
random_user_df = user_movie_df[user_movie_df.index == random_user]

random_user_df.head()

#random userımızın izlediği filmleri getirdik.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

movies_watched[:10]

#userımız 33 tane filmi izlemiş
len(movies_watched)

"""3) Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek"""

#userımızın izlediği 33 filmden herhangi birini veya birkaçını izleyen userları bulduk = movies_watched_df
movies_watched_df = user_movie_df[movies_watched]

movies_watched_df.head()

# kaçar tane aynı izlemişler ona baktık
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count.head()

user_movie_count = user_movie_count.reset_index()

user_movie_count.head()

user_movie_count.columns = ["userId", "movie_count"]

#20den daha fazla aynı izleyenleri sıraladık.
user_movie_count.sort_values("movie_count", ascending = False)

perc = len(movies_watched) * 60 / 100

#izlediklerinin hepsini izleyen kişi sayısına baktık. 214 kişiymiş.
user_movie_count[user_movie_count["movie_count"] == perc].count()

#yüzde 60'dan fazla aynı filmi izleyenlerin userIdlerini user_same_movies'e atadık.
user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]['userId']

user_same_movies.head()

user_same_movies.count()

"""4) Öneri Yapılacak Kullanıcı ile En Benzer Davranışı Gösteren Kullanıcıların Belirlenmesi

Bunun için 3 adım gerçekleştireceğiz:
* 1 Userımız ve diğer kullanıcıların verilerini bir araya getireceğiz.
* 2 Korelayon dataframe'ini oluşturacağız.
* 3 En benzer kullanıcıları (Top Users) bulacağız.
"""

movies_watched_df.head()

#userın izlediği filmlerden en az birini izleyenlerin içinde olduğu movies_watched_df içerisinden
#en az yüzde 60 tane izleyenleri aldık ve bunun ile kendi userımızın 33 tane izlediği filmleri birleştirdik.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies)],
                    random_user_df[movies_watched]])
#ilk kısın 137 bin küsür film içerisinden sadece same izleyenleri getirdi.
# 2. kısım ise bize userımızın izlediği filmleri getirdi. yani 191 tane

final_df.head(10)

final_df.shape
#1 tane user arttı neden? kendi userımızda eklendi. filmler de 33 tane.

#useridlerin hepsinin birbirleri ile olan benzerliklerine baktık.
final_df.T.corr()

# bir şeyi birden fazla göstermesini önledik ve tablo haline getirdik.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

#corelasyonu column olarak ekledik 
corr_df = pd.DataFrame(corr_df, columns = ["corr"])

corr_df.head()

#userlar karışmasın diye isim düzenlemesi
corr_df.index.names = ["user_id_1", "user_id_2"]

corr_df.head()

corr_df = corr_df.reset_index()

corr_df.head()

"""Bizim random_user sayesinde belirlediğimiz userımız ile yüzde 65 üzeri korelasyona sahip olan kullanıcılara erişelim."""

#korelasyonu %65 üstü olan ve ilk userı bizim seçtiğimiz user olanların userId2 sini ve korelasyonunu
# getirdik.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users.head()

#corelasyonuna göre büyükten küçüğe sıralı olsun dedik
top_users = top_users.sort_values(by='corr', ascending=False)

top_users.head()

#user_id_2 ismini değiştirdik
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('../input/movielens-20m-dataset/rating.csv')

#top userlarımızın ratinglerini, movieidlerini ve user idlerini barındıran bir df yaptık.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]],how = 'inner')

top_users_ratings.head()

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

top_users_ratings.head()

"""5) Weighted Average Recommendation Score'un Hesaplanması ve ilk 5 Filmin Tutulması"""

top_users_ratings["weighted_rating"] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.head()

top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'})

recommendation_df = recommendation_df.reset_index()

import matplotlib.pyplot as plt

# ağırlık ortalması yaklaşık 2.5 çıktı. buna göre hareket edebiliriz.
recommendation_df["weighted_rating"].hist()

plt.show()

# weighted_rating'i 2.5'den büyük olanları getirelim.
movies_to_be_recommended = recommendation_df[recommendation_df['weighted_rating'] > 2.5].sort_values("weighted_rating", ascending = False).head()

movies_to_be_recommended.head()

movie = pd.read_csv('../input/movielens-20m-dataset/movie.csv')

recommended_user_based_df = movies_to_be_recommended.merge(movie[["movieId","title"]])

recommended_user_based_df.head()

recommended_user_based_df.shape

"""6) Item-Based Recommendation

Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin alınması
"""

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending = False)["movieId"][0:1].values[0]

user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

"""### """

user_movie_df.corrwith(movie).sort_values(ascending=False).head(5)

def item_based_recommender(movie_name, user_movie_df, head=10):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith (movie).sort_values(ascending=False).head(head)

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df, 20).reset_index()

movies_from_item_based.head()

movies_from_item_based.rename(columns={0:"corr"}, inplace=True)

movies_from_item_based.head()

recommended_item_based_df = movies_from_item_based.loc[~movies_from_item_based["title"].isin(movies_watched)][:5]

recommended_item_based_df

hybrid_rec_df = pd.concat([recommended_user_based_df["title"], recommended_item_based_df["title"]]).reset_index(drop=True)

hybrid_rec_df

