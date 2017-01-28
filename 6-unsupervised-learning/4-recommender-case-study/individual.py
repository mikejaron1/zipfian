import graphlab as gl
import numpy as np


def eda(filename):
    # 1. Explore the data files. To start load them into a GraphLab SFrame.
    song_sf = gl.SFrame.read_csv(filename, delimiter=',', header=None,
                                 error_bad_lines=False)
    song_sf.rename({'X1':'user', 'X2':'timestamp', 'X3': 'aid', 'X4': 'artist',
                    'X5':'sid','X6':'song'})


    # 2. We will do some exploratory analysis in pandas first. How many users
    # are there? How many artists are there? How many songs are there?
    print "There are %d users." % len(song_sf['user'].unique())
    print "There are %d artists." % len(song_sf['artist'].unique())
    print "There are %d songs." % len(song_sf['song'].unique())
        # OUTPUT:
        # There are 991 users.
        # There are 121330 artists.
        # There are 691888 songs.


    # 3. What is user 666's favorite song (and how absolutely heavy metal is
    # it)? What are user 333's top 10 favorite songs?
    grouped_s = song_sf.groupby(key_columns=['user', 'song'],
                                operations=gl.aggregate.COUNT())
    only666 = grouped_s.filter_by(['user_000666'], 'user')
    user666_song = only666.topk('Count', 1)['song'][0]
    print "User 666's favorite song is:", user666_song
        # OUTPUT:
        # User 666's favorite song is: Why Should I Cry For You?

    only333 = grouped_s.filter_by(['user_000333'], 'user')
    user333_songs = only333.topk('Count', 10)['song']
    print "User 333's 10 favorite songs are:"
    print "\n".join(user333_songs)
        # OUTPUT:
        # User 333's 10 favorite songs are:
        # It'S Time To Rock
        # Mount Wroclai (Idle Days)
        # Frontier
        # East Of Eden
        # Public Image
        # My Lady'S House
        # Ceremony
        # Slowhand Hussein
        # Slowly
        # Phoenix

    return grouped_s


def recommend(grouped_s):
    # 1. Before we can start recommending we need to create our feature matrix.
    # To create our feature matrix, we will convert our implicit user
    # preferences (song listens) into something meaningful. Create a feature
    # matrix where each row represents a (user, song, listen count) triplet
    # (i.e. 3 columns).

        # This is grouped_s

    # 2. When we try to compute our recommendations, we will blow up our memory
    # unless we take a subset of the data. Subset your SFrame such that you
    # only include (user, song) pairs with at least 25 listens.
    smaller_s = grouped_s[grouped_s['Count'] > 25]

    # 3. The simplest model might be to just predict the global mean. Using the
    # recommender toolkit, create an ItemMeansModel.
    model = gl.recommender.create(smaller_s, 'user', 'song', 'Count', 
                                  method='item_means')

    print "User 666's top 3 recommendations:"
    print model.recommend(['user_000666'], k=3)

        # OUTPUT:
        # +-------------+-----------------------------+-------+------+
        # |     user    |             song            | score | rank |
        # +-------------+-----------------------------+-------+------+
        # | user_000666 |         Say You Will        | 620.0 |  1   |
        # | user_000666 | Paranoid (Feat. Mr. Hudson) | 615.0 |  2   |
        # | user_000666 |        Coldest Winter       | 600.0 |  3   |
        # +-------------+-----------------------------+-------+------+

    return smaller_s


def evaluation(smaller_s):
    # 1. Using GraphLab, create a test-train split (random_split or
    # random_split_by_user) and evaluate how your basic recommender performs
    # with RMSE.
    train_set, test_set = \
        gl.recommender.util.random_split_by_user(smaller_s,
                                                 user_id='user',
                                                 item_id='song')
    model = gl.recommender.create(train_set, 'user', 'song', 'Count',
                                  method='item_means')
    print "baseline rmse:", \
        gl.evaluation.rmse(test_set['Count'], model.predict(test_set))

        # OUTPUT:
        # rmse: 39.5033236786


    # 2. Since some users may listen to music much more in general, we want to
    # normalize the counts. Normalize the listen counts for each user by the
    # total song plays by that user.
    total_counts = train_set.groupby('user',
                                     {'total_cnt':gl.aggregate.SUM('Count')})
    n_train = train_set.join(total_counts, on='user')
    n_train['normed'] = n_train['Count'] / n_train['total_cnt']
    del n_train['total_cnt']
    n_test = test_set.join(total_counts, on='user')
    n_test['normed'] = n_test['Count'] / n_test['total_cnt']
    del n_test['total_cnt']


    # 3. Create a new recommender using this new normalized feature matrix and
    # compare its performance to the un-normalized feature matrix. Which
    # performs better? And by how much?
    model = gl.recommender.create(n_train, 'user', 'song', 'normed',
                                  method='item_means')
    print "baseline normalized rmse:", \
        gl.evaluation.rmse(n_test['normed'], model.predict(n_test))

        # OUTPUT:
        # baseline normalized rmse: 0.145853506219

    return n_train, n_test


def collaboration(n_train, n_test):
    # 1. Create a basic collaborative filter on the listening data. Use the
    # item-similarity model with cosine distance as the similarity_type.
    # NOTE: This is an item-item collaborative filter
    model = gl.recommender.create(n_train, 'user', 'song', 'Count',
                                  method='popularity')
    print "popularity rmse:", \
        gl.evaluation.rmse(n_test['Count'], model.predict(n_test))

        # OUTPUT:
        # popularity rmse: 39.6128680248

    model = gl.recommender.create(n_train, 'user', 'song', 'normed',
                                  method='popularity')
    print "popularity normalized rmse:", \
        gl.evaluation.rmse(n_test['normed'], model.predict(n_test))

        # OUTPUT:
        # popularity normalized rmse: 0.145853506219

    model = gl.recommender.create(n_train, 'user', 'song', 'Count',
                                  method='item_similarity',
                                  similarity_type='pearson')
    print "pearson similarity rmse:", \
        gl.evaluation.rmse(n_test['Count'], model.predict(n_test))

        # OUTPUT:
        # pearson rmse: 54.2878354977

    model = gl.recommender.create(n_train, 'user', 'song', 'normed',
                                  method='item_similarity',
                                  similarity_type='pearson')
    print "pearson similarity normalized rmse:", \
        gl.evaluation.rmse(n_test['normed'], model.predict(n_test))

        # OUTPUT:
        # pearson normalized rmse: 0.145853506219

    model = gl.recommender.create(n_train, 'user', 'song', 'Count',
                                  method='item_similarity',
                                  similarity_type='cosine')
    print "cosine similarity rmse:", \
        gl.evaluation.rmse(n_test['Count'], model.predict(n_test))

        # OUTPUT:
        # cosine rmse: 54.0312932461

    model = gl.recommender.create(n_train, 'user', 'song', 'normed',
                                  method='item_similarity',
                                  similarity_type='cosine')
    print "cosine similarity normalized rmse:", \
        gl.evaluation.rmse(n_test['normed'], model.predict(n_test))

        # OUTPUT:
        # cosine normalized rmse: 0.151452655676


    # 2. Compare it's performance to your baseline.

        # rmse noted above


    # 3. Use your model to generate the top 5 recommendations for each user. Do
    # this for the baseline as well as the collaborative filter.

    print "top 10"
    print model.recommend()


def more_models(n_train, n_test):
    # 1. Create a Matrix Factorization model and compare it's performance to the
    # item-based collaborative filter. Which performs better?
    model = gl.recommender.create(n_train, 'user', 'song', 'normed',
                                  method='matrix_factorization')
    print "matrix factorization normalized rmse:", \
        gl.evaluation.rmse(n_test['normed'], model.predict(n_test))



if __name__ == '__main__':
    grouped_s = eda('data/lastfm/sample.csv')
    smaller_s = recommend(grouped_s)
    n_train, n_test = evaluation(smaller_s)
    collaboration(n_train, n_test)
    more_models(n_train, n_test)