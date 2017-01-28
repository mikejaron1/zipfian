import pandas as pd
import numpy as np
from scipy import sparse
from matrix_factorization_soln import MatrixFactorizationRec
from item_soln import ItemItemRec

def get_ratings_data(fname):
    ratings_contents = pd.read_table(fname, names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user-1, row.movie-1] = row.rating
    return ratings_contents, ratings_as_mat

def validation(recommender, pct_users_to_val, pct_items_to_val, ratings_mat):
    n_users = ratings_mat.shape[0]
    n_items = ratings_mat.shape[1]
    n_users_in_val = int(n_users * pct_users_to_val)
    n_items_in_val = int(n_items * pct_items_to_val)
    val_data = ratings_mat[:n_users_in_val, :n_items_in_val].copy()
    train_data = ratings_mat.copy()
    train_data[:n_users_in_val, :n_items_in_val] = 0
    recommender.fit(train_data)
    preds = recommender.pred_all_users()
    val_preds = preds[:n_users_in_val, :n_items_in_val]
    return(mse_sparse_with_dense(val_data, val_preds))

def mse_sparse_with_dense(sparse_mat, dense_mat):
    """
    Computes mean-squared-error between a sparse and a dense matrix.  Does not include the 0's from
    the sparse matrix in computation (treats them as missing)
    """
    #get mask of non-zero, mean-square of those, divide by count of those
    nonzero_idx = sparse_mat.nonzero()
    mse = (np.array(sparse_mat[nonzero_idx] - dense_mat[nonzero_idx])**2).mean()
    return mse

if __name__ == "__main__":
    ratings_data_fname = "/Users/dan/Desktop/recommendation-systems/data/u.data"
    ratings_data_contents, ratings_mat = get_ratings_data(ratings_data_fname)
    my_mf_rec_engine = MatrixFactorizationRec()
    print(validation(my_mf_rec_engine, .3, .3, ratings_mat))


    #my_item_item_rec_engine = ItemItemRec(neighborhood_size=75)
    #print(validation(my_item_item_rec_engine, .3, .3, ratings_mat))