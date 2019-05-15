import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise.model_selection import cross_validate

from rec.baseline import Baseline
from rec.slope_one_full import SlopeOneFull
from rec.slope_one import SlopeOne
from rec.slope_one_weighted import SlopeOneWeighted
from rec.slope_one_var import SlopeOneVar

# def get_fold_idxs(num_elems, num_folds):
#   permutation = list(range(num_elems))
#   rn.shuffle(permutation)
#   fold_size = num_elems // num_folds
#   return [np.array(permutation[fold_num * fold_size:(fold_num + 1) * fold_size])
#           for fold_num in range(num_folds)]

def main():
  data = Dataset.load_builtin('ml-100k')
  baseline = Baseline()
  cross_validate(baseline, data, verbose=True)
  slope_one_full = SlopeOneFull()
  cross_validate(slope_one_full, data, verbose=True)
  slope_one = SlopeOne()
  cross_validate(slope_one, data, verbose=True)
  slope_one_weighted = SlopeOneWeighted()
  cross_validate(slope_one_weighted, data, verbose=True)
  slope_one_var = SlopeOneVar()
  cross_validate(slope_one_var, data, verbose=True)

# def main():
#   ratings = pd.read_csv('../ml-latest-small/ratings.csv')
#   user_id_to_idx = {user_id: idx for idx, user_id in zip(range(len(ratings)), ratings.userId.unique())}
#   movie_id_to_idx = {movie_id: idx for idx, movie_id in zip(range(len(ratings)), ratings.movieId.unique())}
#   num_folds = 5
#   folds = get_fold_idxs(len(ratings), num_folds)
#   for test_fold_idx in range(num_folds):
#     test = ratings.iloc[folds[test_fold_idx]]
#     train = ratings.iloc[np.concatenate([fold for i, fold in enumerate(folds) if i != test_fold_idx])]
#     known =


if __name__ == "__main__": main()
