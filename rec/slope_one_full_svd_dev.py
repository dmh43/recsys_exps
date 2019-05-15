from collections import defaultdict
from itertools import combinations

import numpy as np
from surprise import AlgoBase

from .restricted_svd import RestrictedSVD

class SlopeOneFullSVDDev(AlgoBase):
  def __init__(self):
    AlgoBase.__init__(self)
    self.rating_lookup_by_user = None
    self.u_mean = None
    self.dev = None
    self.cnts = None
    self.dev_svd = None

  def fit(self, trainset):
    AlgoBase.fit(self, trainset)
    self.rating_lookup_by_user = {}
    self.u_mean = defaultdict(lambda: trainset.global_mean)
    self.dev = defaultdict(lambda: defaultdict(int))
    self.cnts = defaultdict(lambda: defaultdict(int))
    for u, ratings in trainset.ur.items():
      self.u_mean[u] = np.mean([rating for i, rating in ratings])
      ratings_lookup = {item: rating for item, rating in ratings}
      self.rating_lookup_by_user[u] = ratings_lookup
      for item_1, item_2 in combinations(ratings_lookup.keys(), 2):
        self.dev[item_1][item_2] += ratings_lookup[item_1] - ratings_lookup[item_2]
        self.dev[item_2][item_1] += ratings_lookup[item_2] - ratings_lookup[item_1]
        self.cnts[item_1][item_2] += 1
        self.cnts[item_2][item_1] += 1
    dev_tuples = (i_1, i_2, self.dev[i_1][i_2]
                  for i_2 in self.dev[i_1]
                  for i_1 in self.dev)
    self.dev_svd = RestrictedSVD()(dev_tuples)
    return self

  def estimate(self, u, i):
    devs = [self.dev_svd(i, item)
            for item in self.rating_lookup_by_user[u]]
    ratings = self.rating_lookup_by_user[u].values()
    cnts = [self.cnts[i][item]
            for item in self.rating_lookup_by_user[u]]
    total_cnts = sum(cnts)
    if total_cnts > 0:
      return sum(dev + cnt * rating
                 for cnt, dev, rating in zip(cnts, devs, ratings)) / total_cnts
    else:
      return self.u_mean[u]
