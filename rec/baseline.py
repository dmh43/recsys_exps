from collections import defaultdict

import numpy as np
from surprise import AlgoBase

class Baseline(AlgoBase):

  def __init__(self):
    AlgoBase.__init__(self)

  def fit(self, trainset):
    AlgoBase.fit(self, trainset)
    self.u_mean = defaultdict(lambda: trainset.global_mean)
    for u, ratings in trainset.ur.items():
      self.u_mean[u] = np.mean([rating for i, rating in ratings])

    return self

  def estimate(self, u, i):
    return self.u_mean[u]
