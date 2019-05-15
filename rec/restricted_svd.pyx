cimport numpy as np
import numpy as np
from surprise.utils import get_rng
from surprise.prediction_algorithms.predictions import PredictionImpossible

class RestrictedSVD():
  def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
         init_std_dev=.1, lr_all=.005,
         reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
         reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
         random_state=None, verbose=False):
    self.n_factors = n_factors
    self.n_epochs = n_epochs
    self.biased = biased
    self.init_mean = init_mean
    self.init_std_dev = init_std_dev
    self.lr_bu = lr_bu if lr_bu is not None else lr_all
    self.lr_bi = lr_bi if lr_bi is not None else lr_all
    self.lr_pu = lr_pu if lr_pu is not None else lr_all
    self.lr_qi = lr_qi if lr_qi is not None else lr_all
    self.reg_bu = reg_bu if reg_bu is not None else reg_all
    self.reg_bi = reg_bi if reg_bi is not None else reg_all
    self.reg_pu = reg_pu if reg_pu is not None else reg_all
    self.reg_qi = reg_qi if reg_qi is not None else reg_all
    self.random_state = random_state
    self.verbose = verbose

  def __call__(self, train_tuples):
    self.fit(train_tuples)
    return self.estimate

  def fit(self, train_tuples):
    self.sgd(train_tuples)
    return self

  def sgd(self, train_tuples):
    cdef np.ndarray[np.double_t] bu
    cdef np.ndarray[np.double_t] bi
    cdef np.ndarray[np.double_t, ndim=2] pu
    cdef np.ndarray[np.double_t, ndim=2] qi
    cdef int u, i, f
    cdef double r, err, dot, puf, qif
    cdef double global_mean = sum(r for u, i, r in train_tuples) / len(train_tuples)
    cdef double lr_bu = self.lr_bu
    cdef double lr_bi = self.lr_bi
    cdef double lr_pu = self.lr_pu
    cdef double lr_qi = self.lr_qi
    cdef double reg_bu = self.reg_bu
    cdef double reg_bi = self.reg_bi
    cdef double reg_pu = self.reg_pu
    cdef double reg_qi = self.reg_qi
    self.global_mean = global_mean
    rng = get_rng(self.random_state)
    self.num_users = max(u for u, i, r in train_tuples) + 1
    self.num_items = max(i for u, i, r in train_tuples) + 1
    bu = np.zeros(self.num_users, np.double)
    bi = np.zeros(self.num_items, np.double)
    pu = rng.normal(self.init_mean, self.init_std_dev,
            (self.num_users, self.n_factors))
    qi = rng.normal(self.init_mean, self.init_std_dev,
            (self.num_items, self.n_factors))
    if not self.biased:
      global_mean = 0
    for current_epoch in range(self.n_epochs):
      if self.verbose:
        print("Processing epoch {}".format(current_epoch))
      for u, i, r in train_tuples:
        # compute current error
        dot = 0  # <q_i, p_u>
        for f in range(self.n_factors):
          dot += qi[i, f] * pu[u, f]
        err = r - (global_mean + bu[u] + bi[i] + dot)

        # update biases
        if self.biased:
          bu[u] += lr_bu * (err - reg_bu * bu[u])
          bi[i] += lr_bi * (err - reg_bi * bi[i])

        # update factors
        for f in range(self.n_factors):
          puf = pu[u, f]
          qif = qi[i, f]
          pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
          qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

    self.bu = bu
    self.bi = bi
    self.pu = pu
    self.qi = qi

  def estimate(self, u, i):
    known_user = u < self.num_users if isinstance(u, int) else False
    known_item = i < self.num_items if isinstance(i, int) else False
    if self.biased:
      est = self.global_mean
      if known_user:
        est += self.bu[u]
      if known_item:
        est += self.bi[i]
      if known_user and known_item:
        est += np.dot(self.qi[i], self.pu[u])
    else:
      if known_user and known_item:
        est = np.dot(self.qi[i], self.pu[u])
      else:
        raise PredictionImpossible('Pair unkown. ' + str(u) + ', ' + str(i))
    return est
