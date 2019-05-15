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


if __name__ == "__main__": main()
