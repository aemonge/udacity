import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
  r = 0
  for ix, p in enumerate(P):
    y = Y[ix]
    r-= y * np.log(p) + (1 - y) * np.log(1 - p)
  return r
