import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
# TODO: Use the magic of the np.array vector multiplication, istead of the python thingi
def softmax(L):
    r = [];
    for l in L:
        r.append(np.power(np.e, l))
    total = np.sum(r);
    return [ round((np.power(np.e, l) / total), 2) for l in L];

softmax([2, 1, 0])
