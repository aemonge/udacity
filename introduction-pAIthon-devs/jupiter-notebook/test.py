import numpy as np;
def basic():
    X = np.arange(2, 34, 2).reshape(4, 4)

    print(' _______________ \n')
    print('X:\n', X, '\n')
    Y = np.delete(np.delete(X, [2], axis=1), [2], axis=0)
    print('Y:\n', Y, '\n')
    Y = np.append(Y, np.array([[7]*3]), axis=0)
    print('Y:\n', Y, '\n')
    print(np.append(Y, np.array([[3],[5],[9], [13]]), axis=1), '\n')

    x = np.array([[0]*4]*4)
    print('x:\n', x, '\n')

    y = np.array([[8]*4]*4)
    print('y:\n', y, '\n')
    print(' _______________ \n')

    w = np.hstack((x, y))
    print(w)
    w = np.hstack((y, x))
    print(w)

    o = np.vstack((y, x))
    print(o)
    o = np.vstack((x, y))
    print(o)

    print(' _______________ ')

def basic2():
    x = np.array([[0]*4]*4)
    y = np.array([[8]*4]*4)
    print(' _______________ \n')
    a = np.vstack((np.hstack((y,x)), np.hstack((x,y))))
    print(a)
    print(' _______________ ')

    j = a[3:5, 3:5]
    print(j)

    print(' _______________ ')
    print(j*4)

    print(' _______________ ')
    print(np.diag(j*4))
    print(np.diag(a, k=2))
    print(np.diag(a, k=2))
    print(' _______________ ')

def threven(x):
    return x % 3 == 0

def basic3():
    x = np.arange(1,10).reshape(3,3)
    print(x, '\n')
    print(x[x > 5])
    print(x[x % 2 == 0])
    print(x[x % 2 == 1])
    print(x[threven(x)])
    print(' _______________ \n')
    print(np.union1d(x[x % 2 == 0], x[x %2 == 1]))
    print(np.intersect1d(x[x % 2 == 1], x[threven(x)]))
    print(np.setdiff1d(x[x % 2 == 1], x[threven(x)]))
    print(' _______________ \n')

def statistical():
    X = np.array([[1,2,3,4]*4]).reshape(4,4)
    print(X)
    print('', X.mean(), X.sum(), X.std(), X.max(), X.min())
    print(X-3)
    print(X//3)
    print(' _______________ \n')
    X = np.array([[n]*4 for n in range(1,5)]).transpose()
    X1 = np.ones((4,4)) * np.arange(1,5)
    print(X)
    print(X1)

def notebook():
    X = np.random.randint(5001, size=(1000,20))
    # ave_cols = np.array([np.std(i) for i in X.T])
    ave_cols = np.std(X, axis=0)
    std_cols = np.std(X, axis=0)

    X_norm = (X - ave_cols) / std_cols
    # print("Average of all the values of X_norm: ", np.min(X_norm, axis=0).mean())
    # print(np.random.permutation(50))
    row_indices = np.random.permutation(X_norm.shape[0])

    six = row_indices[0:int(len(row_indices) * .6)]
    between = int(len(row_indices) * .6) + int(len(row_indices) * .2)
    twen = row_indices[int(len(row_indices) * .6):between]
    tween = row_indices[between:]
    # print(six, '\n')
    # print(twen, '\n')
    # print(tween, '\n')

    X_train = [X_norm[i] for i in six]
    X_crossVal = [X_norm[i] for i in twen]
    X_test = [X_norm[i] for i in tween]
    print(X_train, '\n')
    print(X_crossVal, '\n')
    print(X_test, '\n')


notebook()
