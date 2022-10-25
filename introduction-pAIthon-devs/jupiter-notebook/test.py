import numpy as np;

X = np.arange(2, 34, 2).reshape(4, 4)

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

a = np.vstack((np.hstack((y,x)), np.hstack((x,y))))
print(a)
