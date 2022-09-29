#!/usr/bin/python3
# a=[190,170,165,180,165]
# a=[2400, 125, 148, 160, 110, 325, 180]
# a=[4,3,32,33,4,32,3,38,4, 39, 36]
# a.sort()
def three_m(a):
    print("Mean: ", sum(a)/len(a))
    a.sort()
    print("Median: ", a[len(a)//2])
    b = {}
    for i in a:
        b[i] = b.get(i, 0)+1
    c = [(val, key) for key, val in b.items()]
    c.sort()
    print("Mode: ", c[-1][1])

# three_m([5,9,100,9,97,6,9,98,9])
three_m([3,9,3,8,2,9,1,9,2,4])
