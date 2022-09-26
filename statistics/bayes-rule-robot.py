#!/usr/bin/python3

def f(p0, p1, p2):
    return p1*p0 + (1-p0)*(1-p2)

print(f(.1, .9, .8))
