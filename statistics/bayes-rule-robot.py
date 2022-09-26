#!/usr/bin/python3

def f(p0, p1, p2):
    """P(C) = P(C)*P(Pos|C) + P(C)*P(Neg|C)"""
    """P(C|Pos) = P(C)*P(Pos|C) + """
    return p0*p1 / (p0*p1 + (1-p0)*(1-p2))

print(f(.1, .9, .8))
