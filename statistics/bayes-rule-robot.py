#!/usr/bin/python3

def f(p0, p1, p2):
    """P(H|c1) = p1"""
    """P(H|c2) = p2"""

    return (p0*p1 + (1 -p0)*p2)

print(f(.3, .5, .9))
