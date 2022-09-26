#!/usr/bin/python3
#Return the probability of A conditioned on Not B given that
# P(A|~B) = P(A)*P(B|A) / (P(A)*P(B|A)) + (1-P(A) * )
#P(A)=p0, P(B|A)=p1, and P(Not B|Not A)=p2

def f(p0,p1,p2):
#Insert your code here
    return p0*p1 / ( (p0*p1) + (1-p0)*p2 )

print(f(.1, .9, .8))
