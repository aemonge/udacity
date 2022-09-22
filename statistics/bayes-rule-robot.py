#!/usr/bin/python3
msg = """
A: Red   Box
B: Green Box
C" Green Box

Robot Lives in one box
    P(A)                    = {}
    P(B)                    = {}
    P(C)                    = {}
    ------
    P(R|A) ~see red in A~   = {}
    P(G|B) ~see green in B~ = {}
    P(G|C) ~see green in C~ = {}
"""

ans= """
    ----------------------------
    P(A,R)                  = {}
"""
red, green1, green2 = .3333, .3333, .3333
see_red_a, see_green_b, see_green_c =  .9, .9, .9
print(msg.format(
    round(red, 4), round(green1, 4), round(green2, 4),
    round(see_red_a, 4), round(see_green_b, 4), round(see_green_c, 4)
))
# ---------------
a_see_red = (red * see_red_a)
print(ans.format(
    round(a_see_red, 4)
))
