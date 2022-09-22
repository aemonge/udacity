#!/usr/bin/python3
msg = """

P(A∣B)= ( P(B∣A)⋅P(A) ) / P(B)

Robot Lives in G: Green and R: Red
    P(R)                   = P(G) = 0.5
    see Red   P(sR | at R) = 0.8
    see Green P(sG | at G) = 0.8
    P(R | sR)             >= {}
    P(G | sR)             >= {}
    ------
    P(sR)                  = P(R) * P(sR|R) + P(G) * P(sR|G)
"""
red, green = .5, .5
see_red_at_red, see_greenat_green= .8, .8
# ---------------
see_red_at_green = 1 - see_red_at_red
see_red = red * see_red_at_red + green * see_red_at_green
# ---------------
at_red_see_green = (see_red_at_red * (red) / see_red)
at_green_see_red = (1 - at_red_see_green) # + (green)

print(msg.format(round(at_red_see_green, 5), round(at_green_see_red, 5)))
