#!/usr/bin/python3
msg = """

P(A∣B)= ( P(B∣A)⋅P(A) ) / P(B)

Robot Lives in G: Green and R: Red
    P(R)                   = {}
    P(G)                   = {}
    see Red   P(sR | at R) = {}
    see Green P(sG | at G) = {}
    P(R | sR)             >= {}
    P(G | sR)             >= {}
    ------
    P(sR)                  = P(R) * P(sR|R) + P(G) * P(sR|G)
"""
red, green = .5, .5
see_red_at_red, see_green_at_green= .8, .5
# ---------------
at_red_see_red   = red * see_red_at_red
at_green_see_red = green * (1 - see_green_at_green)
zum = sum([at_green_see_red, at_red_see_red])
# ------ normalize
at_red_see_red   /= zum
at_green_see_red /= zum
print(msg.format(
    round(red, 4), round(green, 4), round(see_red_at_red, 4), round(see_green_at_green, 4),
    # ------ Results
    round(at_red_see_red, 4), round(at_green_see_red, 4)))
