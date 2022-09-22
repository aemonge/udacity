#!/usr/bin/python3
msg="""
Sebastian travels so much, he can't remember where he is
    P(away)      = {}
    P(home)      = {}
    P(rain|home) = {}
    P(rain|away) = {}
"""
away, home = .6, .4
rains_home  = .01
rains_away  = .3

print(msg.format(away, home, rains_home, rains_away))

ans="""
  Sebastian wakes up and it's raining
    P(home|rain) = {}
"""

mid="""
    P(home, rain) =

"""
home_raining = home * rains_home
away_raining = away * rains_away
zum = sum([away_raining, home_raining])

print(ans.format(home_raining/zum))
