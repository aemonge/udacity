
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sb
# %matplotlib inline

pokemon = pd.read_csv('./pokemon.csv')

# |%%--%%| <k1LwgcKMag|VEQ5NNc7W6>
"""°°°
# Pokemon to see the data outliers and manage them
°°°"""
# |%%--%%| <VEQ5NNc7W6|6XIDhD7MOh>

delta = 0.5;
# pokemon['height'].min() => 0.1
bins = np.arange(pokemon['height'].min(), pokemon['height'].max() + delta, delta)
plt.hist(data=pokemon, x = 'height', bins = bins)

# |%%--%%| <6XIDhD7MOh|9NOwsfhc38>
"""°°°
## Slim the data to focus on a [0, 6] limits
°°°"""
# |%%--%%| <9NOwsfhc38|MSCqnvRfJy>

plt.hist(data=pokemon, x = 'height', bins = bins)
plt.xlim(0, 6)

# |%%--%%| <MSCqnvRfJy|jlIunTK8pd>
"""°°°
# Scale the data to a logarithmically scale to appreciate the data better
## Basic
Note that you can either scale the full data, or simply the visualization. Making only the visualization change, is
easier to read since noone mentally knows that 40k = log10(`4,6`).
°°°"""
# |%%--%%| <jlIunTK8pd|KUPn9RCGQL>

delta = 40
bins = np.arange(pokemon['weight'].min(), pokemon['weight'].max() + delta, delta)
plt.hist(data = pokemon, x = 'weight', bins=bins)

# |%%--%%| <KUPn9RCGQL|a7iPZTTs1P>
"""°°°
## Scaled and tidy
°°°"""
# |%%--%%| <a7iPZTTs1P|rJSKkXjpUh>

delta = 0.1 # logDesc = np.log10(pokemon['weight']).describe()
min = math.floor(np.log10(pokemon['weight']).min())
max = math.ceil(np.log10(pokemon['weight']).max())
bins = 10 ** np.arange(min, max + delta, delta)
plt.hist(data = pokemon, x = 'weight', bins=bins, rwidth=0.88)
plt.xscale('log')

# |%%--%%| <rJSKkXjpUh|BoioXvKecd>
"""°°°
## Hard coding ticks, to add threes to see the data better
Just to understand the X axis better, and add readability.
°°°"""
# |%%--%%| <BoioXvKecd|SFrAkWv90j>

ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
labels = [ "{}".format(v) for v in ticks ]
plt.hist(data = pokemon, x = 'weight', bins=bins, rwidth=0.88)
plt.xscale('log')
plt.xticks(ticks, labels)
