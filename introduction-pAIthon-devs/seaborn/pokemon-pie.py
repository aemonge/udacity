
%%html
<script>
    // AUTORUN ALL CELLS ON NOTEBOOK-LOAD!
    require(
        ['base/js/namespace', 'jquery'],
        function(jupyter, $) {
            $(jupyter.events).on("kernel_ready.Kernel", function () {
                console.log("Auto-running all cells-below...");
                jupyter.actions.call('jupyter-notebook:run-all-cells-below');
                jupyter.actions.call('jupyter-notebook:save-notebook');
            });
        }
    );
</script>

# |%%--%%| <mZ4hT70dFv|VEQ5NNc7W6>
"""°°°
# Pokemon with Pie/Donut charts
> Above is the initialization
°°°"""
# |%%--%%| <VEQ5NNc7W6|6XIDhD7MOh>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %matplotlib inline

pokemon = pd.read_csv('./pokemon.csv')

# |%%--%%| <6XIDhD7MOh|glAddV0kac>
"""°°°
# Rank the pokemons by speed with a speed check of `3`
°°°"""
# |%%--%%| <glAddV0kac|TCjxPNNUdI>

inc = 3
bins = np.arange(pokemon['speed'].min(), pokemon['speed'].max() + inc, inc)
plt.hist(data = pokemon, x = 'speed', bins = bins)

# |%%--%%| <TCjxPNNUdI|v2l8llXOqc>
"""°°°
# Rank the pokemons by speed with a speed check of `10
## **but** ✳️  using seaborn and not matplotlib.
°°°"""
# |%%--%%| <v2l8llXOqc|ELJt0iOVBm>

inc = 10
bins = np.arange(pokemon['speed'].min(), pokemon['speed'].max() + inc, inc)
sb.displot(pokemon['speed'], kde = True) # kde = False -> To remove the line
#|%%--%%| <ELJt0iOVBm|kqTeQmvxAc>
"""°°°
# Sub plots
°°°"""
#|%%--%%| <kqTeQmvxAc|PAKl1fGhYa>

# Resize the chart, and have two plots side-by-side
# set a larger figure size for subplots
plt.figure(figsize = [20, 5])

# histogram on left, example of too-large bin size
# 1 row, 2 cols, subplot 1
plt.subplot(1, 2, 1)
bins = np.arange(0, pokemon['speed'].max()+4, 4)
plt.hist(data = pokemon, x = 'speed', bins = bins);

# histogram on right, example of too-small bin size
plt.subplot(1, 2, 2) # 1 row, 2 cols, subplot 2
bins = np.arange(0, pokemon['speed'].max()+1/4, 1/4)
plt.hist(data = pokemon, x = 'speed', bins = bins);

#|%%--%%| <PAKl1fGhYa|TXBIVvW3dJ>
"""°°°
## What if we remove the subplot
°°°"""
#|%%--%%| <TXBIVvW3dJ|FgBLUtWBMQ>

# Resize the chart, and have two plots side-by-side
# set a larger figure size for subplots
plt.figure(figsize = [20, 5])

# histogram on left, example of too-large bin size
# 1 row, 2 cols, subplot 1
bins = np.arange(0, pokemon['speed'].max()+4, 4)
plt.hist(data = pokemon, x = 'speed', bins = bins);

# histogram on right, example of too-small bin size
bins = np.arange(0, pokemon['speed'].max()+1/4, 1/4)
plt.hist(data = pokemon, x = 'speed', bins = bins);

#|%%--%%| <FgBLUtWBMQ|oSP3cxGtMO>
"""°°°
## Spacing out the data
°°°"""
#|%%--%%| <oSP3cxGtMO|8FnucGvSFw>

# Resize the chart, and have two plots side-by-side
# set a larger figure size for subplots
plt.figure(figsize = [20, 5])

# histogram on left, example of too-large bin size
# 1 row, 2 cols, subplot 1
bins = np.arange(0, pokemon['speed'].max()+4, 4)
plt.hist(data = pokemon, x = 'speed', bins = bins, rwidth=0.7);

# histogram on right, example of too-small bin size
bins = np.arange(0, pokemon['speed'].max()+1/4, 1/4)
plt.hist(data = pokemon, x = 'speed', bins = bins, rwidth=0.7);

