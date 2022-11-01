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

# |%%--%%| <29JsqOcNVw|Rmk6agrneL>
"""°°°
## Introduction
In workspaces like this one, you will be able to practice visualization techniques you've seen in the course materials. In this particular Jupyter Notebook, you'll practice creating single-variable plots for categorical data.

The cells where you are expected to contribute, are highlighted with **TO DO** markdown.
°°°"""
# |%%--%%| <Rmk6agrneL|Qh5Vay6Hpu>

# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

# The `solutions_univ.py` is a Python file available in the Notebook server that contains solution to the TO DO tasks.
# The solution to each task is present in a separate function in the `solutions_univ.py` file.
# Do not refer to the file untill you attempt to write code yourself.
# from solutions_univ import bar_chart_solution_1, bar_chart_solution_2

# |%%--%%| <Qh5Vay6Hpu|V8DCjWOQRt>
"""°°°
## About the Dataset
In this workspace, you'll be working with the dataset comprised of attributes of creatures in the video game series Pokémon. The data was assembled from the database of information found in this [GitHub repository](https://github.com/veekun/pokedex/tree/master/pokedex/data/csv).
°°°"""
# |%%--%%| <V8DCjWOQRt|LbjZBdcRdb>

pokemon = pd.read_csv('./pokemon.csv')
pokemon.head()

# |%%--%%| <LbjZBdcRdb|qtwrb8gCzF>
"""°°°
### **DONE Task 1**
1. Explore the `pokemon` dataframe, and try to understand the significance of each of its column.
2. There have been quite a few Pokémon introduced over the series' history. Display the count of Pokémon introduced in each generation? Create a _bar chart_ of these frequencies using the 'generation_id' column.
°°°"""
# |%%--%%| <qtwrb8gCzF|CiDVCoJJKk>

base_color = sb.color_palette()[2]
sb.countplot(data=pokemon, x='generation_id', color = base_color, order = (pokemon['generation_id'].value_counts()).index)

# |%%--%%| <CiDVCoJJKk|qluscTxNU1>
"""°°°
Once you've created your chart, run the cell below to check the output from our solution. **Your visualization does not need to be exactly the same as ours, but it should be able to come up with the same conclusions.**

### **TO DO Task 2**
1. Each Pokémon species has either `type_1`, `type_2` or both `types` that play a part in its offensive and defensive capabilities. The code below creates a new dataframe `pkmn_types` that club the rows of both `type_1` and `type_2`, so that the resulting dataframe has **new** column, `type_level`.

**Display, how frequent is each type?**


The function below will do the following in the pokemon dataframe *out of place*:
1. Select the 'id', and 'species' columns from pokemon.
2. Remove the 'type_1', 'type_2' columns from pokemon
3. Add a new column 'type_level' that can have a value either 'type_1' or 'type_2'
4. Add another column 'type' that will contain the actual value contained in the 'type_1', 'type_2' columns. For example, the first row in the pokemon dataframe having `id=1`  and `species=bulbasaur` will now occur twice in the resulting dataframe after the `melt()` operation. The first occurrence will have `type=grass`, whereas, the second occurrence will have `type=poison`.
°°°"""
# |%%--%%| <qluscTxNU1|C17Dshb280>

pkmn_types = pokemon.melt(id_vars = ['id','species'],
                          value_vars = ['type_1', 'type_2'],
                          var_name = 'type_level', value_name = 'type').dropna()
pkmn_types.head(5)

# |%%--%%| <C17Dshb280|B0uOjrcnRX>

t2 = pkmn_types.query("type_level == 'type_2'")
t2.head()

# |%%--%%| <B0uOjrcnRX|YRQc9G2UdK>

sb.countplot(data=pkmn_types, y="type", color = base_color, order = (pkmn_types['type'].value_counts()).index)

# |%%--%%| <YRQc9G2UdK|kNPZvz4Pbf>
"""°°°
2. Your task is to use this dataframe to create a _relative frequency_ plot of the proportion of Pokémon with each type, _sorted_ from most frequent to least. **Hint**: The sum across bars should be greater than 100%, since many Pokémon have two types. Keep this in mind when considering a denominator to compute relative frequencies.
°°°"""
# |%%--%%| <kNPZvz4Pbf|V204w5lqZM>

n_pokemons = pkmn_types['type'].value_counts().sum()
max_type_count = pkmn_types['type'].value_counts()[0] # .max()
max_prop = max_type_count / n_pokemons
tick_names = [ "{:0.2f}".format(v) for v in np.arange(0, max_prop, .02) ] # a complex .to_s fn
plt.xticks(np.arange(0, max_prop, .02) * n_pokemons, tick_names)
plt.xlabel('% proportion')
