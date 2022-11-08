# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

from solutions_biv import scatterplot_solution_1, scatterplot_solution_2

# |%%--%%| <GvwelYiFBA|XrPhX4x23k>
"""°°°
In this workspace, you'll make use of this data set describing various car attributes, such as fuel efficiency. The cars in this dataset represent about 3900 sedans tested by the EPA from 2013 to 2018. This dataset is a trimmed-down version of the data found [here](https://catalog.data.gov/dataset/fuel-economy-data).
°°°"""
# |%%--%%| <XrPhX4x23k|CgRnGL9mnQ>

fuel_econ = pd.read_csv('./data/fuel_econ.csv')
fuel_econ.head()

# |%%--%%| <CgRnGL9mnQ|ShdwvndTjU>
"""°°°
### **TO DO 1**:
Let's look at the relationship between fuel mileage ratings for city vs. highway driving, as stored in the 'city' and 'highway' variables (in miles per gallon, or mpg). **Use a _scatter plot_ to depict the data.**
1. What is the general relationship between these variables?
2. Are there any points that appear unusual against these trends?
°°°"""
# |%%--%%| <ShdwvndTjU|rhHSmyGLrF>

# YOUR CODE HERE

# |%%--%%| <rhHSmyGLrF|0WHxMjf3nj>
"""°°°
### Expected Output
°°°"""
# |%%--%%| <0WHxMjf3nj|mUonaSmkwA>

# run this cell to check your work against ours
scatterplot_solution_1()

# |%%--%%| <mUonaSmkwA|smIeZYdrIS>
"""°°°
### **TO DO 2**:
Let's look at the relationship between two other numeric variables. How does the engine size relate to a car's CO2 footprint? The 'displ' variable has the former (in liters), while the 'co2' variable has the latter (in grams per mile). **Use a heat map to depict the data.** How strong is this trend?
°°°"""
# |%%--%%| <smIeZYdrIS|s9bnDHiE0S>

# YOUR CODE HERE

# |%%--%%| <s9bnDHiE0S|vrSvUBMAIU>
"""°°°
### Expected Output
°°°"""
# |%%--%%| <vrSvUBMAIU|7iP451i1Gm>

# run this cell to check your work against ours
scatterplot_solution_2()
