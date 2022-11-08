# Data Visualization
Add the `bins` always after understanding the data, which can usually be done by `.describe()`

Understand the different between quantitate and qualitative data.

## Python Libraries
> 🐼 pandas is not a chart nor plot library it self, but is extremely useful for vector and matrix math operations with tons of useful sugar in it.

### Matplotlib as plt
The simplest plotter library. I would choose a CLI plotter instead of this one, and for more complex jump directly into seaborn

### seaborn as sb
Useful for colorful and complex data, also support multiple type in the same chart; as a histogram with a average line included.


## Type of charts and plots
As a general lesson, add another chart in the same chart to understand differences between a single data. In other words, accumulative data from blondes and brunette woman can drive a different conclusion than when splitting the initial data into more category as hair color.

### Plain old data
- Histogram.
- Scatter plot.
- Pie chart.

### Complex Data
> Add to the old charts the following:
- Transparency, and color as a heat map.
- Jitter: Random data deviation points around the (x,y) => (x+∂, y+∂) ∂:random_deviation

#### Violin Plot
Useful for when the X axis is representing a category, making a violin shape when the scatter dots are transformed into lines that filled the space in them.

#### Box Plots
Useful for displaying the outlier form the violin plots, and displaying the exact median values as a line. So choose this one as a summary for the data displayed on the violin.
