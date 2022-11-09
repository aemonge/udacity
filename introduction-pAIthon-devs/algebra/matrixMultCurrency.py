import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#|%%--%%| <j1L1fY9h5p|ZwOJosvNjj>
"""°°°
# Currency Conversion with Matrix Multiplication

In this notebook you will solve a currency problem using matrix multiplication and the python package [NumPy](http://www.numpy.org/). This demonstration is provided to prepare you for using matrix multiplication to solve more complex problems.

## Currency Conversion Problem

Over the years you have traveled to eight different countries and just happen to have leftover local currency from each of your trips.
You are planning to return to one of the eight countries, but you aren't sure which one just yet.
You are waiting to find out which will have the cheapest airfare.

In preparation, for the trip you *will* want convert *all* your local currency into the currency local of the place you will be traveling to.
Therefore, to double check the bank's conversion of your currency, you want to compute the total amount of currency you would expect for each of the eight countries.
To compute the conversion you first need to import a matrix that contains the currency conversion rates for each of the eight countries. The data we will be use comes from the [Overview Matrix of Exchange Rates from Bloomberg Cross-Rates _Overall Chart_](https://www.bloomberg.com/markets/currencies/cross-rates) on January, 10 2018.
°°°"""
# |%%--%%| <ZwOJosvNjj|5rXNRNudBF>
"""°°°
<img src="currencyProbImage.png" height=300 width=750>


You can think about this problem as taking a _vector of **inputs**_ (the currencies from the 8 countries) and applying a _matrix of **weights**_ (the conversion rates matrix) to these inputs to produce a _vector of **outputs**_ (total amount of currency for each country) using matrix multiplication with the NumPy package.
°°°"""
# |%%--%%| <5rXNRNudBF|JFM8jqUUHE>
"""°°°
### Coding the Currency Conversion Problem
First you will need to create the _**inputs** vector_ that holds the currency you have from the eight countries into a numpy vector. To begin, first import the NumPy package and then use the package to create a vector from a list. Next we convert the vector into a pandas dataframe so that it will print out nicely below with column labels to indicate the country the currency amount is associated to.
°°°"""
# |%%--%%| <JFM8jqUUHE|35YHtKh3R3>

# Creates numpy vector from a list to represent money (inputs) vector.
money = np.asarray([70, 100, 20, 80, 40, 70, 60, 100])

# Creates pandas dataframe with column labels(currency_label) from the numpy vector for printing.
currency_label = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "HKD"]
money_df = pd.DataFrame(data=money, index=currency_label, columns=["Amounts"])
print("Inputs Vector:")
money_df.T


# |%%--%%| <35YHtKh3R3|KK6EfaGsUo>
"""°°°
Next we need to create the _**weights** matrix_ by importing the currency conversion rates matrix. We will use python package [Pandas](https://pandas.pydata.org/) to quickly read in the matrix and approriately assign row and colunm labels. Additionally, we define a variable **_path_** to define the location of the currency conversion matrix. The code below imports this weights matrix, converts the dataframe into a numpy matrix, and displays its content to help you determine how to solve the problem using matrix multiplication.
°°°"""
# |%%--%%| <KK6EfaGsUo|BeqCaGuLmh>

# Sets path variable to the 'path' of the CSV file that contains the conversion rates(weights) matrix.
path = %pwd

# Imports conversion rates(weights) matrix as a pandas dataframe.
conversion_rates_df = pd.read_csv(path+"/currencyConversionMatrix.csv",header=0,index_col=0)

# Creates numpy matrix from a pandas dataframe to create the conversion rates(weights) matrix.
conversion_rates = conversion_rates_df.values

# Prints conversion rates matrix.
print("Weights Matrix:")
conversion_rates_df


# |%%--%%| <BeqCaGuLmh|bSHy8G6oXU>
"""°°°
The _**weights** matrix_ above provides the conversion rates between each of the eight countries. For example, in row 1, column 1 the value **1.0000** represents the conversion rate from US dollars to US dollars. In row 2, column 1 the value **1.1956** represents that 1 Euro is worth **1.1956** US dollars.  In row 1, column 2 the value **0.8364** represents that 1 US dollar is only worth **0.8364** Euro.
°°°"""
# |%%--%%| <bSHy8G6oXU|zqJFOKjTlu>
"""°°°
The _**outputs** vector_ is computed below using matrix multiplication. The numpy package provides the [function _**matmul**_](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html) for multiplying two matrices (or a vector and a matrix). Below you will find the equation for matrix multiplication as it applies to AI, where the _**inputs** vector_($x_{1}...x_{n}$) multiplied by the _**weights** matrix_($w_{11}...w_{nm}$) to compute the _**outputs** vector_($y_{1}...y_{m}$).

$\hspace{4cm} \begin{bmatrix} x_{1}&x_{2}&...&x_{n}\end{bmatrix} \begin{bmatrix} w_{11}&w_{12}&...&w_{1m}\\ w_{21}&w_{22}&...&w_{2m}\\ ...&...&...&... \\ w_{n1}&w_{n2}&...&w_{nm}\end{bmatrix} = \begin{bmatrix} y_{1}&y_{2}&...&y_{m}\end{bmatrix}$

The example matrix multiplication below, has $n$ as 4 in **inputs** and **weights** and $m$ as 3 in **weights** and **outputs**.

$\hspace{4cm} \begin{bmatrix} 10 & 2 & 1 & 5\end{bmatrix} \begin{bmatrix} 1 & 20 & 7\\ 3 & 15 & 6 \\ 2 & 5 & 12 \\ 4 & 25 & 9 \end{bmatrix} = \begin{bmatrix} 38 & 360 & 139 \end{bmatrix}$

As seen with the example above, matrix multiplication resulting matrix(_**outputs** vector_) will have same row dimension as the first matrix(_**inputs** vector_) and the same column dimension as the second matrix(_**weights** matrix_). With the currency example the number of columns in the inputs and weights matrices are the same, but this won't always be the case in AI.
°°°"""
# |%%--%%| <zqJFOKjTlu|26JLnbkrIc>
"""°°°
## TODO: Matrix Multiplication
Replace the **None** below with code that uses the [function _**matmul**_](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html) for multiplying **money** and **conversion_rates** to compute the vector **money_totals**. Recall that we used the alias _**np**_ when we imported the Numpy package above, so be certain to use the _**np**_ alias when calling the _**matmul**_ function below. Additionally, be certain to select _'Cell'_ and _'Run All'_ to check the code you insert below.
°°°"""
# |%%--%%| <26JLnbkrIc|8RnPqm0FfK>

money_totals = np.matmul(money, conversion_rates)

# Converts the resulting money totals vector into a dataframe for printing.
money_totals_df = pd.DataFrame(data = money_totals, index = currency_label, columns = ["Money Totals"])
print("Outputs Vector:")
money_totals_df.T


# |%%--%%| <8RnPqm0FfK|ewt0tD2Aq6>
"""°°°
### Solution for Currrency Conversion with Matrix Multiplication
Your output from above should match the **Money Totals** displayed below. If you need any help or want to check your answer, feel free to check out the solution notebook by clicking [here](matrixMultCurrencySolution.ipynb). The results can be interperted as converting all the currency to US dollars(**USD**) would provide **454.28** US dollars, converting all the currency to Euros(**EUR**) would provide **379.96** Euros, and etc.

<img src="money_totals.png" height=225 width=563>

### Solution Video for Currrency Conversion with Matrix Multiplication
The solution video can be found in the **Linear Mapping Lab Solution** section. You may want to open another browser window to allow you to easily toggle between the Vector's Lab Jupyter Notebook and the solution videos for this lab.
°°°"""
