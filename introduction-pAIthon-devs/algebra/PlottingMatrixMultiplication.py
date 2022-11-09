# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from operator import matmul
import numpy as np
import matplotlib.pyplot as plt


#|%%--%%| <1LYuJux8tN|zytPLtHk1S>
"""°°°
# Visualizing Matrix Multiplication
In the videos on *__Linear Transformation and Matrices__*, you learned how a vector can be decomposed into it's basis vectors $\hat{i}$ and $\hat{j}$.
You also learned that you can tranform a vector by multiplying that vector's $x$ and $y$ values by the *transformed* basis vectors, $\hat{i_T}$ and $\hat{j_T}$, summing their results (see *Equation 1*).

$\hspace{1cm}\textit{transformed } \vec{v} = x\mathbin{\color{green}{\hat{i_T}}} +\, y\, \mathbin{\color{red}{\hat{j_T}}} $

$\hspace{2.3cm}$*Equation 1*


You learned how this method of transforming a vector through use of the *transformed* basis vectors is the same as matrix multiplication (see *Equation 2*).

$\hspace{1cm} \begin{bmatrix} \mathbin{\color{green}a} & \mathbin{\color{red}b}\\ \mathbin{\color{green}c} & \mathbin{\color{red}d} \end{bmatrix} \begin{bmatrix} x\\ y\end{bmatrix} = x \begin{bmatrix}\mathbin{\color{green}a}\\ \mathbin{\color{green}c} \end{bmatrix} + y \begin{bmatrix} \mathbin{\color{red}b}\\ \mathbin{\color{red}d} \end{bmatrix} = \begin{bmatrix} \mathbin{\color{green}a}x + \mathbin{\color{red}b}y\\ \mathbin{\color{green}c}x + \mathbin{\color{red}d}y\end{bmatrix}$

$\hspace{4.1cm}$*Equation 2*


In this lab you will:
- Graph a vector decomposed into it's basis vectors $\hat{i}$ and $\hat{j}$
- Graph a vector transformation that uses *Equation 1*
- Demonstrate that the same vector transformation can be achieved with matrix multiplication (*Equation 2*)
°°°"""
# |%%--%%| <zytPLtHk1S|jWPkWq4WIE>
"""°°°
## Graphing a Vector $\vec{v}$ Decomposed into Basis Vectors $\vec{\hat{i}}$ and $\vec{\hat{j}}$

For the first part of the lab, we will be defining vector $\vec{v}$ as follows:

$\hspace{1cm}\vec{v} = \begin{bmatrix} -1\\ 2\end{bmatrix}$

Below is an outline that describes what is included in the Python code *below* to plot vectors $\vec{v}$, $\vec{\hat{i}}$, and $\vec{\hat{j}}$ .
1. Make both NumPy and Matlibplot python packages available using the _import_ method
&nbsp;
2. Define vector $\vec{v}$
&nbsp;
3. Define basis vector $\vec{\hat{i}}$
&nbsp;
4. Define basis vector $\vec{\hat{j}}$
&nbsp;
5. Define *__v_ihat__* as $x$ multiplied by basis vector $\vec{\hat{i}}$
&nbsp;
6. Define *__v_jhat__* as $y$ multiplied by basis vector $\vec{\hat{y}}$
&nbsp;
7. Plot vector $\vec{v}$ decomposed into *__v_ihat__* and *__v_jhat__* using Matlibplot
    1. Create a variable *__ax__* to reference the axes of the plot
    2. Plot the origin as a red dot at point 0,0 using *__ax__* and _plot_ method
    3. Plot vector *__v_ihat__* as a green *dotted* arrow with origin at 0,0 using *__ax__* and _arrow_ method
    4. Plot vector *__v_jhat__* as a red *dotted* arrow with origin at tip of *__v_ihat__* using *__ax__* and _arrow_ method
    5. Plot vector $\vec{v}$ as a blue arrow with origin at 0,0 using *__ax__* and _arrow_ method
    6. Format x-axis
        1. Set limits using _xlim_ method
        2. Set major tick marks using *__ax__* and *set_xticks* method
    7. Format y-axis
        1. Set limits using _ylim_ method
        2. Set major tick marks using *__ax__* and *set_yticks* method
    8. Create the gridlines using _grid_ method
    9. Display the plot using _show_ method
°°°"""
# |%%--%%| <jWPkWq4WIE|mqGWOnQtWp>

# Define vector v
v = np.array([-1,2])

# Define basis vector i_hat as unit vector
i_hat = np.array([1,0])

# Define basis vector j_hat as unit vector
j_hat = np.array([0,1])

# Define v_ihat - as v[0](x) multiplied by basis vector ihat
v_ihat = v[0] * i_hat

# Define v_jhat_t - as v[1](y) multiplied by basis vector jhat
v_jhat = v[1] * j_hat

# Plot that graphically shows vector v (color='b') - whose position can be
# decomposed into v_ihat and v_jhat

# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0,0,'or')


# Plots vector v_ihat as dotted green arrow starting at origin 0,0
ax.arrow(0, 0, *v_ihat, color='g', linestyle='dotted', linewidth=2.5, head_width=0.30,
         head_length=0.35)

# Plots vector v_jhat as dotted red arrow starting at origin defined by v_ihat
ax.arrow(v_ihat[0], v_ihat[1], *v_jhat, color='r', linestyle='dotted', linewidth=2.5,
         head_width=0.30, head_length=0.35)

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.30, head_length=0.35)


# Sets limit for plot for x-axis
plt.xlim(-4, 2)

# Set major ticks for x-axis
major_xticks = np.arange(-4, 2)
ax.set_xticks(major_xticks)


# Sets limit for plot for y-axis
plt.ylim(-2, 4)

# Set major ticks for y-axis
major_yticks = np.arange(-2, 4)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(b=True, which='major')

# Displays final plot
plt.show()

# |%%--%%| <mqGWOnQtWp|0JXFyOJAdD>
"""°°°
## *Transforming* a Vector using *Transformed* Vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$
For this part of the lab, we will plot the results of *transforming* vector $\vec{v}$ using *transformed* vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$. Vectors $\vec{v}$, $\vec{\hat{i_T}}$, and $\vec{\hat{j_T}}$ have been defined below.


$\hspace{1cm}\vec{v} = \begin{bmatrix} -1\\ 2\end{bmatrix}$

$\hspace{1cm}\vec{\mathbin{\color{green}{\hat{i_T}}}} = \begin{bmatrix}\mathbin{\color{green}3}\\ \mathbin{\color{green}1} \end{bmatrix}$

$\hspace{1cm}\vec{\mathbin{\color{red}{\hat{j_T}}}} = \begin{bmatrix}\mathbin{\color{red}1}\\ \mathbin{\color{red}2} \end{bmatrix}$

### TODO: Computing and Plotting *Transformed* Vector $\vec{v_T}$ using Vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$
For this part of the lab you will be creating *transformed* vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$ and using them to *transform* vector $\vec{v}$ using *Equation 1* above.

1. Define vector $\vec{\hat{i_T}}$ by replacing $x$ and $y$ with $3$ and $1$ (see *__TODO 1.:__*).
&nbsp;

2. Define vector $\vec{\hat{j_T}}$ by replacing $x$ and $y$ with $1$ and $2$ (see *__TODO 2.:__*).
&nbsp;

3. Define vector $\vec{v_T}$ by adding vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$ (see *__TODO 3.:__*).
&nbsp;

4. Plot vector $\vec{v_T}$ by copying the _ax.arrow(...)_ statement for vector $\vec{v}$ and changing _color = 'b'_ in the _ax.arrow(...)_ statement to plot vector $\vec{v_T}$ as blue colored vector (see *__TODO 4.:__*).
&nbsp;

*__Notice that__*:

- To *run* your code:
    - Click on the Save icon (disk icon right under *'File'* in the menu bar above), to save your work.
    - Select *'Kernel'* and *'Restart & Run All'*, to run your code.

°°°"""
# |%%--%%| <0JXFyOJAdD|kjJxzTQTEN>

# Define vector v
v = np.array([-1, 2])

# where x=3 and y=1 instead of x=1 and y=0
ihat_t = np.array([3, 1])

# where x=1 and y=2 instead of x=0 and y=1
jhat_t = np.array([1, 2])

# Define v_ihat_t - as v[0](x) multiplied by transformed vector ihat
v_ihat_t = v[0] * ihat_t

# Define v_jhat_t - as v[1](y) multiplied by transformed vector jhat
v_jhat_t = v[1] * jhat_t

# vector v_ihat_t added to vector v_jhat_t
v_t = v_ihat_t + v_ihat_t


# Plot that graphically shows vector v (color='skyblue') can be transformed
# into transformed vector v (v_trfm - color='b') by adding v[0]*transformed
# vector ihat to v[0]*transformed vector jhat


# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0,0,'or')


# Plots vector v_ihat_t as dotted green arrow starting at origin 0,0
ax.arrow(0, 0, *v_ihat_t, color='g', linestyle='dotted', linewidth=2.5, head_width=0.30,
         head_length=0.35)

# Plots vector v_jhat_t as dotted red arrow starting at origin defined by v_ihat
ax.arrow(v_ihat_t[0], v_ihat_t[1], *v_jhat_t, color='r', linestyle='dotted', linewidth=2.5,
         head_width=0.30, head_length=0.35)

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='skyblue', linewidth=2.5, head_width=0.30, head_length=0.35)

# vector v's ax.arrow() statement above as template for the plot
plt.plot(data= v_t, color='b')


# Sets limit for plot for x-axis
plt.xlim(-4, 2)

# Set major ticks for x-axis
major_xticks = np.arange(-4, 2)
ax.set_xticks(major_xticks)


# Sets limit for plot for y-axis
plt.ylim(-2, 4)

# Set major ticks for y-axis
major_yticks = np.arange(-2, 4)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(b=True, which='major')

# Displays final plot
plt.show()

# |%%--%%| <kjJxzTQTEN|Joojb3D6wy>
"""°°°
### Solution for Computing and Plotting *Transformed* Vector $\vec{v_T}$
Your output from above should match the output below. If you need any help or want to check your answer, feel free to check out the solution notebook by clicking [here](PlottingMatrixMultiplicationSolution.ipynb#TODO:-Computing-and-Plotting-Transformed-Vector-$\vec{v_T}$-using-Vectors-$\vec{\hat{i_T}}$-and-$\vec{\hat{j_T}}$).

<img src="linearMappingLab_GraphingTransformedVector.png" height=300 width=350 />


### Solution Video for Computing and Plotting *Transformed* Vector $\vec{v_T}$
The solution video can be found in the **Linear Mapping Lab Solution** section. You may want to open another browser window to allow you to easily toggle between the Vector's Lab Jupyter Notebook and the solution videos for this lab.
°°°"""
# |%%--%%| <Joojb3D6wy|6oT7JmLh89>
"""°°°
## Matrix Multiplication
For this part of the lab, we will demonstrate that the same vector transformation from the section above can be achieved with matrix multiplication (*Equation 2*). Vectors $\vec{v}$ and $\vec{ij}$ have been defined below.

$\hspace{1cm}\vec{v} = \begin{bmatrix} -1\\ 2\end{bmatrix}$

$\hspace{1cm}\vec{ij} = \begin{bmatrix} \mathbin{\color{green}3} & \mathbin{\color{red}1}\\ \mathbin{\color{green}1} & \mathbin{\color{red}2}\end{bmatrix}$

### TODO: Matrix Multiplication
For this part of the lab, define *__transformed__* vector **$\vec{v_T}$** using the [function _**matmul**_](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html) for multiplying 2x2 matrix **$\vec{ij}$** and vector **$\vec{v}$**.

1. Replace **None** below with code that defines *__transformed__* vector **$\vec{v_T}$** using the *__matmul__* function to multiply matrix **$\vec{ij}$** and vector **$\vec{v}$** (see *__TODO 1.__*)
&nbsp;

*__Notice that__*:

- The alias _**np**_ was used to with the import of the Numpy package; therefore, use the _**np**_ alias when you call the _**matmul**_ function below.


- To *run* your code:
    - Click on the Save icon (disk icon right under *'File'* in the menu bar above), to save your work.
    - Select *'Kernel'* and *'Restart & Run All'*, to run your code.

°°°"""
# |%%--%%| <6oT7JmLh89|ddA9F74J8A>

# Define vector v
v = np.array([-1,2])

# Define 2x2 matrix ij
ij = np.array([[3, 1],[1, 2]])

# TODO 1.: Demonstrate getting v_trfm by matrix multiplication
# by using matmul function to multiply 2x2 matrix ij by vector v
# to compute the transformed vector v (v_t)
v_t = matmul(ij, v)

# Prints vectors v, ij, and v_t
print("\nMatrix ij:", ij, "\nVector v:", v, "\nTransformed Vector v_t:", v_t, sep="\n")


# |%%--%%| <ddA9F74J8A|uC0o3CSTf7>
"""°°°
### Solution for Matrix Multiplication
Your output from above for *transformed* vector $\vec{v_T}$ should match the solution below. Notice that in NumPy vectors are written horizontally so that the *[-1  2]* from above is the way vector $\vec{v}$ will be defined.
If you need any help or want to check your answer, feel free to check out the solution notebook by clicking [here](PlottingMatrixMultiplicationSolution.ipynb#TODO:-Matrix-Multiplication).

With this matrix multiplication you have completed the computation in *Equation 2* using *transformed* vectors $\vec{\hat{i_T}}$ and $\vec{\hat{j_T}}$ (see below).


$\hspace{1cm} \begin{bmatrix} \mathbin{\color{green}3} & \mathbin{\color{red}1}\\ \mathbin{\color{green}1} & \mathbin{\color{red}2}\end{bmatrix} \begin{bmatrix} -1\\ 2\end{bmatrix} = -1 \begin{bmatrix}\mathbin{\color{green}3}\\ \mathbin{\color{green}1} \end{bmatrix} + 2 \begin{bmatrix} \mathbin{\color{red}1}\\ \mathbin{\color{red}2} \end{bmatrix} = \begin{bmatrix} {-1}{*}\mathbin{\color{green}3} +\,2{*}\mathbin{\color{red}1}\\ {-1}{*}\mathbin{\color{green}1} +\, 2{*}\mathbin{\color{red}2}\end{bmatrix} = \begin{bmatrix} -1\\ 3\end{bmatrix}$


You expect the following value for *transformed* $\vec{v_T}$, it will be written by NumPy as *[-1  3]*:

$\hspace{1cm}\textit{tranformed }\ \vec{v_T} = \begin{bmatrix} -1\\ 3\end{bmatrix}$

### Solution Video for Matrix Multiplication
The solution video can be found in the **Linear Mapping Lab Solution** section. You may want to open another browser window to allow you to easily toggle between the Vector's Lab Jupyter Notebook and the solution videos for this lab.
°°°"""
