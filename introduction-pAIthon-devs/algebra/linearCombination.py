# Makes Python package NumPy available using import method
import numpy as np
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#|%%--%%| <VNmIWxhjqI|7HsM4NfWGc>
"""°°°
# Linear Combination

In this notebook you will learn how to solve linear combination problems using the python package [NumPy](http://www.numpy.org/) and its linear algebra subpackage [linalg](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html). This lab is provided to prepare you for the linear algebra you will be using in Neural Networks.

## Determining a Vector's span

From the lesson on linear combination, recall that the set of all possible vectors that you can reach with a linear combination of a given pair of vectors is called the span of those two vectors. Let's say we are given the pair of vectors $\vec{v}$ and $\vec{w}$, and we want to determine if a third vector $\vec{t}$ is within their span. If vector $\vec{t}$ is determined to be within their span, this means that $\vec{t}$ can be written as a linear combination of the pair of vectors $\vec{v}$ and $\vec{w}$.

This could be written as:

$\hspace{1cm}a\vec{v} + b\vec{w} = \vec{t}$,$\hspace{0.3cm}$where $\vec{v}$ and $\vec{w}$ are multiplied by scalars $a$ and $b$ and then added together to produce vector $\vec{t}$.

$\hspace{1.2cm}$*Equation 1*

This means if we can find a set of values for the scalars $a$ and $b$ that make *equation 1* true, then $\vec{t}$ is within the span of $\vec{v}$ and $\vec{w}$. Otherwise, if there is **no** set of values of the scalars $a$ and $b$ that make *equation 1* true, then $\vec{t}$ is **not** within their span.


°°°"""
# |%%--%%| <7HsM4NfWGc|MhtNocTESx>
"""°°°

We can determine a vector's span computationally using NumPy's linear algebra subpackage [linalg](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html). Below we will go through an example below.

If the vectors have the following values:

$\hspace{1cm}\vec{v} = \begin{bmatrix} 1\\ 3\end{bmatrix}$
$\hspace{0.3cm}\vec{w} = \begin{bmatrix} 2\\ 5\end{bmatrix}$
$\hspace{0.3cm}\vec{t} = \begin{bmatrix} 4\\ 11\end{bmatrix}$

We can rewrite $a\vec{v} + b\vec{w} = \vec{t}$ as:

$\hspace{1cm} a \begin{bmatrix} 1\\ 3\end{bmatrix} + b \begin{bmatrix} 2\\ 5\end{bmatrix} = \begin{bmatrix} 4\\ 11\end{bmatrix}$

In a linear algebra class you might have solved this problem by hand, using reduced row echelon form and rewriting *equation 1* as the augmented matrix. We have provided the augmented matrix for *equation 1* below.

$
\hspace{1cm}
\left[
\begin{array}{cc|c}
1 & 2  & 4 \\
3 & 5 & 11 \\
\end{array}
\right]
$

Notice that the augmented matrix's right side contains the vector $\vec{t}$. This is the vector that we are trying to determine if it's contained within the span of the other vectors, $\vec{v}$ and $\vec{w}$. Those other vectors whose span we are checking, compose the left side of the augmented matrix.
°°°"""
# |%%--%%| <MhtNocTESx|0zouvcZ0ZV>
"""°°°
## Determining Span Computationally
Instead of solving the problem by hand, we are going to solve this problem computationally using NumPy's linear algebra subpackage ([linalg](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)) .

**Steps to Determine a Vector's Span Computationally**:

1. Make the [NumPy](http://www.numpy.org/) Python package available using the import method
&nbsp;
2. Create right and left sides of the augmented matrix
    1. Create a [NumPy vector](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.creation.html) $\vec{t}$ to represent the right side of the augmented matrix.
    2. Create a [NumPy matrix](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.creation.html) named $vw$ that represents the left side of the augmented matrix ($\vec{v}$ and $\vec{w}$)
    &nbsp;
3. Use NumPy's [**linalg.solve** function](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve) to check a vector's span computationally by solving for the scalars that make the equation true. For this lab you will be using the *__check_vector_span__* function you will defined below.

With the Python code below, you will have completed steps **1** and **2** from the list above.
°°°"""
# |%%--%%| <0zouvcZ0ZV|gwkfzv1yvu>

# Creates matrix t (right side of the augmented matrix).
t = np.array([4, 11])

# Creates matrix vw (left side of the augmented matrix).
vw = np.array([[1,2],[3,5]])

# Prints vw and t
print("\nMatrix `vw`:", vw, "\nVector `t`:", t, sep="\n")

# |%%--%%| <gwkfzv1yvu|2Zpbzjg0kH>
"""°°°
### TODO: Check Vector's Span with *__linalg.solve__* function
You will be using NumPy's [**linalg.solve** function](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve)
to check if vector $\vec{t}$ is within the span of the other two vectors, $\vec{v}$ and $\vec{w}$. To complete this task, you will be inserting your code into the function *__check_vector_span__* that is defined in the coding cell below.

**Note the Following**:
- Use the [**linalg.solve** function](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve) to solve for the scalars (**vector_of_scalars**) that will make *equation 1* above
**true**, *ONLY* when the vector that's being checked (**vector_to_check**) is within the span of the other vectors (**set_of_vectors**).


- *Otherwise*, the vector (**vector_to_check**) is **not** within the span and an empty vector is returned.


Below you will find the definitions of the parameters and returned variable to help you with this task.

- **Function Parameters:**
    - **set_of_vectors** is the left side of the augmented matrix. This parameter represents the set of vectors (e.g. $\vec{v}$ and $\vec{w}$) whose span you are checking.
    - **vector_to_check** is the right side of the augmented matrix. This parameter represents the vector (e.g. $\vec{t}$) that's checked to see if it's within the span of the vectors in **set_of_vectors**.


- **Returned variable:**
    - **vector_of_scalars** contains the scalars that will solve the equations **"if"** the checked vector is within the span of the set of vectors. Otherwise, this will be an empty vector.

With the Python code below, you will be completing step **3** of *determine a vector's span computationally*. In the code below (see *__TODO:__*), you will need to replace **None** below with code that uses [**linalg.solve** function](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve) to solve for the scalars (*vector_of_scalars*).
°°°"""
# |%%--%%| <2Zpbzjg0kH|OfCcUhAsNH>

def check_vector_span(set_of_vectors, vector_to_check):
    # Creates an empty vector of correct size
    vector_of_scalars = np.asarray([None]*set_of_vectors.shape[0])

    # Solves for the scalars that make the equation true if vector is within the span
    try:
        vector_of_scalars = np.linalg.solve(vw, t)
        if not vector_of_scalars is None:
            print("\nVector is within span.\nScalars in s:", vector_of_scalars)
    # Handles the cases when the vector is NOT within the span
    except Exception as exception_type:
        if str(exception_type) == "Singular matrix":
            print("\nNo single solution\nVector is NOT within span")
        else:
            print("\nUnexpected Exception Error:", exception_type)
    return vector_of_scalars


# |%%--%%| <OfCcUhAsNH|S2e7niFHZL>
"""°°°
### Checking *check_vector_span* by Solving for Scalars
Let's see if $\vec{t}$ is within the span of vectors $\vec{v}$ and $\vec{w}$ and check the code you added to the *check_vector_span* function above.

*Notice that*:

- There is code added to check two additional sets of vectors (see the additional vectors below).


- To *run* your code:
    - Click on the Save icon (disk icon right under *'File'* in the menu bar above), to save your work.
    - Select *'Kernel'* and *'Restart & Run All'*, to run your code.


The second set of vectors have the follwing values and augmented matrix:

$\hspace{1cm}\vec{v2} = \begin{bmatrix} 1\\ 2\end{bmatrix}$
$\hspace{0.3cm}\vec{w2} = \begin{bmatrix} 2\\ 4\end{bmatrix}$
$\hspace{0.3cm}\vec{t2} = \begin{bmatrix} 6\\ 12\end{bmatrix}$  $\hspace{0.9cm}
\left[
\begin{array}{cc|c}
1 & 2  & 6 \\
2 & 4 & 12 \\
\end{array}
\right]
$

The third set of vectors have the follwing values and augmented matrix:

$\hspace{1cm}\vec{v3} = \begin{bmatrix} 1\\ 1\end{bmatrix}$
$\hspace{0.3cm}\vec{w3} = \begin{bmatrix} 2\\ 2\end{bmatrix}$
$\hspace{0.3cm}\vec{t3} = \begin{bmatrix} 6\\ 10\end{bmatrix}$  $\hspace{0.9cm}
\left[
\begin{array}{cc|c}
1 & 2  & 6 \\
1 & 2 & 10 \\
\end{array}
\right]
$

With the Python code below, you will be checking the function you created with step **3** of *determine a vector's span computationally*.
°°°"""
# |%%--%%| <S2e7niFHZL|xgd5w2Upri>

# Call to check_vector_span to check vectors in Equation 1
print("\nEquation 1:\n Matrix vw:", vw, "\nVector t:", t, sep="\n")
s = check_vector_span(vw,t)

# Call to check a new set of vectors vw2 and t2
vw2 = np.array([[1, 2], [2, 4]])
t2 = np.array([6, 12])
print("\nNew Vectors:\n Matrix vw2:", vw2, "\nVector t2:", t2, sep="\n")
# Call to check_vector_span
s2 = check_vector_span(vw2,t2)

# Call to check a new set of vectors vw3 and t3
vw3 = np.array([[1, 2], [1, 2]])
t3 = np.array([6, 10])
print("\nNew Vectors:\n Matrix vw3:", vw3, "\nVector t3:", t3, sep="\n")
# Call to check_vector_span
s3 = check_vector_span(vw3,t3)


# |%%--%%| <xgd5w2Upri|fUU3nNq1ee>
"""°°°
### Solution for Checking *check_vector_span* by Solving for Scalars
Your output from above should match the output below. If you need any help or want to check your answer, feel free to check out the solution notebook by clicking [here](linearCombinationSolution.ipynb).

You will notice that with *Equation 1*, $a\vec{v} + b\vec{w} = \vec{t}$, vector $\vec{t}$ was within the span of $\vec{v}$ and $\vec{w}$ such that scalars had the following values $a = 2$ and $b = 1$:

$\hspace{1cm} 2 \begin{bmatrix} 1\\ 3\end{bmatrix} + 1 \begin{bmatrix} 2\\ 5\end{bmatrix} = \begin{bmatrix} 4\\ 11\end{bmatrix}$

You will also notice that both the two new sets of vectors $\vec{t2}$ and $\vec{t3}$ were **not** within the span; such that, there were no value of the scalars that would provide a solution to the equation.


<img src="linearCombinationAnswer1.png" height=270 width=676>


### Solution Video for Checking *check_vector_span* by Solving for Scalars
The solution video can be found in the **Linear Combination Lab Solution** section. You may want to open another browser window to allow you to easily toggle between the Vector's Lab Jupyter Notebook and the solution videos for this lab.
°°°"""
# |%%--%%| <fUU3nNq1ee|O36YhfnPWs>
"""°°°
## System of Equations
All the cases that we tested above could have also been written as a system of two equations, where we are trying to solve for the values of the scalars that make both equations true. For the system of equations, scalar $a$ becomes $x$ and scalar $b$ becomes $y$.

So *Equation 1*, $a\vec{v} + b\vec{w} = \vec{t}$, which could be written as:

$\hspace{1cm} a \begin{bmatrix} 1\\ 3\end{bmatrix} + b \begin{bmatrix} 2\\ 5\end{bmatrix} = \begin{bmatrix} 4\\ 11\end{bmatrix}$, where $a = 2$ and $b = 1$

Becomes the following system of two equations that is written as:

$\hspace{1cm} \begin{array}{rcl} x + 2y & = & 4 \\ 3x + 5y  & = & 11 \end{array}$, where $x = 2$ and $y = 1$

*__Notice that__*:

- The vectors $\vec{v}$ and $\vec{w}$ become the coefficients on the *left* side of both equations.

- The vector $\vec{t}$ become the solution on the *right* side of both equations.

- The scalar $a$ becomes the variable $x$ and the scalar $b$ becomes variable $y$ in both equations.

- Each of the equations can be represented by a line plotted in two dimensions.


Systems of equations always result in *one* of *three* possible solutions. One occurs when the vector is within the span and there's a solution, like with the example above. The other two cases can occur when the vector is **not** within span. Below we describe each of the three cases.


### Case 1 - Single solution
We could have considered *Equation 1* as the following system of two equations:

$\hspace{1cm} \begin{array}{rcl} x + 2y & = & 4 \\ 3x + 5y  & = & 11 \end{array}$, where $x = 2$ and $y = 1$

We would have used the same method to solve this system of equations for $x$ and $y$, as we did to determine vector $\vec{t}$'s span. This means when the vector is within the span, there is a single solution to the system of equations. This single solution graphically is represented where the lines intersect (the red dot on the graph below).
°°°"""
# |%%--%%| <O36YhfnPWs|aVTtxoqOms>

# %matplotlib inline
import matplotlib.pyplot as plt
plt.plot([4,0],[0,2],'b',linewidth=3)
plt.plot([3.6667,0],[0,2.2],'c-.',linewidth=3)
plt.plot([2],[1],'ro',linewidth=3)
plt.xlabel('Single Solution')
plt.show()

# |%%--%%| <aVTtxoqOms|aAelpcbq70>
"""°°°
### Case 2 - Infinite Solutions
The second case is when there are infinite values that the scalars could have taken because at least two of the equations are redundant. In our case, our only two equations are redundant and they represent the same line (see graph below).

This second case is represented by $vw2$ and $t2$ where:

$\hspace{1cm} \begin{array}{rcl} x + 2y & = & 6 \\ 2x + 4y  & = & 12 \end{array}$, where **any** $x$ and $y$ makes this *__true__* because the equations are redundant.
°°°"""
# |%%--%%| <aAelpcbq70|hHaRLqNWDG>

import matplotlib.pyplot as plt
plt.plot([6,0],[0,3],'b',linewidth=5)
plt.plot([1,4,6,0],[2.5,1,0,3],'c-.',linewidth=2)
plt.xlabel('Redundant Equations')
plt.show()

# |%%--%%| <hHaRLqNWDG|IyNc8fE06E>
"""°°°
### Case 3 - No Solution
The third case is that there are **no** values that the scalars could have taken that would have simutaneously solved all equations.
In our case, our only two equations are represented by parallel lines because they have no solution (see graph below).

This third case is represented by $vw3$ and $t3$ where:

$\hspace{1cm} \begin{array}{rcl} x + 2y & = & 6 \\ x + 2y  & = & 10 \end{array}$, where **no** $x$ and $y$ make this true.
°°°"""
# |%%--%%| <IyNc8fE06E|vlKjVGd0nv>

import matplotlib.pyplot as plt
plt.plot([10,0],[0,5],'b',linewidth=3)
plt.plot([0,6],[3,0],'c-.',linewidth=3)
plt.xlabel('No Solution')
plt.show()

# |%%--%%| <vlKjVGd0nv|Zf6AXtzKjb>
"""°°°
### Importance of the Lab

Understanding how to check a vector's span and how to solve a system of equations are important foundations for solving more complex problems we will work with in AI.
°°°"""
