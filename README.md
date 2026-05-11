🌟 Lab 1: Python, NumPy, Pandas, and Matplotlib Review

This lab is an introduction to the most important Python libraries used in:

📊 Data Science
🤖 Machine Learning
🧠 Deep Learning
📈 Data Analysis

The lab focuses on:

Library	Purpose
NumPy	Numerical computing & arrays
Pandas	Data analysis & tables
Matplotlib	Data visualization & plotting
1️⃣ NumPy 🧮

NumPy is used for:

fast mathematical operations
arrays
matrices
numerical computation

Main object:

ndarray

(ndimensional array)

2️⃣ Creating Arrays with np.array() 📦

Converts Python list into NumPy array.

Example
import numpy as np

my_list = [1, 2, 3]

arr = np.array(my_list)

print(arr)
Output
[1 2 3]
Explanation 🧠

Python list:

[1,2,3]

becomes:

array([1,2,3])

NumPy arrays are:

faster
optimized for math
3️⃣ np.arange() 🔢

Creates evenly spaced numbers.

Syntax:

np.arange(start, stop, step)
Example
arr = np.arange(0, 5, 2)

print(arr)
Output
[0 2 4]
Explanation

Starts at:

0

Adds:

2

Stops BEFORE:

5

Sequence:

0 → 2 → 4
4️⃣ np.zeros() ⚪

Creates array filled with zeros.

Example
arr = np.zeros((2,2))

print(arr)
Output
[[0. 0.]
 [0. 0.]]
Explanation

Shape:

(2,2)

means:

2 rows
2 columns
5️⃣ np.ones() 🟢

Creates array filled with ones.

Example
arr = np.ones((1,3))

print(arr)
Output
[[1. 1. 1.]]
6️⃣ Random Integers 🎲

np.random.randint()

Generates random integers.

Example
arr = np.random.randint(0,10,size=3)

print(arr)
Example Output
[7 2 5]
Explanation

Generate:

3 random numbers
between 0 and 9
7️⃣ Random Decimal Numbers 🎯

np.random.rand()

Creates random floats between:

0 and 1
Example
arr = np.random.rand(2,2)

print(arr)
Example Output
[[0.13 0.54]
 [0.79 0.89]]
8️⃣ np.linspace() 📏

Creates evenly spaced numbers.

Example
arr = np.linspace(0,10,3)

print(arr)
Output
[ 0.  5. 10.]
Explanation

Generate:

3 equally spaced values
from 0 to 10
9️⃣ Reshaping Arrays 🔄

reshape()

Changes array dimensions.

Example
arr = np.arange(6)

reshaped = arr.reshape((2,3))

print(reshaped)
Output
[[0 1 2]
 [3 4 5]]
Explanation

Original:

[0 1 2 3 4 5]

becomes:

2 rows × 3 columns
🔟 Array Statistics 📊
Maximum
arr.max()
Minimum
arr.min()
Mean (Average)
arr.mean()
Example
arr = np.array([1,2,3,4,5])

print(arr.max())
print(arr.min())
print(arr.mean())
Output
5
1
3.0
1️⃣1️⃣ argmax() and argmin() 📍

Return index positions.

Example
arr = np.array([10,50,20])

print(arr.argmax())
print(arr.argmin())
Output
1
0
Explanation

Largest value:

50

located at index:

1
1️⃣2️⃣ Indexing and Slicing 🔍

Access elements inside arrays.

Example
arr = np.array([10,20,30,40])

print(arr[1])

Output:

20
Slicing
print(arr[1:3])

Output:

[20 30]
1️⃣3️⃣ Pandas 🐼

Pandas is used for:

data analysis
tables
CSV files
statistics

Main structures:

Series
DataFrame
1️⃣4️⃣ Pandas Series 📏

1-dimensional labeled array.

Example
import pandas as pd

s = pd.Series([10,20,30],
              index=['a','b','c'])

print(s)
Output
a    10
b    20
c    30
Explanation

Like dictionary + array together.

1️⃣5️⃣ DataFrame 📋

2D table structure.

Like Excel sheet.

Example
data = {
    'col1':[1,2],
    'col2':[3,4]
}

df = pd.DataFrame(data)

print(df)
Output
   col1  col2
0     1     3
1     2     4
1️⃣6️⃣ Reading CSV Files 📂

Use:

pd.read_csv()
Example
df = pd.read_csv('data.csv')
Explanation

Loads CSV file into DataFrame.

1️⃣7️⃣ head() and tail() 👀

Show first or last rows.

Example
print(df.head(2))

Shows first 2 rows.

1️⃣8️⃣ .loc[] 📍

Access data by LABEL.

Example
print(df.loc[0])

Gets row with label 0.

Example
print(df.loc[:, 'age'])

Gets age column.

1️⃣9️⃣ .iloc[] 🔢

Access data using integer positions.

Example
print(df.iloc[0])

First row.

Example
print(df.iloc[:,1])

Second column.

2️⃣0️⃣ Filtering Data 🔍

Select rows using conditions.

Example
print(df[df['age'] > 25])

Returns rows where:

age > 25
2️⃣1️⃣ GroupBy 🔥

Groups similar rows together.

Example
df.groupby('category').mean()
Explanation

Group rows by:

category

Then compute:

mean

for each group.

2️⃣2️⃣ Statistical Functions 📊

Pandas supports:

mean()
median()
mode()
std()
var()
Example
print(df['age'].mean())
2️⃣3️⃣ Matplotlib 📈

Matplotlib is used for:

plotting graphs
visualization
charts
2️⃣4️⃣ Histogram 📊

Shows distribution of data.

Example
import matplotlib.pyplot as plt
import numpy as np

data = np.array([1,2,2,3,3,3,4])

plt.hist(data,bins=4)

plt.show()
Explanation

Histogram counts:

how many values fall in each range.
2️⃣5️⃣ Line Plot 📉

Shows relationship between variables.

Example
plt.plot([1,2,3],[2,4,1])

plt.show()
Explanation

Points:

(1,2)
(2,4)
(3,1)

connected with lines.

2️⃣6️⃣ Full Workflow in Data Science 🚀

Typical process:

Load Data
    ↓
Clean Data
    ↓
Analyze Data
    ↓
Visualize Data
    ↓
Train ML Model
2️⃣7️⃣ Libraries Relationship 🌍
Library	Used For
NumPy	Math + arrays
Pandas	Data tables
Matplotlib	Visualization
2️⃣8️⃣ Important Concepts 🧠
Concept	Meaning
ndarray	NumPy array
Series	1D labeled data
DataFrame	2D table
Indexing	Access elements
Slicing	Access subarrays
GroupBy	Group rows
Histogram	Distribution plot
Line Plot	Relationship plot
2️⃣9️⃣ Why This Lab Important? 🌟

These libraries are foundations of:

Machine Learning
Deep Learning
Data Science
AI projects

You will use them in:

preprocessing
analysis
visualization
model training
3️⃣0️⃣ Final Big Picture 🌍
NumPy
 ↓
Numerical operations

Pandas
 ↓
Data analysis

Matplotlib
 ↓
Visualization

Together they form the core Python data science stack 🚀

📘 Lab 1 MCQs (NumPy, Pandas, Matplotlib)
<details> <summary><strong>1️⃣ What is the primary purpose of NumPy?</strong></summary>
A) Web development
B) Numerical computing and arrays
C) Game design
D) Database management

✅ Answer: B

</details>
<details> <summary><strong>2️⃣ Which NumPy object represents an n-dimensional array?</strong></summary>
A) Series
B) DataFrame
C) ndarray
D) Tensor

✅ Answer: C

</details>
<details> <summary><strong>3️⃣ What will be the output of the following code?</strong></summary>
import numpy as np
arr = np.array([1,2,3])
print(arr)
A) (1,2,3)
B) [1 2 3]
C) {1,2,3}
D) Error

✅ Answer: B

</details>
<details> <summary><strong>4️⃣ Which function creates evenly spaced values within an interval?</strong></summary>
A) np.zeros()
B) np.ones()
C) np.arange()
D) np.reshape()

✅ Answer: C

</details>
<details> <summary><strong>5️⃣ What is the output?</strong></summary>
np.arange(0,5,2)
A) [0 1 2 3 4]
B) [0 2 4]
C) [2 4 6]
D) [0 2 4 6]

✅ Answer: B

</details>
<details> <summary><strong>6️⃣ Which function creates an array filled with zeros?</strong></summary>
A) np.empty()
B) np.full()
C) np.zeros()
D) np.array()

✅ Answer: C

</details>
<details> <summary><strong>7️⃣ What is the shape of this array?</strong></summary>
np.zeros((2,3))
A) 2 rows and 3 columns
B) 3 rows and 2 columns
C) 6 rows
D) 1 row and 6 columns

✅ Answer: A

</details>
<details> <summary><strong>8️⃣ Which function creates an array of ones?</strong></summary>
A) np.ones()
B) np.zeros()
C) np.identity()
D) np.random()

✅ Answer: A

</details>
<details> <summary><strong>9️⃣ What does np.random.randint() generate?</strong></summary>
A) Floating-point numbers
B) Strings
C) Random integers
D) Boolean values

✅ Answer: C

</details>
<details> <summary><strong>🔟 Which function generates random values between 0 and 1?</strong></summary>
A) np.random.randint()
B) np.random.rand()
C) np.arange()
D) np.linspace()

✅ Answer: B

</details>
<details> <summary><strong>1️⃣1️⃣ What does np.linspace() do?</strong></summary>
A) Creates random values
B) Creates evenly spaced values
C) Creates identity matrix
D) Creates zeros

✅ Answer: B

</details>
<details> <summary><strong>1️⃣2️⃣ What is the output?</strong></summary>
np.linspace(0,10,3)
A) [0 10]
B) [0 5 10]
C) [1 5 9]
D) [0 2 4]

✅ Answer: B

</details>
<details> <summary><strong>1️⃣3️⃣ What does reshape() do?</strong></summary>
A) Deletes data
B) Changes array shape
C) Sorts values
D) Adds elements

✅ Answer: B

</details>
<details> <summary><strong>1️⃣4️⃣ Which shape is valid for reshaping an array of 6 elements?</strong></summary>
A) (2,3)
B) (4,4)
C) (5,2)
D) (7,1)

✅ Answer: A

</details>
<details> <summary><strong>1️⃣5️⃣ What does arr.max() return?</strong></summary>
A) Minimum value
B) Mean value
C) Maximum value
D) Index of maximum value

✅ Answer: C

</details>
<details> <summary><strong>1️⃣6️⃣ What does arr.mean() compute?</strong></summary>
A) Median
B) Mode
C) Average
D) Variance

✅ Answer: C

</details>
<details> <summary><strong>1️⃣7️⃣ What does argmax() return?</strong></summary>
A) Largest value
B) Index of largest value
C) Smallest value
D) Mean value

✅ Answer: B

</details>
<details> <summary><strong>1️⃣8️⃣ Which operation accesses specific elements of an array?</strong></summary>
A) Plotting
B) Reshaping
C) Indexing
D) Pooling

✅ Answer: C

</details>
<details> <summary><strong>1️⃣9️⃣ What is slicing used for?</strong></summary>
A) Deleting arrays
B) Accessing subarrays
C) Random generation
D) Sorting arrays

✅ Answer: B

</details>
<details> <summary><strong>2️⃣0️⃣ Which library is mainly used for data analysis?</strong></summary>
A) TensorFlow
B) Pandas
C) OpenCV
D) Flask

✅ Answer: B

</details>
<details> <summary><strong>2️⃣1️⃣ What is a Pandas Series?</strong></summary>
A) 2D table
B) Image object
C) 1D labeled array
D) Neural network

✅ Answer: C

</details>
<details> <summary><strong>2️⃣2️⃣ What is a DataFrame?</strong></summary>
A) One-dimensional array
B) Two-dimensional labeled table
C) Neural network layer
D) Random generator

✅ Answer: B

</details>
<details> <summary><strong>2️⃣3️⃣ Which function reads CSV files?</strong></summary>
A) pd.load()
B) pd.read_csv()
C) pd.import()
D) pd.open()

✅ Answer: B

</details>
<details> <summary><strong>2️⃣4️⃣ What does df.head() display?</strong></summary>
A) Last rows
B) Random rows
C) First rows
D) Middle rows

✅ Answer: C

</details>
<details> <summary><strong>2️⃣5️⃣ What does df.tail() display?</strong></summary>
A) First rows
B) Last rows
C) Random rows
D) Column names

✅ Answer: B

</details>
<details> <summary><strong>2️⃣6️⃣ Which method uses labels for indexing?</strong></summary>
A) iloc[]
B) loc[]
C) mean()
D) tail()

✅ Answer: B

</details>
<details> <summary><strong>2️⃣7️⃣ Which method uses integer positions?</strong></summary>
A) loc[]
B) iloc[]
C) groupby()
D) plot()

✅ Answer: B

</details>
<details> <summary><strong>2️⃣8️⃣ What does groupby() do?</strong></summary>
A) Deletes groups
B) Groups rows based on values
C) Creates plots
D) Reshapes arrays

✅ Answer: B

</details>
<details> <summary><strong>2️⃣9️⃣ Which statistical function computes average?</strong></summary>
A) median()
B) std()
C) mean()
D) var()

✅ Answer: C

</details>
<details> <summary><strong>3️⃣0️⃣ Which function calculates standard deviation?</strong></summary>
A) mean()
B) std()
C) var()
D) median()

✅ Answer: B

</details>
<details> <summary><strong>3️⃣1️⃣ Which function calculates variance?</strong></summary>
A) std()
B) mode()
C) var()
D) mean()

✅ Answer: C

</details>
<details> <summary><strong>3️⃣2️⃣ Which library is used for plotting graphs?</strong></summary>
A) NumPy
B) Pandas
C) Matplotlib
D) SciPy

✅ Answer: C

</details>
<details> <summary><strong>3️⃣3️⃣ What does plt.hist() create?</strong></summary>
A) Line plot
B) Pie chart
C) Histogram
D) Scatter plot

✅ Answer: C

</details>
<details> <summary><strong>3️⃣4️⃣ What is a histogram used for?</strong></summary>
A) Data distribution visualization
B) Sorting arrays
C) Neural network training
D) CSV reading

✅ Answer: A

</details>
<details> <summary><strong>3️⃣5️⃣ Which function creates line plots?</strong></summary>
A) plt.hist()
B) plt.plot()
C) plt.array()
D) plt.group()

✅ Answer: B

</details>
<details> <summary><strong>3️⃣6️⃣ What does plt.show() do?</strong></summary>
A) Saves file
B) Displays plot
C) Deletes plot
D) Reads plot

✅ Answer: B

</details>
<details> <summary><strong>3️⃣7️⃣ Which library provides fast numerical operations?</strong></summary>
A) Pandas
B) Matplotlib
C) NumPy
D) Flask

✅ Answer: C

</details>
<details> <summary><strong>3️⃣8️⃣ Which structure is similar to an Excel sheet?</strong></summary>
A) ndarray
B) Series
C) DataFrame
D) Tuple

✅ Answer: C

</details>
<details> <summary><strong>3️⃣9️⃣ Which function creates random floating-point values?</strong></summary>
A) randint()
B) rand()
C) arange()
D) zeros()

✅ Answer: B

</details>
<details> <summary><strong>4️⃣0️⃣ What is the purpose of Matplotlib?</strong></summary>
A) Database management
B) Web development
C) Data visualization
D) Audio processing

✅ Answer: C

</details>
<details> <summary><strong>4️⃣1️⃣ Which operation changes array dimensions?</strong></summary>
A) reshape()
B) mean()
C) groupby()
D) loc[]

✅ Answer: A

</details>
<details> <summary><strong>4️⃣2️⃣ What does df.iloc[:,1] select?</strong></summary>
A) First row
B) Second column
C) Last row
D) Entire DataFrame

✅ Answer: B

</details>
<details> <summary><strong>4️⃣3️⃣ Which function returns the index of the minimum value?</strong></summary>
A) argmax()
B) min()
C) argmin()
D) mean()

✅ Answer: C

</details>
<details> <summary><strong>4️⃣4️⃣ Which NumPy function creates arrays with evenly spaced values including decimals?</strong></summary>
A) arange()
B) linspace()
C) randint()
D) zeros()

✅ Answer: B

</details>
<details> <summary><strong>4️⃣5️⃣ Which command creates a 2×2 matrix of ones?</strong></summary>
A) np.zeros((2,2))
B) np.ones((2,2))
C) np.arange((2,2))
D) np.rand((2,2))

✅ Answer: B

</details>
<details> <summary><strong>4️⃣6️⃣ Which Pandas object is one-dimensional?</strong></summary>
A) DataFrame
B) ndarray
C) Series
D) Matrix

✅ Answer: C

</details>
<details> <summary><strong>4️⃣7️⃣ What does filtering in Pandas mean?</strong></summary>
A) Removing libraries
B) Selecting rows based on conditions
C) Plotting graphs
D) Reshaping arrays

✅ Answer: B

</details>
<details> <summary><strong>4️⃣8️⃣ Which command accesses the first element of a NumPy array arr?</strong></summary>
A) arr(0)
B) arr[0]
C) arr{0}
D) arr.first()

✅ Answer: B

</details>
<details> <summary><strong>4️⃣9️⃣ Which library is commonly used together with NumPy and Pandas for plotting?</strong></summary>
A) OpenCV
B) Flask
C) Matplotlib
D) Django

✅ Answer: C

</details>
<details> <summary><strong>5️⃣0️⃣ What is the main purpose of this lab?</strong></summary>
A) Build websites
B) Learn Python data science libraries
C) Develop mobile apps
D) Create databases

✅ Answer: B

</details>
