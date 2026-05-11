# 🧠 NN_Labs — Lab 1: Python Data Science Foundations

> An introduction to the core Python libraries used in **Data Science**, **Machine Learning**, and **Deep Learning**.

---

## 📚 Libraries Covered

| Library | Purpose |
|---|---|
| **NumPy** | Numerical computing & multi-dimensional arrays |
| **Pandas** | Data analysis & tabular structures |
| **Matplotlib** | Data visualization & plotting |

---

## 🔢 NumPy

NumPy's core object is the **`ndarray`** (n-dimensional array) — faster and more memory-efficient than Python lists for mathematical operations.

### Creating Arrays

```python
import numpy as np

# From a Python list
arr = np.array([1, 2, 3])               # [1 2 3]

# Evenly spaced integers (start, stop, step)
arr = np.arange(0, 5, 2)               # [0 2 4]

# Evenly spaced floats (start, stop, count)
arr = np.linspace(0, 10, 3)            # [ 0.  5. 10.]

# Filled arrays
arr = np.zeros((2, 2))                 # [[0. 0.] [0. 0.]]
arr = np.ones((1, 3))                  # [[1. 1. 1.]]
```

### Random Arrays

```python
# Random integers between 0–9
arr = np.random.randint(0, 10, size=3)  # e.g. [7 2 5]

# Random floats between 0–1
arr = np.random.rand(2, 2)             # e.g. [[0.13 0.54] [0.79 0.89]]
```

### Reshaping

```python
arr = np.arange(6).reshape((2, 3))
# [[0 1 2]
#  [3 4 5]]
```

### Statistics & Indexing

```python
arr = np.array([10, 20, 30, 40])

arr.max()      # 40
arr.min()      # 10
arr.mean()     # 25.0
arr.argmax()   # index of max → 3
arr.argmin()   # index of min → 0

arr[1]         # indexing  → 20
arr[1:3]       # slicing   → [20 30]
```

---

## 🐼 Pandas

Pandas provides two main data structures: **Series** (1D) and **DataFrame** (2D).

### Series

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# a    10
# b    20
# c    30
```

### DataFrame

```python
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
#    col1  col2
# 0     1     3
# 1     2     4
```

### Loading & Inspecting Data

```python
df = pd.read_csv('data.csv')

df.head(2)    # First 2 rows
df.tail(2)    # Last 2 rows
```

### Selecting Data

```python
# By label
df.loc[0]           # Row with label 0
df.loc[:, 'age']    # 'age' column

# By integer position
df.iloc[0]          # First row
df.iloc[:, 1]       # Second column
```

### Filtering & Grouping

```python
# Filtering rows
df[df['age'] > 25]

# GroupBy + aggregation
df.groupby('category').mean()
```

### Statistical Functions

```python
df['age'].mean()     # Average
df['age'].median()   # Median
df['age'].mode()     # Most frequent
df['age'].std()      # Standard deviation
df['age'].var()      # Variance
```

---

## 📈 Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Histogram — shows data distribution
data = np.array([1, 2, 2, 3, 3, 3, 4])
plt.hist(data, bins=4)
plt.show()

# Line plot — shows relationship between variables
plt.plot([1, 2, 3], [2, 4, 1])
plt.show()
```

---

## 🚀 Typical Data Science Workflow

```
Load Data  →  Clean Data  →  Analyze Data  →  Visualize Data  →  Train ML Model
```

---

## 🧠 Key Concepts

| Concept | Meaning |
|---|---|
| `ndarray` | NumPy's n-dimensional array |
| `Series` | 1D labeled data structure (Pandas) |
| `DataFrame` | 2D table structure (Pandas) |
| Indexing | Accessing individual elements |
| Slicing | Accessing a range of elements |
| `GroupBy` | Grouping rows by a shared value |
| Histogram | Plot showing data distribution |
| Line Plot | Plot showing variable relationships |

---

## ❓ Lab MCQs

The lab includes **50 multiple-choice questions** covering:

- NumPy array creation, reshaping, and statistics
- Pandas Series, DataFrames, filtering, and GroupBy
- Matplotlib histogram and line plot usage
- Combined data science workflow concepts

---

## 🌍 Why These Libraries Matter

These three libraries form the **core Python data science stack** and are foundational for:

- 🤖 Machine Learning
- 🧠 Deep Learning
- 📊 Data Analysis
- 📈 AI Projects

---

❓ Lab 1 MCQs (NumPy, Pandas, Matplotlib)
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

# 📘 Lab 2: Introduction to TensorFlow 2.x

---

## 🎯 Lab Summary

This lab introduces **TensorFlow 2.x**, a powerful deep learning framework used to build and train neural networks.

It focuses on:

- 🔢 Tensors (data structure)
- ⚙️ Variables (trainable parameters)
- 🧮 Mathematical operations
- 📊 Computation graphs (`@tf.function`)
- 🔁 Automatic differentiation (`tf.GradientTape`)

---

# 🧠 1. What is TensorFlow 2.x?

TensorFlow is a library used for:

- 🤖 Machine Learning  
- 🧠 Deep Learning  
- 📊 Numerical computation  

## ✨ Key Feature: Eager Execution

TensorFlow 2.x runs eagerly by default:

✔ operations run immediately  
✔ easier debugging  
✔ more Python-like behavior  

---

# 📦 2. Core Data Structures

---

## 🔵 2.1 Tensor (`tf.Tensor`)

A tensor is the main data structure in TensorFlow.

👉 It is:

- immutable (cannot be changed)
- n-dimensional array

### Example

```python
import tensorflow as tf

tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

print(tensor)

## 👤 Author

**Mohamed Shaaban** — [GitHub Profile](https://github.com/Mohmed-Shaaban)
