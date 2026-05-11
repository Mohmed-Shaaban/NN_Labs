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

## 👤 Author

**Mohamed Shaaban** — [GitHub Profile](https://github.com/Mohmed-Shaaban)
