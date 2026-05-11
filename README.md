# 🧠 NN_Labs — Neural Network Labs

> A hands-on series of labs covering the core Python libraries and frameworks used in **Data Science**, **Machine Learning**, and **Deep Learning**.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 📋 Table of Contents

- [Lab Overview](#-lab-overview)
- [Lab 1 — Python Data Science Foundations](#-lab-1-python-data-science-foundations)
  - [NumPy](#-numpy)
  - [Pandas](#-pandas)
  - [Matplotlib](#-matplotlib)
  - [Data Science Workflow](#-typical-data-science-workflow)
  - [Lab 1 Key Concepts](#-key-concepts--lab-1)
  - [Lab 1 MCQs](#-lab-1-mcqs-50-questions)
- [Lab 2 — Introduction to TensorFlow 2.x](#-lab-2-introduction-to-tensorflow-2x)
  - [What is TensorFlow?](#-what-is-tensorflow-2x)
  - [Core Data Structures](#-core-data-structures)
  - [Basic Operations](#-basic-tensor-operations)
  - [Random Tensors](#-random-tensors)
  - [Reduction Operations](#-reduction-operations)
  - [Computation Graphs](#️-computation-graphs-tffunction)
  - [Automatic Differentiation](#-automatic-differentiation-tfgradienttape)
  - [Full Workflow](#-full-tensorflow-workflow)
  - [Lab 2 Key Concepts](#-key-concepts-summary--lab-2)
  - [Lab 2 MCQs](#-lab-2-mcqs-40-questions)
- [Combined Summary](#-combined-key-concepts-both-labs)
- [Author](#-author)
- [License](#-license)

---

## 📦 Lab Overview

| Lab | Topic | Libraries |
|-----|-------|-----------|
| **Lab 1** | Python Data Science Foundations | NumPy, Pandas, Matplotlib |
| **Lab 2** | Introduction to TensorFlow 2.x | TensorFlow 2.x |

---
---

# 📗 Lab 1: Python Data Science Foundations

> An introduction to the core Python libraries used in **Data Science**, **Machine Learning**, and **Deep Learning**.

## 📚 Libraries Covered

| Library | Purpose |
|---------|---------|
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

## 🧠 Key Concepts — Lab 1

| Concept | Meaning |
|---------|---------|
| `ndarray` | NumPy's n-dimensional array |
| `Series` | 1D labeled data structure (Pandas) |
| `DataFrame` | 2D table structure (Pandas) |
| Indexing | Accessing individual elements |
| Slicing | Accessing a range of elements |
| `GroupBy` | Grouping rows by a shared value |
| Histogram | Plot showing data distribution |
| Line Plot | Plot showing variable relationships |

---

## ❓ Lab 1 MCQs (50 Questions)

<details>
<summary><strong>1️⃣ What is the primary purpose of NumPy?</strong></summary>

A) Web development  
B) Numerical computing and arrays  
C) Game design  
D) Database management  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣ Which NumPy object represents an n-dimensional array?</strong></summary>

A) Series  
B) DataFrame  
C) ndarray  
D) Tensor  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣ What will be the output of the following code?</strong></summary>

```python
import numpy as np
arr = np.array([1, 2, 3])
print(arr)
```

A) (1,2,3)  
B) [1 2 3]  
C) {1,2,3}  
D) Error  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ Which function creates evenly spaced values within an interval?</strong></summary>

A) np.zeros()  
B) np.ones()  
C) np.arange()  
D) np.reshape()  

✅ **Answer: C**
</details>

<details>
<summary><strong>5️⃣ What is the output of np.arange(0,5,2)?</strong></summary>

A) [0 1 2 3 4]  
B) [0 2 4]  
C) [2 4 6]  
D) [0 2 4 6]  

✅ **Answer: B**
</details>

<details>
<summary><strong>6️⃣ Which function creates an array filled with zeros?</strong></summary>

A) np.empty()  
B) np.full()  
C) np.zeros()  
D) np.array()  

✅ **Answer: C**
</details>

<details>
<summary><strong>7️⃣ What is the shape of np.zeros((2,3))?</strong></summary>

A) 2 rows and 3 columns  
B) 3 rows and 2 columns  
C) 6 rows  
D) 1 row and 6 columns  

✅ **Answer: A**
</details>

<details>
<summary><strong>8️⃣ Which function creates an array of ones?</strong></summary>

A) np.ones()  
B) np.zeros()  
C) np.identity()  
D) np.random()  

✅ **Answer: A**
</details>

<details>
<summary><strong>9️⃣ What does np.random.randint() generate?</strong></summary>

A) Floating-point numbers  
B) Strings  
C) Random integers  
D) Boolean values  

✅ **Answer: C**
</details>

<details>
<summary><strong>🔟 Which function generates random values between 0 and 1?</strong></summary>

A) np.random.randint()  
B) np.random.rand()  
C) np.arange()  
D) np.linspace()  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣1️⃣ What does np.linspace() do?</strong></summary>

A) Creates random values  
B) Creates evenly spaced values  
C) Creates identity matrix  
D) Creates zeros  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ What is the output of np.linspace(0,10,3)?</strong></summary>

A) [0 10]  
B) [0 5 10]  
C) [1 5 9]  
D) [0 2 4]  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣3️⃣ What does reshape() do?</strong></summary>

A) Deletes data  
B) Changes array shape  
C) Sorts values  
D) Adds elements  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ Which shape is valid for reshaping an array of 6 elements?</strong></summary>

A) (2,3)  
B) (4,4)  
C) (5,2)  
D) (7,1)  

✅ **Answer: A**
</details>

<details>
<summary><strong>1️⃣5️⃣ What does arr.max() return?</strong></summary>

A) Minimum value  
B) Mean value  
C) Maximum value  
D) Index of maximum value  

✅ **Answer: C**
</details>

<details>
<summary><strong>1️⃣6️⃣ What does arr.mean() compute?</strong></summary>

A) Median  
B) Mode  
C) Average  
D) Variance  

✅ **Answer: C**
</details>

<details>
<summary><strong>1️⃣7️⃣ What does argmax() return?</strong></summary>

A) Largest value  
B) Index of largest value  
C) Smallest value  
D) Mean value  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ Which operation accesses specific elements of an array?</strong></summary>

A) Plotting  
B) Reshaping  
C) Indexing  
D) Pooling  

✅ **Answer: C**
</details>

<details>
<summary><strong>1️⃣9️⃣ What is slicing used for?</strong></summary>

A) Deleting arrays  
B) Accessing subarrays  
C) Random generation  
D) Sorting arrays  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ Which library is mainly used for data analysis?</strong></summary>

A) TensorFlow  
B) Pandas  
C) OpenCV  
D) Flask  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ What is a Pandas Series?</strong></summary>

A) 2D table  
B) Image object  
C) 1D labeled array  
D) Neural network  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣2️⃣ What is a DataFrame?</strong></summary>

A) One-dimensional array  
B) Two-dimensional labeled table  
C) Neural network layer  
D) Random generator  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ Which function reads CSV files?</strong></summary>

A) pd.load()  
B) pd.read_csv()  
C) pd.import()  
D) pd.open()  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ What does df.head() display?</strong></summary>

A) Last rows  
B) Random rows  
C) First rows  
D) Middle rows  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣5️⃣ What does df.tail() display?</strong></summary>

A) First rows  
B) Last rows  
C) Random rows  
D) Column names  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ Which method uses labels for indexing?</strong></summary>

A) iloc[]  
B) loc[]  
C) mean()  
D) tail()  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ Which method uses integer positions?</strong></summary>

A) loc[]  
B) iloc[]  
C) groupby()  
D) plot()  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣8️⃣ What does groupby() do?</strong></summary>

A) Deletes groups  
B) Groups rows based on values  
C) Creates plots  
D) Reshapes arrays  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ Which statistical function computes average?</strong></summary>

A) median()  
B) std()  
C) mean()  
D) var()  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣0️⃣ Which function calculates standard deviation?</strong></summary>

A) mean()  
B) std()  
C) var()  
D) median()  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ Which function calculates variance?</strong></summary>

A) std()  
B) mode()  
C) var()  
D) mean()  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣2️⃣ Which library is used for plotting graphs?</strong></summary>

A) NumPy  
B) Pandas  
C) Matplotlib  
D) SciPy  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣3️⃣ What does plt.hist() create?</strong></summary>

A) Line plot  
B) Pie chart  
C) Histogram  
D) Scatter plot  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣4️⃣ What is a histogram used for?</strong></summary>

A) Data distribution visualization  
B) Sorting arrays  
C) Neural network training  
D) CSV reading  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣5️⃣ Which function creates line plots?</strong></summary>

A) plt.hist()  
B) plt.plot()  
C) plt.array()  
D) plt.group()  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣6️⃣ What does plt.show() do?</strong></summary>

A) Saves file  
B) Displays plot  
C) Deletes plot  
D) Reads plot  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣7️⃣ Which library provides fast numerical operations?</strong></summary>

A) Pandas  
B) Matplotlib  
C) NumPy  
D) Flask  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣8️⃣ Which structure is similar to an Excel sheet?</strong></summary>

A) ndarray  
B) Series  
C) DataFrame  
D) Tuple  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣9️⃣ Which function creates random floating-point values?</strong></summary>

A) randint()  
B) rand()  
C) arange()  
D) zeros()  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ What is the purpose of Matplotlib?</strong></summary>

A) Database management  
B) Web development  
C) Data visualization  
D) Audio processing  

✅ **Answer: C**
</details>

<details>
<summary><strong>4️⃣1️⃣ Which operation changes array dimensions?</strong></summary>

A) reshape()  
B) mean()  
C) groupby()  
D) loc[]  

✅ **Answer: A**
</details>

<details>
<summary><strong>4️⃣2️⃣ What does df.iloc[:,1] select?</strong></summary>

A) First row  
B) Second column  
C) Last row  
D) Entire DataFrame  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣3️⃣ Which function returns the index of the minimum value?</strong></summary>

A) argmax()  
B) min()  
C) argmin()  
D) mean()  

✅ **Answer: C**
</details>

<details>
<summary><strong>4️⃣4️⃣ Which NumPy function creates arrays with evenly spaced values including decimals?</strong></summary>

A) arange()  
B) linspace()  
C) randint()  
D) zeros()  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣5️⃣ Which command creates a 2×2 matrix of ones?</strong></summary>

A) np.zeros((2,2))  
B) np.ones((2,2))  
C) np.arange((2,2))  
D) np.rand((2,2))  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣6️⃣ Which Pandas object is one-dimensional?</strong></summary>

A) DataFrame  
B) ndarray  
C) Series  
D) Matrix  

✅ **Answer: C**
</details>

<details>
<summary><strong>4️⃣7️⃣ What does filtering in Pandas mean?</strong></summary>

A) Removing libraries  
B) Selecting rows based on conditions  
C) Plotting graphs  
D) Reshaping arrays  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣8️⃣ Which command accesses the first element of a NumPy array arr?</strong></summary>

A) arr(0)  
B) arr[0]  
C) arr{0}  
D) arr.first()  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣9️⃣ Which library is commonly used together with NumPy and Pandas for plotting?</strong></summary>

A) OpenCV  
B) Flask  
C) Matplotlib  
D) Django  

✅ **Answer: C**
</details>

<details>
<summary><strong>5️⃣0️⃣ What is the main purpose of this lab?</strong></summary>

A) Build websites  
B) Learn Python data science libraries  
C) Develop mobile apps  
D) Create databases  

✅ **Answer: B**
</details>

---
---

# 📘 Lab 2: Introduction to TensorFlow 2.x

> A hands-on introduction to the core building blocks of TensorFlow 2.x — tensors, variables, math ops, computation graphs, and automatic differentiation.

## 📋 Overview

| Topic | Description |
|-------|-------------|
| 🔢 **Tensors** | Core data representation |
| ⚙️ **Variables** | Trainable model parameters |
| 🧮 **Math Ops** | Arithmetic & matrix operations |
| 📊 **Graphs** | Optimized computation via `@tf.function` |
| 🔁 **Autodiff** | Gradient computation via `tf.GradientTape` |

---

## 🤖 What is TensorFlow 2.x?

**TensorFlow** is an open-source library developed by Google for machine learning, deep learning, and numerical computation.

### ✨ Eager Execution (Default in TF 2.x)

TensorFlow 2.x runs in **eager mode** by default — no more building static graphs before running them.

```
✔ Operations execute immediately
✔ Easier debugging
✔ More Python-like behavior
```

---

## 📦 Core Data Structures

### 🔵 Tensor (`tf.Tensor`)

The **primary data structure** in TensorFlow — an n-dimensional, immutable array.

```python
import tensorflow as tf

tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print(tensor)
```

**Output:**
```
tf.Tensor(
[[1. 2.]
 [3. 4.]], shape=(2, 2), dtype=float32)
```

---

### 🟢 Variable (`tf.Variable`)

Used for **trainable parameters** like weights and biases.

```python
var = tf.Variable([[1.0, 2.0],
                   [3.0, 4.0]])

# Updating a Variable
var.assign([[5.0, 6.0],
            [7.0, 8.0]])

print(var.numpy())
# [[5. 6.]
#  [7. 8.]]
```

---

### 🟣 RaggedTensor

Used when rows have **different lengths** (e.g., variable-length NLP sequences).

```
[[1, 2, 3],
 [4, 5],
 [6]]
```

---

### 🟡 SparseTensor

Used when **most values are zero** — memory efficient for large sparse datasets.

---

### Data Structure Comparison

| Type | Mutable | Use Case |
|------|---------|----------|
| `tf.Tensor` | ❌ | General data representation |
| `tf.Variable` | ✅ | Trainable weights & biases |
| `RaggedTensor` | ❌ | Variable-length sequences |
| `SparseTensor` | ❌ | Mostly-zero data |

---

## 🧮 Basic Tensor Operations

### Arithmetic Operations

```python
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

print(tf.add(x, y))       # [5 7 9]
print(tf.multiply(x, y))  # [4 10 18]
```

### Matrix Multiplication

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print(tf.linalg.matmul(a, b))
# [[19 22]
#  [43 50]]
```

---

## 🎲 Random Tensors

```python
rand = tf.random.normal((2, 2))
print(rand)
```

| Use Case | Function |
|----------|----------|
| Normal distribution | `tf.random.normal()` |
| Uniform distribution | `tf.random.uniform()` |
| Reproducible results | `tf.random.set_seed()` |

---

## 📊 Reduction Operations

```python
t = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])

print(tf.reduce_mean(t))           # 2.5       — global mean
print(tf.reduce_mean(t, axis=0))   # [2. 3.]   — column-wise
print(tf.reduce_mean(t, axis=1))   # [1.5 3.5] — row-wise
```

---

## ⚙️ Computation Graphs (`@tf.function`)

The `@tf.function` decorator converts Python functions into **optimized TensorFlow graphs**.

```python
@tf.function
def add(a, b):
    return a + b

print(add(1, 2))  # tf.Tensor(3, shape=(), dtype=int32)
```

| Benefit | Description |
|---------|-------------|
| ⚡ Faster execution | Compiled graph runs faster than eager mode |
| 📦 Exportable | Can be saved and deployed to production |
| 🔄 Auto-optimization | XLA compilation and graph pruning |

---

## 🔁 Automatic Differentiation (`tf.GradientTape`)

TensorFlow records operations inside a `GradientTape` context to **compute gradients automatically** — the backbone of backpropagation.

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x   # y = x²

dy_dx = tape.gradient(y, x)
print(dy_dx)  # 6.0
```

**Math Explanation:**

```
y = x²        →     dy/dx = 2x

At x = 3:     →     2 × 3 = 6 ✔
```

---

## 🌍 Full TensorFlow Workflow

```
📥 Input Data
     ↓
🔢 Tensors           (data representation)
     ↓
⚙️  Variables         (model parameters: weights & biases)
     ↓
➡️  Forward Pass       (predictions)
     ↓
📉 Loss Function      (measure error)
     ↓
🔁 GradientTape       (compute gradients)
     ↓
🔄 Weight Update      (optimizer step)
     ↓
🔁 Repeat until convergence
```

---

## 🔥 Key Concepts Summary — Lab 2

### ⚙️ Core Tools

| Tool | Purpose |
|------|---------|
| `tf.constant` | Create fixed tensors |
| `tf.Variable` | Create trainable parameters |
| `tf.GradientTape` | Compute gradients (backprop) |
| `@tf.function` | Optimize & compile functions |

### 🧮 Mathematical Operations

| Symbol | Function |
|--------|----------|
| ➕ | `tf.add` |
| ➖ | `tf.subtract` |
| ✖️ | `tf.multiply` |
| ➗ | `tf.divide` |
| 🔢 | `tf.linalg.matmul` |

---

## ❓ Lab 2 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ What is TensorFlow mainly used for?</strong></summary>

A) Web development  
B) Deep learning and numerical computation  
C) Game development  
D) Operating systems  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣ What is a TensorFlow tensor?</strong></summary>

A) Mutable array  
B) Database table  
C) N-dimensional immutable array  
D) String object  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣ Which execution mode does TensorFlow 2.x use by default?</strong></summary>

A) Graph mode only  
B) Eager execution  
C) Batch mode  
D) Static mode  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ Which object is mutable in TensorFlow?</strong></summary>

A) tf.Tensor  
B) tf.Variable  
C) tf.constant  
D) tf.function  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ What does tf.constant create?</strong></summary>

A) Variable tensor  
B) Immutable tensor  
C) Random tensor  
D) Sparse matrix  

✅ **Answer: B**
</details>

<details>
<summary><strong>6️⃣ What is tf.Variable used for?</strong></summary>

A) Plotting graphs  
B) Storing trainable parameters  
C) Data cleaning  
D) File reading  

✅ **Answer: B**
</details>

<details>
<summary><strong>7️⃣ What does assign() do?</strong></summary>

A) Deletes variable  
B) Updates variable value  
C) Converts tensor  
D) Creates graph  

✅ **Answer: B**
</details>

<details>
<summary><strong>8️⃣ What does tf.add() do?</strong></summary>

A) Matrix inversion  
B) Element-wise addition  
C) Sorting  
D) Normalization  

✅ **Answer: B**
</details>

<details>
<summary><strong>9️⃣ What does tf.multiply() do?</strong></summary>

A) Element-wise multiplication  
B) Matrix transpose  
C) Loss calculation  
D) Gradient computation  

✅ **Answer: A**
</details>

<details>
<summary><strong>🔟 What does tf.linalg.matmul() perform?</strong></summary>

A) Scalar multiplication  
B) Matrix multiplication  
C) Random sampling  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣1️⃣ tf.random.normal generates values from which distribution?</strong></summary>

A) Uniform distribution  
B) Normal distribution  
C) Bernoulli distribution  
D) Poisson distribution  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ What does tf.reduce_mean calculate?</strong></summary>

A) Maximum value  
B) Minimum value  
C) Mean value  
D) Median  

✅ **Answer: C**
</details>

<details>
<summary><strong>1️⃣3️⃣ What is tf.Tensor?</strong></summary>

A) Mutable variable  
B) Immutable data structure  
C) File format  
D) Model optimizer  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ What is tf.RaggedTensor used for?</strong></summary>

A) Fixed-size arrays  
B) Variable-length sequences  
C) Images only  
D) Scalars only  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ What is tf.SparseTensor used for?</strong></summary>

A) Dense matrices  
B) Mostly zero data  
C) Text only  
D) Images  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣6️⃣ What does @tf.function do?</strong></summary>

A) Slows execution  
B) Converts function into graph  
C) Deletes function  
D) Prints tensor  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ Why use @tf.function?</strong></summary>

A) Debugging  
B) Performance optimization  
C) Data cleaning  
D) Visualization  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ What does tf.GradientTape do?</strong></summary>

A) Saves model  
B) Computes gradients  
C) Loads data  
D) Plots graphs  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ Gradients are used for?</strong></summary>

A) Sorting data  
B) Optimization  
C) Printing values  
D) File handling  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ What is backpropagation?</strong></summary>

A) Forward prediction  
B) Gradient computation process  
C) Data loading  
D) Plotting graphs  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ What shape does tf.constant support?</strong></summary>

A) Only 1D  
B) Only 2D  
C) Multi-dimensional  
D) Only scalars  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣2️⃣ What is eager execution?</strong></summary>

A) Delayed execution  
B) Immediate execution  
C) Parallel execution  
D) Random execution  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ What happens inside @tf.function?</strong></summary>

A) Python runs slowly  
B) Code is optimized into a graph  
C) Memory is deleted  
D) Data is lost  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ What is tf.Variable best for?</strong></summary>

A) Static data  
B) Trainable weights  
C) Images  
D) Strings  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣5️⃣ What does matmul require?</strong></summary>

A) Scalars  
B) Shape-compatible tensors  
C) Strings  
D) Lists only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ Random tensors are used for?</strong></summary>

A) Debugging  
B) Weight initialization  
C) Printing  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ tf.reduce_mean reduces which dimension?</strong></summary>

A) Rows only  
B) Columns only  
C) Any specified dimension  
D) Variables  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣8️⃣ TensorFlow supports which hardware?</strong></summary>

A) Only CPU  
B) CPU and GPU  
C) Only GPU  
D) Cloud only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ GradientTape is used in?</strong></summary>

A) Data loading  
B) Training models  
C) Visualization  
D) File storage  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣0️⃣ Which TensorFlow object is immutable?</strong></summary>

A) tf.Variable  
B) tf.Tensor  
C) Python list  
D) NumPy array  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ What does assign_add() do?</strong></summary>

A) Multiply  
B) Add a value to the variable  
C) Delete variable  
D) Normalize  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣2️⃣ TensorFlow computation is based on?</strong></summary>

A) Trees  
B) Graphs  
C) Lists  
D) Files  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣3️⃣ What improves performance in TensorFlow?</strong></summary>

A) tf.constant  
B) tf.function  
C) tf.print  
D) tf.list  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣4️⃣ Which is NOT a TensorFlow data structure?</strong></summary>

A) Tensor  
B) Variable  
C) DataFrame  
D) SparseTensor  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣5️⃣ What does tf.constant store?</strong></summary>

A) Trainable weights  
B) Fixed values  
C) Graphs  
D) Models  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣6️⃣ Matrix multiplication output depends on?</strong></summary>

A) Shape compatibility  
B) Random seed  
C) Loss function  
D) Dataset size  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣7️⃣ tf.random.normal is useful for?</strong></summary>

A) Loss calculation  
B) Weight initialization  
C) File reading  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣8️⃣ Gradient descent uses?</strong></summary>

A) Random search  
B) Gradients  
C) Sorting  
D) Clustering  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣9️⃣ What is the main goal of TensorFlow training?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Delete model  
D) Freeze weights  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ Which concept links GradientTape to model training?</strong></summary>

A) Data loading  
B) Backpropagation  
C) Visualization  
D) Reshaping  

✅ **Answer: B**
</details>

---
---

## 🧠 Combined Key Concepts (Both Labs)

| Concept | Library | Meaning |
|---------|---------|---------|
| `ndarray` | NumPy | N-dimensional array |
| `Series` | Pandas | 1D labeled data structure |
| `DataFrame` | Pandas | 2D table structure |
| `tf.Tensor` | TensorFlow | Immutable n-dimensional array |
| `tf.Variable` | TensorFlow | Trainable model parameter |
| `GradientTape` | TensorFlow | Automatic differentiation |
| `@tf.function` | TensorFlow | Graph-mode optimization |
| `GroupBy` | Pandas | Grouping rows by shared value |
| Histogram | Matplotlib | Data distribution plot |

---

## 🌍 Why These Labs Matter

These labs build the **complete foundation** for Deep Learning:

| Lab | Builds Toward |
|-----|---------------|
| NumPy | Tensor math & array ops |
| Pandas | Data preprocessing pipelines |
| Matplotlib | Training curve visualization |
| TensorFlow Tensors | Model architecture |
| GradientTape | Backpropagation & training |
| `@tf.function` | Production deployment |

---

## 👤 Author

**Mohamed Shaaban** — [GitHub Profile](https://github.com/Mohmed-Shaaban)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for Deep Learning enthusiasts
</p>
