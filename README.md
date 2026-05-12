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
- [Lab 3 — Linear Regression with TensorFlow](#-lab-3-linear-regression-with-tensorflow)
  - [Lab Summary](#-lab-summary)
  - [Data Generation](#-1-data-generation)
  - [Data Preparation](#-2-data-preparation)
  - [Model Building](#-3-model-building)
  - [Loss Function](#-4-loss-function)
  - [Training Loop](#️-5-training-loop-core-idea)
  - [Key TensorFlow Functions](#-6-key-tensorflow-functions)
  - [Evaluation](#-7-evaluation)
  - [Full Training Pipeline](#-8-full-training-pipeline)
  - [Key Takeaways](#-9-key-takeaways)
  - [Lab 3 MCQs](#-lab-3-mcqs-40-questions)
- [Lab 4 — Logic Gates & Neural Networks](#-lab-4-logic-gates--neural-networks)
  - [What is a Neuron?](#-1-what-is-a-neuron)
  - [Sigmoid Activation](#-2-sigmoid-activation)
  - [Logic Gates](#-3-logic-gates)
  - [The XOR Problem](#-4-the-xor-problem)
  - [Training a Neuron](#-5-training-a-neuron)
  - [Lab 4 MCQs](#-lab-4-mcqs-40-questions)
- [Lab 5 — MNIST Neural Network](#-lab-5-mnist-neural-network)
  - [MNIST Dataset](#-1-mnist-dataset)
  - [Model Architecture](#-2-model-architecture-mlp)
  - [Training Pipeline](#-3-training-pipeline)
  - [Key Functions](#-4-key-functions)
  - [Lab 5 MCQs](#-lab-5-mcqs-40-questions)
- [Lab 6 — Keras Sequential Model](#-lab-6-keras-sequential-model)
  - [Sequential API](#-1-sequential-api)
  - [Layers & Activations](#-2-layers--activations)
  - [Model Lifecycle](#-3-model-lifecycle)
  - [Lab 6 MCQs](#-lab-6-mcqs-40-questions)
- [Lab 7 — CNN for CIFAR-10](#-lab-7-cnn-for-cifar-10)
  - [CIFAR-10 Dataset](#-1-cifar-10-dataset)
  - [CNN Building Blocks](#-2-cnn-building-blocks)
  - [Model Structure](#-3-cnn-model-structure)
  - [Data Augmentation](#-4-data-augmentation)
  - [Training Process](#-5-model-training-process)
  - [Lab 7 MCQs](#-lab-7-mcqs-40-questions)
- [Combined Summary](#-combined-key-concepts-all-labs)
- [Author](#-author)
- [License](#-license)

---

## 📦 Lab Overview

| Lab | Topic | Libraries |
|-----|-------|-----------|
| **Lab 1** | Python Data Science Foundations | NumPy, Pandas, Matplotlib |
| **Lab 2** | Introduction to TensorFlow 2.x | TensorFlow 2.x |
| **Lab 3** | Linear Regression with TensorFlow | TensorFlow Core API, Gradient Descent |
| **Lab 4** | Logic Gates & Neural Networks | Neuron, Sigmoid, XOR, GradientTape |
| **Lab 5** | MNIST Neural Network (Custom Loop) | MLP, GradientTape, tf.data |
| **Lab 6** | Keras Sequential Model (MNIST) | Keras, Dense, Softmax, Adam |
| **Lab 7** | CNN for CIFAR-10 Classification | Conv2D, MaxPooling, Augmentation |

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

---
---

# 🟠 Lab 3: Linear Regression with TensorFlow

> Build and train a **Multiple Linear Regression** model from scratch using TensorFlow's Core API, a custom training loop, and gradient descent.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Regression](https://img.shields.io/badge/Task-Regression-blue?style=for-the-badge)
![Custom Loop](https://img.shields.io/badge/Training-Custom%20Loop-orange?style=for-the-badge)

---

## 🎯 Lab Summary

This lab demonstrates how to build and train a Multiple Linear Regression model using:

| Component | Tool |
|-----------|------|
| ⚙️ Model API | `tf.Module` |
| 🔁 Optimizer | Gradient Descent (manual) |
| 📊 Training | Custom training loop |
| 📉 Loss | Mean Squared Error (MSE) |

---

## 🧠 Main Idea

We want to learn a function that maps input features → continuous output:

```
y = w₁x₁ + w₂x₂ + ⋯ + b
```

The model learns **weights `w`** and **bias `b`** by minimizing prediction error.

---

## 📦 1. Data Generation

### 📊 Synthetic Dataset

We create artificial data using a known equation:

```
y = w·x + b + noise
```

**Why synthetic data?**

```
✔ We know the true pattern
✔ Helps verify if the model learns correctly
✔ No need for external datasets
```

---

## 🔀 2. Data Preparation

### Steps

**🔹 Shuffle data** — Randomly mixes samples to avoid learning order bias.

**🔹 Split data:**

| Set | Purpose |
|-----|---------|
| 🟢 Training set | Model learns from this |
| 🔵 Testing set | Model is evaluated on this |

### TensorFlow Dataset Pipeline

```python
# Create dataset from arrays
tf.data.Dataset.from_tensor_slices((x, y))
```

```python
# Shuffle — randomizes data order, prevents overfitting patterns
dataset.shuffle(buffer_size=100)
```

```python
# Batch — splits into mini-batches for faster training
dataset.batch(32)
```

---

## 🧠 3. Model Building

We define a custom model using `tf.Module`.

### Linear Regression Equation

```
y = W·x + b
```

| Symbol | Meaning |
|--------|---------|
| `W` | Weights (learned from data) |
| `b` | Bias (learned from data) |
| `x` | Input features |
| `y` | Predicted output |

### Forward Pass

```python
y_pred = W * x + b
```

---

## 📉 4. Loss Function

### 🔵 Mean Squared Error (MSE)

```
MSE = (1/n) · Σ (y - ŷ)²
```

| Property | Description |
|----------|-------------|
| Measures | Prediction error |
| Lower = | Better model |
| Used in | Regression tasks |

---

## ⚙️ 5. Training Loop (Core Idea)

Instead of built-in `model.fit()`, we use a **custom training loop** for full control.

### Steps per Epoch

```
1️⃣  Forward Pass     →  Model makes predictions
2️⃣  Compute Loss     →  Calculate MSE
3️⃣  Compute Gradients →  Using tf.GradientTape()
4️⃣  Update Weights   →  variable.assign_sub(lr * gradient)
```

### Gradient Descent Update Rule

```
w = w - α · (∂L/∂w)
b = b - α · (∂L/∂b)
```

Where `α` is the **learning rate**.

### Code Pattern

```python
with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = mse(y, y_pred)

gradients = tape.gradient(loss, model.trainable_variables)

for var, grad in zip(model.trainable_variables, gradients):
    var.assign_sub(learning_rate * grad)
```

---

## 🔁 6. Key TensorFlow Functions

### `tf.stack()`

Combines multiple tensors into one along a new dimension.

```python
tf.stack([a, b], axis=0)   # ✔ Adds new dimension
```

---

### `tf.squeeze()`

Removes dimensions of size 1.

```python
tf.squeeze(tensor)         # ✔ Removes extra dimensions
```

---

### `tf.data.Dataset`

Pipeline system for efficient data handling:

| Method | Purpose |
|--------|---------|
| `from_tensor_slices()` | Convert arrays to dataset |
| `shuffle()` | Randomize order |
| `batch()` | Group into mini-batches |

---

### `assign_sub()`

Performs in-place subtraction on a `tf.Variable` — used for weight updates.

```python
W.assign_sub(learning_rate * grad)
# Equivalent to: W = W - learning_rate * grad
```

---

## 📊 7. Evaluation

After training, we evaluate the model on both sets:

| Metric | Meaning |
|--------|---------|
| 📉 Training loss | Error on data the model trained on |
| 📊 Testing loss | Error on unseen data (generalization) |

### Interpretation

```
✔ Training loss ↓   →  Model is learning
✔ Test loss ↓       →  Good generalization
❌ Gap increases    →  Overfitting risk
```

We visualize **loss vs. epochs** to track model learning progress.

---

## 🔥 8. Full Training Pipeline

```
Input Data
     ↓
Shuffle → Batch
     ↓
Model (W, b)
     ↓
Forward Pass  →  y_pred = W·x + b
     ↓
Loss (MSE)    →  (1/n)·Σ(y - ŷ)²
     ↓
tf.GradientTape  →  compute ∂L/∂W, ∂L/∂b
     ↓
Weight Update  →  W -= lr · grad
     ↓
Repeat (epochs)
     ↓
Evaluate on Test Set
```

---

## 🎯 9. Key Takeaways

```
✔ Linear regression  =  Weighted sum model
✔ Gradient descent   =  Parameter optimization
✔ tf.data            =  Efficient data pipeline
✔ tf.GradientTape    =  Automatic differentiation
✔ assign_sub()       =  Manual weight update
✔ MSE                =  Regression loss function
```

---

## 🚀 Why This Lab is Important

This lab teaches the **foundation of all neural network training**:

| Concept | Leads To |
|---------|----------|
| Custom training loop | Neural network training logic |
| GradientTape | Backpropagation |
| MSE loss | Regression objective functions |
| tf.data pipeline | Scalable data loading |
| Weight updates | Any deep learning optimizer |

---

## ❓ Lab 3 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ What is the main goal of linear regression?</strong></summary>

A) Classification  
B) Clustering  
C) Predict continuous values  
D) Image detection  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣ Linear regression models the relationship between?</strong></summary>

A) Images and labels  
B) Inputs and continuous outputs  
C) Text and speech  
D) Pixels only  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣ What is the general equation of linear regression?</strong></summary>

A) y = x²  
B) y = wx + b  
C) y = log(x)  
D) y = sin(x)  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ In y = wx + b, what is w?</strong></summary>

A) Bias  
B) Weight  
C) Loss  
D) Gradient  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ In y = wx + b, what is b?</strong></summary>

A) Bias  
B) Batch  
C) Gradient  
D) Feature  

✅ **Answer: A**
</details>

<details>
<summary><strong>6️⃣ What is used to measure error in this lab?</strong></summary>

A) Accuracy  
B) Cross entropy  
C) Mean Squared Error  
D) Precision  

✅ **Answer: C**
</details>

<details>
<summary><strong>7️⃣ MSE stands for?</strong></summary>

A) Mean Standard Error  
B) Mean Squared Error  
C) Model Score Estimation  
D) Mean Sample Error  

✅ **Answer: B**
</details>

<details>
<summary><strong>8️⃣ What does MSE measure?</strong></summary>

A) Model speed  
B) Difference between predictions and true values  
C) Data size  
D) Learning rate  

✅ **Answer: B**
</details>

<details>
<summary><strong>9️⃣ What is gradient descent used for?</strong></summary>

A) Data visualization  
B) Minimizing loss  
C) Increasing error  
D) Sorting data  

✅ **Answer: B**
</details>

<details>
<summary><strong>🔟 Gradient descent updates?</strong></summary>

A) Data  
B) Model parameters  
C) Labels  
D) Files  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣1️⃣ tf.data.Dataset is used for?</strong></summary>

A) Plotting  
B) Data pipeline creation  
C) Model saving  
D) Loss calculation  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ What does dataset.shuffle() do?</strong></summary>

A) Sorts data  
B) Randomizes order  
C) Deletes data  
D) Normalizes data  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣3️⃣ What does dataset.batch() do?</strong></summary>

A) Merges models  
B) Splits data into batches  
C) Removes samples  
D) Sorts features  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ Why do we use batching?</strong></summary>

A) Slower training  
B) Faster training and memory efficiency  
C) Increase loss  
D) Random output  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ tf.GradientTape is used for?</strong></summary>

A) Visualization  
B) Automatic differentiation  
C) Data cleaning  
D) Plotting graphs  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣6️⃣ What does GradientTape compute?</strong></summary>

A) Loss  
B) Gradients  
C) Accuracy  
D) Predictions  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ tf.Module is used to?</strong></summary>

A) Store datasets  
B) Build custom models  
C) Plot graphs  
D) Load CSV files  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ In linear regression, the prediction is?</strong></summary>

A) Sum of random values  
B) Weighted sum of inputs  
C) Image classification  
D) Clustering result  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ What does assign_sub() do?</strong></summary>

A) Adds value  
B) Subtracts value  
C) Multiplies value  
D) Resets variable  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ assign_sub() is used for?</strong></summary>

A) Forward pass  
B) Updating weights  
C) Data loading  
D) Visualization  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ Training loss represents?</strong></summary>

A) Model accuracy  
B) Error on training data  
C) Test speed  
D) Dataset size  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣2️⃣ Test loss measures?</strong></summary>

A) Performance on unseen data  
B) Training speed  
C) Memory usage  
D) Gradient size  

✅ **Answer: A**
</details>

<details>
<summary><strong>2️⃣3️⃣ One training iteration includes?</strong></summary>

A) Only prediction  
B) Forward + loss + backward + update  
C) Only plotting  
D) Only loading data  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ Gradient descent tries to?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Randomize weights  
D) Delete model  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣5️⃣ Synthetic data means?</strong></summary>

A) Real-world data  
B) Generated artificial data  
C) Missing data  
D) Image data  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ tf.stack() is used to?</strong></summary>

A) Remove tensors  
B) Combine tensors  
C) Sort tensors  
D) Normalize tensors  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ tf.squeeze() removes?</strong></summary>

A) Large values  
B) Dimensions of size 1  
C) Data samples  
D) Loss values  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣8️⃣ Learning rate controls?</strong></summary>

A) Dataset size  
B) Step size in gradient descent  
C) Number of features  
D) Model architecture  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ If learning rate is too high?</strong></summary>

A) Fast convergence always  
B) Model may diverge  
C) No change  
D) Perfect training  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣0️⃣ If learning rate is too low?</strong></summary>

A) Very fast training  
B) Slow convergence  
C) No training  
D) Random loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ What is the purpose of the forward pass?</strong></summary>

A) Update weights  
B) Make predictions  
C) Compute gradients  
D) Shuffle data  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣2️⃣ Backpropagation is used to?</strong></summary>

A) Compute gradients  
B) Load data  
C) Plot graphs  
D) Normalize inputs  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣3️⃣ Linear regression is a?</strong></summary>

A) Classification model  
B) Regression model  
C) Clustering model  
D) Reinforcement model  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣4️⃣ Which TensorFlow feature improves performance?</strong></summary>

A) tf.constant  
B) tf.function  
C) tf.print  
D) tf.list  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣5️⃣ Dataset.from_tensor_slices is used to?</strong></summary>

A) Create dataset from tensors  
B) Plot graphs  
C) Train model directly  
D) Save model  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣6️⃣ The goal of training is to?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Randomize outputs  
D) Freeze model  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣7️⃣ Which function calculates gradients automatically?</strong></summary>

A) tf.constant  
B) tf.GradientTape  
C) tf.stack  
D) tf.data  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣8️⃣ Model parameters in this lab are?</strong></summary>

A) Images  
B) Weights and bias  
C) Labels only  
D) Dataset  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣9️⃣ What indicates good training?</strong></summary>

A) Increasing loss  
B) Decreasing loss  
C) Random loss  
D) Constant loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ The main output of a linear regression model is?</strong></summary>

A) Class label  
B) Continuous value  
C) Image  
D) Text  

✅ **Answer: B**
</details>

---
---

---
---

# 🟡 Lab 4: Logic Gates & Neural Networks

> Learn how a single **neuron** (perceptron) works, implement logic gates (AND, OR, XOR), discover why XOR needs multiple layers, and train neurons using **TensorFlow GradientTape**.

![Neuron](https://img.shields.io/badge/Model-Perceptron-yellow?style=for-the-badge)
![Activation](https://img.shields.io/badge/Activation-Sigmoid-blue?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 🧠 Lab Summary

| Concept | Description |
|---------|-------------|
| 🧠 Neuron | Basic unit of a neural network |
| ⚡ Sigmoid | Activation function → output in (0, 1) |
| 🔌 Logic Gates | AND, OR, XOR patterns |
| ❌ XOR Problem | Shows limits of a single neuron |
| 🔁 GradientTape | Used for training the neuron |
| 📉 MSE Loss | Measures prediction error |

---

## 🧠 1. What is a Neuron?

A **neuron** is the basic unit of a neural network, inspired by the human brain.

```
Input (x)  →  Weighted Sum (wx + b)  →  Activation  →  Output (ŷ)
```

| Component | Role |
|-----------|------|
| `x` | Input feature |
| `w` | Weight (importance) |
| `b` | Bias (shift) |
| Activation | Adds non-linearity |

**Neuron equation:**
```
z = w·x + b
ŷ = sigmoid(z)
```

---

## ⚡ 2. Sigmoid Activation

Sigmoid squashes any value into the range **(0, 1)**, making it ideal for **probability output**.

```
sigmoid(x) = 1 / (1 + e^(-x))
```

| Input | Output |
|-------|--------|
| Very negative | ≈ 0 |
| 0 | 0.5 |
| Very positive | ≈ 1 |

```python
import tensorflow as tf
output = tf.sigmoid(z)
```

---

## 🔌 3. Logic Gates

Logic gates are simple binary functions — perfect for testing what a single neuron can learn.

| Gate | Input (0,0) | (0,1) | (1,0) | (1,1) |
|------|:-----------:|:-----:|:-----:|:-----:|
| AND  | 0 | 0 | 0 | 1 |
| OR   | 0 | 1 | 1 | 1 |
| XOR  | 0 | 1 | 1 | 0 |

**AND and OR** are **linearly separable** → a single neuron can learn them.

---

## ❌ 4. The XOR Problem

**XOR is NOT linearly separable** — you cannot draw a single straight line to separate the outputs.

```
(0,0) → 0  ●          (0,1) → 1  ○
(1,0) → 1  ○          (1,1) → 0  ●

→ No single line can separate ● from ○
```

**Solution:** Use **multiple layers** (hidden layers) to learn non-linear boundaries.

| Model | Can learn AND/OR? | Can learn XOR? |
|-------|:-----------------:|:--------------:|
| Single Neuron | ✅ | ❌ |
| Multi-layer Network | ✅ | ✅ |

---

## 🔁 5. Training a Neuron

### Loss Function — MSE
```
MSE = (1/n) · Σ (y - ŷ)²
```

### Training Loop

```python
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = tf.sigmoid(x @ W + b)
        loss = tf.reduce_mean((y - y_pred) ** 2)

    grads = tape.gradient(loss, [W, b])
    W.assign_sub(lr * grads[0])
    b.assign_sub(lr * grads[1])
```

### Key TensorFlow Functions

| Function | Purpose |
|----------|---------|
| `tf.sigmoid()` | Apply sigmoid activation |
| `tf.GradientTape()` | Compute gradients |
| `tf.equal()` | Compare predictions to labels |
| `tf.round()` | Round probabilities to 0/1 |
| `tf.cast()` | Change tensor data type |

---

## 🎯 Key Takeaways — Lab 4

```
✔ Neuron = weighted sum + activation
✔ Sigmoid = probability output (0 to 1)
✔ AND/OR = linearly separable → single neuron works
✔ XOR = NOT linearly separable → needs multiple layers
✔ GradientTape = trains the neuron
✔ MSE = loss for binary tasks here
```

---

## ❓ Lab 4 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ What is the basic unit of a neural network?</strong></summary>

A) Dataset  
B) Neuron  
C) Matrix  
D) Loss function  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣ A neuron computes?</strong></summary>

A) y = x²  
B) y = wx + b  
C) y = log(x)  
D) y = sin(x)  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣ What activation function is commonly used in this lab?</strong></summary>

A) ReLU  
B) Sigmoid  
C) Tanh  
D) Softmax  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ Sigmoid output range is?</strong></summary>

A) (-∞, ∞)  
B) (0, 1)  
C) (-1, 1)  
D) (0, 10)  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ Sigmoid formula is?</strong></summary>

A) 1 / (1 + eˣ)  
B) 1 / (1 + e⁻ˣ)  
C) eˣ  
D) x²  

✅ **Answer: B**
</details>

<details>
<summary><strong>6️⃣ What is the main use of sigmoid?</strong></summary>

A) Sorting  
B) Probability output  
C) Image resizing  
D) File handling  

✅ **Answer: B**
</details>

<details>
<summary><strong>7️⃣ Which logic gate can a single neuron learn easily?</strong></summary>

A) XOR  
B) AND  
C) Complex circuits only  
D) Memory gates  

✅ **Answer: B**
</details>

<details>
<summary><strong>8️⃣ Which gate is NOT linearly separable?</strong></summary>

A) AND  
B) OR  
C) XOR  
D) NAND  

✅ **Answer: C**
</details>

<details>
<summary><strong>9️⃣ Why can't a single neuron learn XOR?</strong></summary>

A) Too slow  
B) Not enough data  
C) Not linearly separable  
D) Too many parameters  

✅ **Answer: C**
</details>

<details>
<summary><strong>🔟 XOR requires?</strong></summary>

A) One neuron  
B) Multiple layers  
C) No training  
D) No activation  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣1️⃣ A neural network is made of?</strong></summary>

A) Files  
B) Neurons  
C) Tables  
D) Images  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ What does bias do?</strong></summary>

A) Controls output shift  
B) Removes data  
C) Sorts inputs  
D) Normalizes images  

✅ **Answer: A**
</details>

<details>
<summary><strong>1️⃣3️⃣ Which loss function is used?</strong></summary>

A) Cross entropy  
B) MSE  
C) Hinge loss  
D) KL divergence  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ MSE stands for?</strong></summary>

A) Mean Signal Error  
B) Mean Squared Error  
C) Model Score Estimation  
D) Matrix Sum Error  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ MSE measures?</strong></summary>

A) Accuracy  
B) Difference between prediction and target  
C) Data size  
D) Gradient value  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣6️⃣ What is used to compute gradients?</strong></summary>

A) tf.constant  
B) tf.GradientTape  
C) tf.stack  
D) tf.sigmoid  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ Gradient descent is used to?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Shuffle data  
D) Store variables  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ Weight update rule is?</strong></summary>

A) w = w + gradient  
B) w = w - learning_rate × gradient  
C) w = random  
D) w = 0  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ A neuron output after sigmoid is?</strong></summary>

A) Always 0  
B) Always 1  
C) Between 0 and 1  
D) Between -1 and 1  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣0️⃣ AND gate outputs 1 when?</strong></summary>

A) Any input is 1  
B) Both inputs are 1  
C) Inputs are 0  
D) Inputs differ  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ OR gate outputs 1 when?</strong></summary>

A) Both inputs are 0  
B) At least one input is 1  
C) Both inputs are 1  
D) Inputs are negative  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣2️⃣ XOR outputs 1 when?</strong></summary>

A) Inputs are equal  
B) Inputs differ  
C) Inputs are both 0  
D) Inputs are both 1  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ A single neuron is also called?</strong></summary>

A) Deep model  
B) Perceptron  
C) CNN  
D) RNN  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ Activation function introduces?</strong></summary>

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣5️⃣ Without activation function, model becomes?</strong></summary>

A) Non-linear  
B) Linear  
C) Random  
D) Deep  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ Multi-layer networks are needed for?</strong></summary>

A) Simple tasks  
B) Non-linear problems  
C) Sorting only  
D) File reading  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ XOR problem shows?</strong></summary>

A) Overfitting  
B) Linear limitation of single neurons  
C) Data imbalance  
D) Memory issue  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣8️⃣ tf.equal() is used for?</strong></summary>

A) Addition  
B) Comparison  
C) Loss calculation  
D) Training  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ tf.round() does?</strong></summary>

A) Floors values  
B) Rounds values  
C) Normalizes data  
D) Sorts tensors  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣0️⃣ tf.cast() is used to?</strong></summary>

A) Train model  
B) Change data type  
C) Plot graphs  
D) Compute loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ A neuron is inspired by?</strong></summary>

A) CPU  
B) Human brain  
C) Database  
D) Compiler  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣2️⃣ Input of neuron is multiplied by?</strong></summary>

A) Bias  
B) Weight  
C) Loss  
D) Gradient  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣3️⃣ Output of neuron before activation is?</strong></summary>

A) Random value  
B) Linear combination  
C) Image  
D) Label  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣4️⃣ Training improves?</strong></summary>

A) Loss increases  
B) Loss decreases  
C) Data size  
D) Model randomness  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣5️⃣ Gradient tells?</strong></summary>

A) Direction of improvement  
B) Data size  
C) Model type  
D) Output label  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣6️⃣ XOR requires?</strong></summary>

A) Linear model  
B) Non-linear model  
C) No model  
D) Random model  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣7️⃣ Neural networks solve XOR using?</strong></summary>

A) One layer  
B) Multiple layers  
C) No activation  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣8️⃣ Sigmoid is?</strong></summary>

A) Linear  
B) Non-linear  
C) Constant  
D) Random  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣9️⃣ Output of sigmoid is interpreted as?</strong></summary>

A) Loss  
B) Probability  
C) Gradient  
D) Weight  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ Main idea of this lab is?</strong></summary>

A) File handling  
B) Logic gates using neurons  
C) Image processing  
D) Database design  

✅ **Answer: B**
</details>

---
---

# 🟢 Lab 5: MNIST Neural Network (Custom Loop)

> Build a **Multi-Layer Perceptron (MLP)** from scratch to classify handwritten digits using a **custom TensorFlow training loop** with `tf.GradientTape`.

![MNIST](https://img.shields.io/badge/Dataset-MNIST-lightgrey?style=for-the-badge)
![MLP](https://img.shields.io/badge/Model-MLP-green?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 🧠 Lab Summary

| Component | Detail |
|-----------|--------|
| 📊 Dataset | MNIST — 70,000 handwritten digit images |
| 🖼️ Image size | 28 × 28 pixels = 784 inputs after flatten |
| 🎯 Task | 10-class classification (digits 0–9) |
| 🏗️ Model | Multi-Layer Perceptron (MLP) |
| 📉 Loss | Sparse Categorical Cross-Entropy |
| ⚙️ Training | Custom loop with `tf.GradientTape` |

---

## 🖼️ 1. MNIST Dataset

```python
import tensorflow_datasets as tfds
ds = tfds.load('mnist', as_supervised=True)
```

| Property | Value |
|----------|-------|
| Total images | 60,000 train + 10,000 test |
| Image size | 28 × 28 grayscale |
| Classes | 10 (digits 0–9) |
| Label format | Integer (0–9) |

**Data pipeline:**
```python
ds.shuffle(1000).batch(32)
```

---

## 🏗️ 2. Model Architecture (MLP)

```
Input Image (28×28)
       ↓
  Flatten → 784
       ↓
Dense (128, ReLU)
       ↓
Dense (10, Softmax)
       ↓
  Class Prediction
```

| Layer | Output | Activation |
|-------|--------|------------|
| Flatten | 784 | — |
| Dense 1 | 128 | ReLU |
| Dense 2 | 10 | Softmax |

**Why ReLU for hidden layers?**
```
✔ Fast computation
✔ No vanishing gradient
✔ Introduces non-linearity
```

**Why Softmax for output?**
```
✔ Outputs sum to 1 (probabilities)
✔ Picks the most likely class
```

---

## 🔁 3. Training Pipeline

```python
for epoch in range(epochs):
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**One training step:**

```
1️⃣  Forward pass    →  y_pred = model(x)
2️⃣  Compute loss    →  cross_entropy(y, y_pred)
3️⃣  Compute grads   →  GradientTape
4️⃣  Update weights  →  optimizer.apply_gradients()
```

---

## 🔧 4. Key Functions

| Function | Purpose |
|----------|---------|
| `tfds.load()` | Load MNIST dataset |
| `as_supervised=True` | Return (image, label) pairs |
| `Flatten` | Convert 28×28 → 784 vector |
| `Dense` | Fully connected layer (Wx + b) |
| `ReLU` | Hidden layer activation: max(0, x) |
| `Softmax` | Output probabilities |
| `tf.argmax(axis=1)` | Get predicted class index |
| `tf.GradientTape` | Automatic differentiation |

---

## 🎯 Key Takeaways — Lab 5

```
✔ MNIST = 28×28 grayscale digit images (10 classes)
✔ Flatten converts image → 1D vector for Dense layers
✔ ReLU = hidden layer activation
✔ Softmax = output layer → probabilities
✔ Cross-entropy = loss for classification
✔ GradientTape = custom backpropagation
✔ argmax = finds predicted class
```

---

## ❓ Lab 5 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ MNIST dataset is used for?</strong></summary>

A) Text classification  
B) Image segmentation  
C) Handwritten digit classification  
D) Speech recognition  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣ Each MNIST image size is?</strong></summary>

A) 16×16  
B) 28×28  
C) 32×32  
D) 64×64  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣ MNIST labels represent?</strong></summary>

A) Animals  
B) Digits 0–9  
C) Letters A–Z  
D) Colors  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ tfds.load() is used to?</strong></summary>

A) Build models  
B) Load datasets  
C) Plot graphs  
D) Train optimizer  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ as_supervised=True returns?</strong></summary>

A) Images only  
B) Labels only  
C) (input, label) pairs  
D) Random tensors  

✅ **Answer: C**
</details>

<details>
<summary><strong>6️⃣ Main purpose of flatten layer?</strong></summary>

A) Increase image size  
B) Convert 2D to 1D  
C) Normalize data  
D) Add noise  

✅ **Answer: B**
</details>

<details>
<summary><strong>7️⃣ Flatten input 28×28 becomes?</strong></summary>

A) 28  
B) 784  
C) 1024  
D) 256  

✅ **Answer: B**
</details>

<details>
<summary><strong>8️⃣ A Dense layer computes?</strong></summary>

A) x²  
B) Wx + b  
C) log(x)  
D) sin(x)  

✅ **Answer: B**
</details>

<details>
<summary><strong>9️⃣ Activation function adds?</strong></summary>

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise  

✅ **Answer: B**
</details>

<details>
<summary><strong>🔟 ReLU function is?</strong></summary>

A) max(0, x)  
B) min(0, x)  
C) x²  
D) log(x)  

✅ **Answer: A**
</details>

<details>
<summary><strong>1️⃣1️⃣ Softmax outputs?</strong></summary>

A) Binary values  
B) Probabilities  
C) Gradients  
D) Loss values  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ Softmax ensures?</strong></summary>

A) Negative values  
B) Sum = 1  
C) Sum = 0  
D) Random output  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣3️⃣ Loss function used in this lab?</strong></summary>

A) MSE  
B) Cross entropy  
C) Hinge loss  
D) MAE  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ Sparse categorical cross entropy is used for?</strong></summary>

A) Regression  
B) Integer labels  
C) Images only  
D) Clustering  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ Output layer activation is?</strong></summary>

A) ReLU  
B) Softmax  
C) Sigmoid  
D) Linear  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣6️⃣ Optimizer used in lab?</strong></summary>

A) SGD only  
B) Adam  
C) RMSProp only  
D) None  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ Adam optimizer is?</strong></summary>

A) Random  
B) Adaptive learning rate optimizer  
C) Sorting algorithm  
D) Data loader  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ Training goal is to?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Randomize weights  
D) Reduce dataset  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ GradientTape is used for?</strong></summary>

A) Data loading  
B) Automatic differentiation  
C) Image resizing  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ Gradient descent updates?</strong></summary>

A) Data  
B) Weights  
C) Labels  
D) Images  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ MLP stands for?</strong></summary>

A) Multi Layer Perceptron  
B) Matrix Linear Process  
C) Multi Label Prediction  
D) Memory Learning Program  

✅ **Answer: A**
</details>

<details>
<summary><strong>2️⃣2️⃣ Hidden layers typically use?</strong></summary>

A) Softmax  
B) ReLU  
C) Argmax  
D) Sigmoid only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ Output layer predicts?</strong></summary>

A) Images  
B) Classes  
C) Gradients  
D) Weights  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ argmax returns?</strong></summary>

A) Minimum value  
B) Index of max value  
C) Mean value  
D) Loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣5️⃣ argmax(axis=1) means?</strong></summary>

A) Across rows  
B) Across columns  
C) Entire tensor  
D) Random axis  

✅ **Answer: A**
</details>

<details>
<summary><strong>2️⃣6️⃣ Dataset pipeline includes?</strong></summary>

A) Train only  
B) Shuffle and batch  
C) Plot only  
D) Normalize only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ batch() function is used to?</strong></summary>

A) Resize images  
B) Group data  
C) Delete data  
D) Sort labels  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣8️⃣ shuffle() function?</strong></summary>

A) Sorts data  
B) Randomizes order  
C) Deletes samples  
D) Normalizes data  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ Model input is?</strong></summary>

A) 1D vector  
B) 2D image  
C) Text  
D) Audio only  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣0️⃣ First step in model?</strong></summary>

A) Softmax  
B) Flatten  
C) Loss  
D) Optimization  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ Neural network learns by?</strong></summary>

A) Hard coding  
B) Gradient updates  
C) Random guessing  
D) Sorting data  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣2️⃣ Loss measures?</strong></summary>

A) Accuracy  
B) Error  
C) Speed  
D) Memory  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣3️⃣ Backpropagation uses?</strong></summary>

A) argmax  
B) Gradients  
C) Sorting  
D) Convolution  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣4️⃣ Weights represent?</strong></summary>

A) Data size  
B) Importance of inputs  
C) Labels  
D) Loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣5️⃣ Bias helps?</strong></summary>

A) Shift activation  
B) Normalize data  
C) Sort data  
D) Delete noise  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣6️⃣ Neural network output is?</strong></summary>

A) Random  
B) Prediction  
C) Dataset  
D) Image  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣7️⃣ MNIST classification has?</strong></summary>

A) 2 classes  
B) 10 classes  
C) 100 classes  
D) 50 classes  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣8️⃣ One-hot encoding is replaced by?</strong></summary>

A) Sparse labels  
B) Text labels  
C) Images  
D) Noise  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣9️⃣ Training improves?</strong></summary>

A) Loss increases  
B) Accuracy increases  
C) Data disappears  
D) Model resets  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ Main goal of Lab 5?</strong></summary>

A) Image generation  
B) Digit classification  
C) Speech recognition  
D) Object detection  

✅ **Answer: B**
</details>

---
---

# 🔵 Lab 6: Keras Sequential Model (MNIST)

> Classify MNIST handwritten digits using the high-level **Keras Sequential API** — a cleaner, faster way to build and train neural networks compared to custom loops.

![Keras](https://img.shields.io/badge/API-Keras-red?style=for-the-badge&logo=keras&logoColor=white)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-lightgrey?style=for-the-badge)
![Adam](https://img.shields.io/badge/Optimizer-Adam-blue?style=for-the-badge)

---

## 🧠 Lab Summary

| Component | Detail |
|-----------|--------|
| 🏗️ API | Keras Sequential |
| 📊 Dataset | MNIST (28×28 grayscale digits) |
| 🎯 Task | 10-class digit classification |
| 📉 Loss | Sparse Categorical Cross-Entropy |
| ⚙️ Optimizer | Adam |
| 📈 Metric | Accuracy |

---

## 🏗️ 1. Sequential API

The **Sequential model** is a linear stack of layers — ideal for simple feedforward networks.

```python
from tensorflow import keras

model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

| API vs Custom Loop | Sequential | Custom (Lab 5) |
|-------------------|:----------:|:--------------:|
| Code length | Short ✅ | Long |
| Flexibility | Limited | High ✅ |
| Built-in metrics | Yes ✅ | Manual |
| Recommended for | Beginners ✅ | Research |

---

## ⚡ 2. Layers & Activations

### Layer Breakdown

| Layer | Function | Output Shape |
|-------|----------|-------------|
| `keras.Input` | Define input shape | (28, 28) |
| `Flatten` | 2D → 1D | 784 |
| `Dense(128, relu)` | Hidden layer | 128 |
| `Dense(10, softmax)` | Output layer | 10 |

### Activation Summary

| Activation | Formula | Used In |
|------------|---------|---------|
| ReLU | max(0, x) | Hidden layers |
| Softmax | eˣᵢ / Σeˣ | Output layer |

---

## 🔄 3. Model Lifecycle

### Step 1 — Inspect

```python
model.summary()   # Shows architecture, layer shapes, parameters
```

### Step 2 — Compile

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Step 3 — Train

```python
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### Step 4 — Evaluate

```python
model.evaluate(x_test, y_test)   # Returns loss + accuracy
```

### Step 5 — Predict

```python
predictions = model.predict(x_new)          # Returns 10 probabilities
class_idx   = tf.argmax(predictions, axis=1) # Returns class index
```

---

## 🎯 Key Takeaways — Lab 6

```
✔ Sequential = linear stack of layers (easy, fast)
✔ keras.Input() defines input shape
✔ Dense = fully connected layer (Wx + b)
✔ ReLU = hidden activation
✔ Softmax = output probabilities for 10 classes
✔ model.compile() configures optimizer + loss + metrics
✔ model.fit() trains the model
✔ model.evaluate() returns loss + accuracy
✔ model.predict() returns class probabilities
✔ argmax() converts probabilities → class index
```

---

## ❓ Lab 6 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ Keras Sequential model is used for?</strong></summary>

A) Complex graph models  
B) Linear stack of layers  
C) Databases  
D) Image storage  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣ Sequential model means?</strong></summary>

A) Random structure  
B) Layers connected one after another  
C) Parallel networks  
D) No layers  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣ Best use case for Sequential model?</strong></summary>

A) CNN with branches  
B) Simple feedforward networks  
C) Graph models  
D) Reinforcement learning  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ First layer in Sequential model defines?</strong></summary>

A) Loss  
B) Input shape  
C) Optimizer  
D) Output  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ keras.Input() is used to?</strong></summary>

A) Train model  
B) Define input shape  
C) Normalize data  
D) Compute loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>6️⃣ Dense layer is?</strong></summary>

A) Convolution layer  
B) Fully connected layer  
C) Pooling layer  
D) Dropout layer  

✅ **Answer: B**
</details>

<details>
<summary><strong>7️⃣ Dense layer computes?</strong></summary>

A) x²  
B) Wx + b  
C) log(x)  
D) Sorting  

✅ **Answer: B**
</details>

<details>
<summary><strong>8️⃣ Activation function adds?</strong></summary>

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise only  

✅ **Answer: B**
</details>

<details>
<summary><strong>9️⃣ ReLU stands for?</strong></summary>

A) Random Linear Unit  
B) Rectified Linear Unit  
C) Regression Linear Unit  
D) Reduced Linear Unit  

✅ **Answer: B**
</details>

<details>
<summary><strong>🔟 ReLU function is?</strong></summary>

A) max(0, x)  
B) min(0, x)  
C) x²  
D) log(x)  

✅ **Answer: A**
</details>

<details>
<summary><strong>1️⃣1️⃣ Softmax output represents?</strong></summary>

A) Loss  
B) Probabilities  
C) Gradients  
D) Images  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ Softmax ensures?</strong></summary>

A) Negative outputs  
B) Sum of probabilities = 1  
C) Random output  
D) Zero output  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣3️⃣ MNIST dataset contains?</strong></summary>

A) Images of animals  
B) Handwritten digits  
C) Text documents  
D) Audio signals  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ MNIST classes are?</strong></summary>

A) 5  
B) 10  
C) 100  
D) 2  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ Flatten layer does?</strong></summary>

A) Expands image  
B) Converts 2D to 1D  
C) Adds noise  
D) Normalizes labels  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣6️⃣ MNIST image size is?</strong></summary>

A) 32×32  
B) 28×28  
C) 64×64  
D) 16×16  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ model.summary() shows?</strong></summary>

A) Loss values  
B) Model architecture  
C) Dataset info  
D) Predictions  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ model.compile() is used for?</strong></summary>

A) Building dataset  
B) Configuring training  
C) Plotting graphs  
D) Saving model  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ Loss function used in MNIST?</strong></summary>

A) MSE  
B) Cross entropy  
C) MAE  
D) Hinge loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ Sparse categorical crossentropy is used for?</strong></summary>

A) Regression  
B) Integer labels  
C) Images only  
D) Clustering  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ Optimizer commonly used?</strong></summary>

A) SGD only  
B) Adam  
C) Random optimizer  
D) None  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣2️⃣ model.fit() is used for?</strong></summary>

A) Testing model  
B) Training model  
C) Saving model  
D) Loading dataset  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ model.evaluate() returns?</strong></summary>

A) Dataset  
B) Loss and accuracy  
C) Weights  
D) Gradients  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣4️⃣ model.predict() gives?</strong></summary>

A) Labels only  
B) Probabilities  
C) Loss  
D) Gradients  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣5️⃣ argmax is used to?</strong></summary>

A) Find minimum  
B) Find class index  
C) Normalize data  
D) Train model  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ argmax returns?</strong></summary>

A) Value  
B) Index  
C) Loss  
D) Image  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣7️⃣ Input to model must be?</strong></summary>

A) String  
B) Tensor  
C) File  
D) Image only  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣8️⃣ Hidden layers usually use?</strong></summary>

A) Softmax  
B) ReLU  
C) Argmax  
D) Loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ Output layer for MNIST uses?</strong></summary>

A) ReLU  
B) Softmax  
C) Sigmoid  
D) Linear  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣0️⃣ Training goal is?</strong></summary>

A) Increase loss  
B) Minimize loss  
C) Shuffle data  
D) Delete data  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣1️⃣ Backpropagation uses?</strong></summary>

A) Gradients  
B) Sorting  
C) Random values  
D) Images  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣2️⃣ Weights represent?</strong></summary>

A) Output labels  
B) Importance of features  
C) Loss values  
D) Images  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣3️⃣ Bias helps?</strong></summary>

A) Shift activation  
B) Normalize data  
C) Delete data  
D) Sort inputs  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣4️⃣ Learning improves?</strong></summary>

A) Accuracy  
B) Data size  
C) Image resolution  
D) Memory  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣5️⃣ Each MNIST prediction gives?</strong></summary>

A) One value  
B) 10 probabilities  
C) 28 values  
D) Random output  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣6️⃣ Sequential model is?</strong></summary>

A) Graph model  
B) Stack of layers  
C) Tree model  
D) Database  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣7️⃣ Training uses?</strong></summary>

A) Forward only  
B) Forward + backward pass  
C) Sorting  
D) Clustering  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣8️⃣ Loss measures?</strong></summary>

A) Accuracy  
B) Error  
C) Speed  
D) Memory  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣9️⃣ MNIST classification type?</strong></summary>

A) Regression  
B) Multi-class classification  
C) Clustering  
D) Detection  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ Main goal of Lab 6?</strong></summary>

A) Image generation  
B) Build MNIST classifier using Keras  
C) Object detection  
D) Text classification  

✅ **Answer: B**
</details>

---
---

# 🔴 Lab 7: CNN for CIFAR-10 Image Classification

> Use **Convolutional Neural Networks (CNNs)** with Keras to classify 10 categories of color images from the **CIFAR-10** dataset — learning how CNNs automatically extract visual features.

![CNN](https://img.shields.io/badge/Model-CNN-red?style=for-the-badge)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-orange?style=for-the-badge)
![Keras](https://img.shields.io/badge/API-Keras-red?style=for-the-badge&logo=keras&logoColor=white)

---

## 🧠 Lab Summary

| Component | Detail |
|-----------|--------|
| 📊 Dataset | CIFAR-10 — 60,000 color images |
| 🖼️ Image size | 32 × 32 × 3 (RGB) |
| 🎯 Classes | 10 (airplane, car, cat, dog, etc.) |
| 🏗️ Model | Convolutional Neural Network (CNN) |
| 📉 Loss | Categorical Cross-Entropy |
| ⚙️ Optimizer | Adam |

---

## 🖼️ 1. CIFAR-10 Dataset

| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Image size | 32 × 32 pixels, RGB |
| Classes | 10 |
| Train / Test split | 50,000 / 10,000 |

**The 10 classes:**

| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
|:-----------:|:-------------:|:-------:|:------:|:-------:|
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck |

**Preprocessing:**
```python
# Normalize pixel values to [0, 1]
images = images / 255.0

# One-hot encode labels (e.g. Cat → [0,0,1,0,0,0,0,0,0,0])
labels = keras.utils.to_categorical(labels, 10)
```

---

## 🧱 2. CNN Building Blocks

### 🔷 Conv2D — Feature Extraction

Scans the image with learned filters to detect edges, textures, and shapes.

```python
keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
```

| Layer | Learns |
|-------|--------|
| Conv 1 | Edges & corners |
| Conv 2 | Textures & patterns |
| Conv 3 | Shapes & parts |
| Dense | Object class |

---

### 🔷 MaxPooling2D — Downsampling

Reduces spatial dimensions while keeping the strongest features.

```python
keras.layers.MaxPooling2D(pool_size=(2,2))
```

```
✔ Reduces computation
✔ Keeps dominant features
✔ Makes model more robust
```

---

### 🔷 Flatten

Converts the 2D feature map → 1D vector, required before Dense layers.

```python
keras.layers.Flatten()
```

---

### 🔷 Dense — Classification

Fully connected layers that perform the final decision.

```python
keras.layers.Dense(128, activation='relu')
keras.layers.Dense(10,  activation='softmax')
```

---

## 🏗️ 3. CNN Model Structure

```
Input Image (32×32×3)
        ↓
Conv2D(32, 3×3) + ReLU      ← detect edges
        ↓
MaxPooling2D(2×2)            ← downsample
        ↓
Conv2D(64, 3×3) + ReLU      ← detect shapes
        ↓
MaxPooling2D(2×2)            ← downsample
        ↓
Flatten                       ← 2D → 1D
        ↓
Dense(128, ReLU)             ← reasoning
        ↓
Dense(10, Softmax)           ← class probabilities
```

---

## 🔄 4. Data Augmentation

Artificially increases training variety to **prevent overfitting**.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
```

| Technique | Effect |
|-----------|--------|
| 🔄 Rotation | Random angle rotation |
| ↔️ Width/Height shift | Random translation |
| 🔁 Horizontal flip | Mirror image |
| 🔍 Zoom | Random scale change |

---

## ⚙️ 5. Model Training Process

```python
# Step 1 — Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 2 — Train
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Step 3 — Evaluate
model.evaluate(test_images, test_labels)

# Step 4 — Predict
predictions = model.predict(new_image)
```

---

## 🎯 Key Takeaways — Lab 7

```
✔ CNN = best architecture for image data
✔ Conv2D = learns filters automatically (no manual feature engineering)
✔ MaxPooling = reduces size, keeps key features
✔ Flatten = bridges Conv layers → Dense layers
✔ Data augmentation = prevents overfitting
✔ Softmax output → 10 class probabilities
✔ CNN learns: edges → textures → shapes → objects
```

---

## ❓ Lab 7 MCQs (40 Questions)

<details>
<summary><strong>1️⃣ CIFAR-10 dataset contains how many classes?</strong></summary>

A) 5  
B) 10  
C) 100  
D) 20  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣ Each CIFAR-10 image size is?</strong></summary>

A) 28×28  
B) 64×64  
C) 32×32  
D) 128×128  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣ CIFAR-10 images are?</strong></summary>

A) Grayscale  
B) RGB color images  
C) Binary images  
D) Text images  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣ CIFAR-10 is mainly used for?</strong></summary>

A) NLP  
B) Image classification  
C) Speech recognition  
D) Reinforcement learning  

✅ **Answer: B**
</details>

<details>
<summary><strong>5️⃣ CIFAR-10 dataset has approximately?</strong></summary>

A) 10,000 images  
B) 50,000 images  
C) 60,000 images  
D) 1 million images  

✅ **Answer: C**
</details>

<details>
<summary><strong>6️⃣ Which of the following is NOT a CIFAR-10 class?</strong></summary>

A) Cat  
B) Dog  
C) Car  
D) Lion  

✅ **Answer: D**
</details>

<details>
<summary><strong>7️⃣ CIFAR-10 images are typically used in?</strong></summary>

A) CNN models  
B) RNN models  
C) LSTM models  
D) GAN only  

✅ **Answer: A**
</details>

<details>
<summary><strong>8️⃣ Input shape of CIFAR-10 images is usually?</strong></summary>

A) (28, 28, 1)  
B) (32, 32, 3)  
C) (64, 64, 1)  
D) (10, 10, 3)  

✅ **Answer: B**
</details>

<details>
<summary><strong>9️⃣ Conv2D layer is used for?</strong></summary>

A) Text processing  
B) Feature extraction from images  
C) Sorting data  
D) Normalization only  

✅ **Answer: B**
</details>

<details>
<summary><strong>🔟 MaxPooling is used to?</strong></summary>

A) Increase image size  
B) Reduce spatial dimensions  
C) Add noise  
D) Convert labels  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣1️⃣ Flatten layer converts?</strong></summary>

A) 1D → 2D  
B) 2D → 1D  
C) RGB → grayscale  
D) Labels → images  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣2️⃣ CNN is especially good for?</strong></summary>

A) Tabular data  
B) Image data  
C) Audio only  
D) Text only  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣3️⃣ Conv2D learns?</strong></summary>

A) Weights manually  
B) Filters automatically  
C) Labels  
D) Loss only  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣4️⃣ Pooling layer helps in?</strong></summary>

A) Overfitting  
B) Feature reduction  
C) Increasing noise  
D) Label encoding  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣5️⃣ MaxPooling selects?</strong></summary>

A) Average value  
B) Minimum value  
C) Maximum value  
D) Random value  

✅ **Answer: C**
</details>

<details>
<summary><strong>1️⃣6️⃣ CNN first learns?</strong></summary>

A) Shapes  
B) Edges  
C) Labels  
D) Loss  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣7️⃣ Flatten layer is required before?</strong></summary>

A) Conv layer  
B) Dense layer  
C) Pooling layer  
D) Input layer  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣8️⃣ CNN stands for?</strong></summary>

A) Central Neural Network  
B) Convolutional Neural Network  
C) Core Neural Node  
D) Complex Numeric Network  

✅ **Answer: B**
</details>

<details>
<summary><strong>1️⃣9️⃣ Keras Sequential model is?</strong></summary>

A) Graph-based model  
B) Linear stack of layers  
C) Tree model  
D) Rule-based system  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣0️⃣ First layer in CNN must define?</strong></summary>

A) Loss  
B) Input shape  
C) Optimizer  
D) Labels  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣1️⃣ Softmax activation is used for?</strong></summary>

A) Regression  
B) Classification probabilities  
C) Feature extraction  
D) Pooling  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣2️⃣ ReLU activation outputs?</strong></summary>

A) Negative values only  
B) 0 or positive values  
C) Only 1 and 0  
D) Probabilities  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣3️⃣ model.compile() is used to?</strong></summary>

A) Build model layers  
B) Train model  
C) Configure training  
D) Predict output  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣4️⃣ model.fit() is used to?</strong></summary>

A) Train model  
B) Test model  
C) Save model  
D) Compile model  

✅ **Answer: A**
</details>

<details>
<summary><strong>2️⃣5️⃣ model.evaluate() is used to?</strong></summary>

A) Train model  
B) Test model  
C) Build model  
D) Predict labels  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣6️⃣ model.predict() is used for?</strong></summary>

A) Training  
B) Evaluation  
C) Inference  
D) Compilation  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣7️⃣ Loss function in CIFAR-10 is usually?</strong></summary>

A) MSE  
B) MAE  
C) Cross-entropy  
D) Hinge loss  

✅ **Answer: C**
</details>

<details>
<summary><strong>2️⃣8️⃣ Optimizer commonly used is?</strong></summary>

A) SGD only  
B) Adam  
C) Random  
D) None  

✅ **Answer: B**
</details>

<details>
<summary><strong>2️⃣9️⃣ Data augmentation helps to?</strong></summary>

A) Increase dataset size artificially  
B) Reduce model size  
C) Remove labels  
D) Remove images  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣0️⃣ Which is NOT an augmentation technique?</strong></summary>

A) Rotation  
B) Flipping  
C) Normalization  
D) Shuffling labels  

✅ **Answer: D**
</details>

<details>
<summary><strong>3️⃣1️⃣ ImageDataGenerator is used for?</strong></summary>

A) Model saving  
B) Real-time augmentation  
C) Loss calculation  
D) Feature extraction  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣2️⃣ Horizontal flip means?</strong></summary>

A) Rotate image  
B) Flip left-right  
C) Flip up-down  
D) Crop image  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣3️⃣ Augmentation helps prevent?</strong></summary>

A) Underfitting  
B) Overfitting  
C) Training  
D) Evaluation  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣4️⃣ Zooming is used to?</strong></summary>

A) Remove images  
B) Change image scale  
C) Convert labels  
D) Normalize data  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣5️⃣ CNN learns features in order?</strong></summary>

A) Random → shapes → edges  
B) Edges → textures → shapes  
C) Shapes → edges → labels  
D) Labels → edges → textures  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣6️⃣ CNN is inspired by?</strong></summary>

A) Human brain visual cortex  
B) Database systems  
C) Sorting algorithms  
D) Decision trees  

✅ **Answer: A**
</details>

<details>
<summary><strong>3️⃣7️⃣ CIFAR-10 output layer has?</strong></summary>

A) 2 neurons  
B) 5 neurons  
C) 10 neurons  
D) 100 neurons  

✅ **Answer: C**
</details>

<details>
<summary><strong>3️⃣8️⃣ One-hot encoding converts?</strong></summary>

A) Images → numbers  
B) Labels → vectors  
C) Pixels → labels  
D) Loss → accuracy  

✅ **Answer: B**
</details>

<details>
<summary><strong>3️⃣9️⃣ CNN reduces parameters using?</strong></summary>

A) Fully connected layers  
B) Weight sharing  
C) Random weights  
D) Label smoothing  

✅ **Answer: B**
</details>

<details>
<summary><strong>4️⃣0️⃣ Main advantage of CNN over Dense network is?</strong></summary>

A) Faster text processing  
B) Better image feature learning  
C) Lower memory always  
D) No training needed  

✅ **Answer: B**
</details>

---
---

## 🧠 Combined Key Concepts (All Labs)

| Concept | Library / Tool | Meaning |
|---------|---------------|---------|
| `ndarray` | NumPy | N-dimensional array |
| `Series` | Pandas | 1D labeled data structure |
| `DataFrame` | Pandas | 2D table structure |
| `tf.Tensor` | TensorFlow | Immutable n-dimensional array |
| `tf.Variable` | TensorFlow | Trainable model parameter |
| `GradientTape` | TensorFlow | Automatic differentiation |
| `@tf.function` | TensorFlow | Graph-mode optimization |
| `tf.Module` | TensorFlow | Custom model container |
| `assign_sub()` | TensorFlow | In-place weight update |
| `tf.data.Dataset` | TensorFlow | Efficient data pipeline |
| MSE | Loss Function | Regression error measure |
| Sigmoid | Activation | Output in (0,1) — probability |
| ReLU | Activation | max(0, x) — hidden layers |
| Softmax | Activation | Multi-class probabilities |
| `Dense` | Keras | Fully connected layer (Wx + b) |
| `Flatten` | Keras | 2D → 1D conversion |
| `Conv2D` | Keras | Spatial feature extraction |
| `MaxPooling2D` | Keras | Downsampling feature maps |
| Cross-entropy | Loss Function | Classification error measure |
| Adam | Optimizer | Adaptive learning rate |
| Data Augmentation | Technique | Prevent overfitting |
| `GroupBy` | Pandas | Grouping rows by shared value |
| Histogram | Matplotlib | Data distribution plot |

---

## 🌍 Why These Labs Matter

These labs build the **complete Deep Learning pipeline** from foundations to CNNs:

| Lab | Topic | Builds Toward |
|-----|-------|---------------|
| Lab 1 | NumPy / Pandas / Matplotlib | Data handling & visualization |
| Lab 2 | TensorFlow Tensors & Variables | Model architecture basics |
| Lab 3 | Linear Regression | Optimization & custom training |
| Lab 4 | Neurons & Logic Gates | Perceptron & activation functions |
| Lab 5 | MNIST MLP (custom loop) | Full neural network pipeline |
| Lab 6 | Keras Sequential MNIST | High-level model building |
| Lab 7 | CNN CIFAR-10 | Image classification & CNNs |

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

