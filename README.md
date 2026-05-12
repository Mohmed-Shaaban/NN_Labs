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
- [Lab 8 — Transfer Learning with DenseNet121](#-lab-8-transfer-learning-with-densenet121)
  - [What is Transfer Learning?](#-what-is-transfer-learning)
  - [DenseNet121](#-what-is-densenet121)
  - [Feature Extraction](#-feature-extraction-strategy)
  - [Fine-Tuning](#-fine-tuning)
  - [Training Strategies](#-three-training-strategies)
  - [Lab 8 MCQs](#-lab-8-mcqs-40-questions)
- [Lab 9 — LSTM on 20 Newsgroups](#-lab-9-lstm-on-20-newsgroups-dataset)
  - [20 Newsgroups Dataset](#-20-newsgroups-dataset)
  - [Text Preprocessing](#-text-preprocessing)
  - [Embedding Layer](#-embedding-layer)
  - [LSTM & Bidirectional LSTM](#-lstm--bidirectional-lstm)
  - [Model Architecture](#-model-architecture)
  - [Lab 9 MCQs](#-lab-9-mcqs-40-questions)
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
| **Lab 8** | Transfer Learning with DenseNet121 | DenseNet121, Feature Extraction, Fine-Tuning |
| **Lab 9** | LSTM on 20 Newsgroups | LSTM, Bidirectional, Tokenizer, Embedding |

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

**1️⃣ What is the primary purpose of NumPy?**

A) Web development  
B) Numerical computing and arrays  
C) Game design  
D) Database management  

✅ **Answer: B**


**2️⃣ Which NumPy object represents an n-dimensional array?**

A) Series  
B) DataFrame  
C) ndarray  
D) Tensor  

✅ **Answer: C**


**3️⃣ What will be the output of the following code?**

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


**4️⃣ Which function creates evenly spaced values within an interval?**

A) np.zeros()  
B) np.ones()  
C) np.arange()  
D) np.reshape()  

✅ **Answer: C**


**5️⃣ What is the output of np.arange(0,5,2)?**

A) [0 1 2 3 4]  
B) [0 2 4]  
C) [2 4 6]  
D) [0 2 4 6]  

✅ **Answer: B**


**6️⃣ Which function creates an array filled with zeros?**

A) np.empty()  
B) np.full()  
C) np.zeros()  
D) np.array()  

✅ **Answer: C**


**7️⃣ What is the shape of np.zeros((2,3))?**

A) 2 rows and 3 columns  
B) 3 rows and 2 columns  
C) 6 rows  
D) 1 row and 6 columns  

✅ **Answer: A**


**8️⃣ Which function creates an array of ones?**

A) np.ones()  
B) np.zeros()  
C) np.identity()  
D) np.random()  

✅ **Answer: A**


**9️⃣ What does np.random.randint() generate?**

A) Floating-point numbers  
B) Strings  
C) Random integers  
D) Boolean values  

✅ **Answer: C**


**🔟 Which function generates random values between 0 and 1?**

A) np.random.randint()  
B) np.random.rand()  
C) np.arange()  
D) np.linspace()  

✅ **Answer: B**


**1️⃣1️⃣ What does np.linspace() do?**

A) Creates random values  
B) Creates evenly spaced values  
C) Creates identity matrix  
D) Creates zeros  

✅ **Answer: B**


**1️⃣2️⃣ What is the output of np.linspace(0,10,3)?**

A) [0 10]  
B) [0 5 10]  
C) [1 5 9]  
D) [0 2 4]  

✅ **Answer: B**


**1️⃣3️⃣ What does reshape() do?**

A) Deletes data  
B) Changes array shape  
C) Sorts values  
D) Adds elements  

✅ **Answer: B**


**1️⃣4️⃣ Which shape is valid for reshaping an array of 6 elements?**

A) (2,3)  
B) (4,4)  
C) (5,2)  
D) (7,1)  

✅ **Answer: A**


**1️⃣5️⃣ What does arr.max() return?**

A) Minimum value  
B) Mean value  
C) Maximum value  
D) Index of maximum value  

✅ **Answer: C**


**1️⃣6️⃣ What does arr.mean() compute?**

A) Median  
B) Mode  
C) Average  
D) Variance  

✅ **Answer: C**


**1️⃣7️⃣ What does argmax() return?**

A) Largest value  
B) Index of largest value  
C) Smallest value  
D) Mean value  

✅ **Answer: B**


**1️⃣8️⃣ Which operation accesses specific elements of an array?**

A) Plotting  
B) Reshaping  
C) Indexing  
D) Pooling  

✅ **Answer: C**


**1️⃣9️⃣ What is slicing used for?**

A) Deleting arrays  
B) Accessing subarrays  
C) Random generation  
D) Sorting arrays  

✅ **Answer: B**


**2️⃣0️⃣ Which library is mainly used for data analysis?**

A) TensorFlow  
B) Pandas  
C) OpenCV  
D) Flask  

✅ **Answer: B**


**2️⃣1️⃣ What is a Pandas Series?**

A) 2D table  
B) Image object  
C) 1D labeled array  
D) Neural network  

✅ **Answer: C**


**2️⃣2️⃣ What is a DataFrame?**

A) One-dimensional array  
B) Two-dimensional labeled table  
C) Neural network layer  
D) Random generator  

✅ **Answer: B**


**2️⃣3️⃣ Which function reads CSV files?**

A) pd.load()  
B) pd.read_csv()  
C) pd.import()  
D) pd.open()  

✅ **Answer: B**


**2️⃣4️⃣ What does df.head() display?**

A) Last rows  
B) Random rows  
C) First rows  
D) Middle rows  

✅ **Answer: C**


**2️⃣5️⃣ What does df.tail() display?**

A) First rows  
B) Last rows  
C) Random rows  
D) Column names  

✅ **Answer: B**


**2️⃣6️⃣ Which method uses labels for indexing?**

A) iloc[]  
B) loc[]  
C) mean()  
D) tail()  

✅ **Answer: B**


**2️⃣7️⃣ Which method uses integer positions?**

A) loc[]  
B) iloc[]  
C) groupby()  
D) plot()  

✅ **Answer: B**


**2️⃣8️⃣ What does groupby() do?**

A) Deletes groups  
B) Groups rows based on values  
C) Creates plots  
D) Reshapes arrays  

✅ **Answer: B**


**2️⃣9️⃣ Which statistical function computes average?**

A) median()  
B) std()  
C) mean()  
D) var()  

✅ **Answer: C**


**3️⃣0️⃣ Which function calculates standard deviation?**

A) mean()  
B) std()  
C) var()  
D) median()  

✅ **Answer: B**


**3️⃣1️⃣ Which function calculates variance?**

A) std()  
B) mode()  
C) var()  
D) mean()  

✅ **Answer: C**


**3️⃣2️⃣ Which library is used for plotting graphs?**

A) NumPy  
B) Pandas  
C) Matplotlib  
D) SciPy  

✅ **Answer: C**


**3️⃣3️⃣ What does plt.hist() create?**

A) Line plot  
B) Pie chart  
C) Histogram  
D) Scatter plot  

✅ **Answer: C**


**3️⃣4️⃣ What is a histogram used for?**

A) Data distribution visualization  
B) Sorting arrays  
C) Neural network training  
D) CSV reading  

✅ **Answer: A**


**3️⃣5️⃣ Which function creates line plots?**

A) plt.hist()  
B) plt.plot()  
C) plt.array()  
D) plt.group()  

✅ **Answer: B**


**3️⃣6️⃣ What does plt.show() do?**

A) Saves file  
B) Displays plot  
C) Deletes plot  
D) Reads plot  

✅ **Answer: B**


**3️⃣7️⃣ Which library provides fast numerical operations?**

A) Pandas  
B) Matplotlib  
C) NumPy  
D) Flask  

✅ **Answer: C**


**3️⃣8️⃣ Which structure is similar to an Excel sheet?**

A) ndarray  
B) Series  
C) DataFrame  
D) Tuple  

✅ **Answer: C**


**3️⃣9️⃣ Which function creates random floating-point values?**

A) randint()  
B) rand()  
C) arange()  
D) zeros()  

✅ **Answer: B**


**4️⃣0️⃣ What is the purpose of Matplotlib?**

A) Database management  
B) Web development  
C) Data visualization  
D) Audio processing  

✅ **Answer: C**


**4️⃣1️⃣ Which operation changes array dimensions?**

A) reshape()  
B) mean()  
C) groupby()  
D) loc[]  

✅ **Answer: A**


**4️⃣2️⃣ What does df.iloc[:,1] select?**

A) First row  
B) Second column  
C) Last row  
D) Entire DataFrame  

✅ **Answer: B**


**4️⃣3️⃣ Which function returns the index of the minimum value?**

A) argmax()  
B) min()  
C) argmin()  
D) mean()  

✅ **Answer: C**


**4️⃣4️⃣ Which NumPy function creates arrays with evenly spaced values including decimals?**

A) arange()  
B) linspace()  
C) randint()  
D) zeros()  

✅ **Answer: B**


**4️⃣5️⃣ Which command creates a 2×2 matrix of ones?**

A) np.zeros((2,2))  
B) np.ones((2,2))  
C) np.arange((2,2))  
D) np.rand((2,2))  

✅ **Answer: B**


**4️⃣6️⃣ Which Pandas object is one-dimensional?**

A) DataFrame  
B) ndarray  
C) Series  
D) Matrix  

✅ **Answer: C**


**4️⃣7️⃣ What does filtering in Pandas mean?**

A) Removing libraries  
B) Selecting rows based on conditions  
C) Plotting graphs  
D) Reshaping arrays  

✅ **Answer: B**


**4️⃣8️⃣ Which command accesses the first element of a NumPy array arr?**

A) arr(0)  
B) arr[0]  
C) arr{0}  
D) arr.first()  

✅ **Answer: B**


**4️⃣9️⃣ Which library is commonly used together with NumPy and Pandas for plotting?**

A) OpenCV  
B) Flask  
C) Matplotlib  
D) Django  

✅ **Answer: C**


**5️⃣0️⃣ What is the main purpose of this lab?**

A) Build websites  
B) Learn Python data science libraries  
C) Develop mobile apps  
D) Create databases  

✅ **Answer: B**


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

**1️⃣ What is TensorFlow mainly used for?**

A) Web development  
B) Deep learning and numerical computation  
C) Game development  
D) Operating systems  

✅ **Answer: B**


**2️⃣ What is a TensorFlow tensor?**

A) Mutable array  
B) Database table  
C) N-dimensional immutable array  
D) String object  

✅ **Answer: C**


**3️⃣ Which execution mode does TensorFlow 2.x use by default?**

A) Graph mode only  
B) Eager execution  
C) Batch mode  
D) Static mode  

✅ **Answer: B**


**4️⃣ Which object is mutable in TensorFlow?**

A) tf.Tensor  
B) tf.Variable  
C) tf.constant  
D) tf.function  

✅ **Answer: B**


**5️⃣ What does tf.constant create?**

A) Variable tensor  
B) Immutable tensor  
C) Random tensor  
D) Sparse matrix  

✅ **Answer: B**


**6️⃣ What is tf.Variable used for?**

A) Plotting graphs  
B) Storing trainable parameters  
C) Data cleaning  
D) File reading  

✅ **Answer: B**


**7️⃣ What does assign() do?**

A) Deletes variable  
B) Updates variable value  
C) Converts tensor  
D) Creates graph  

✅ **Answer: B**


**8️⃣ What does tf.add() do?**

A) Matrix inversion  
B) Element-wise addition  
C) Sorting  
D) Normalization  

✅ **Answer: B**


**9️⃣ What does tf.multiply() do?**

A) Element-wise multiplication  
B) Matrix transpose  
C) Loss calculation  
D) Gradient computation  

✅ **Answer: A**


**🔟 What does tf.linalg.matmul() perform?**

A) Scalar multiplication  
B) Matrix multiplication  
C) Random sampling  
D) Sorting  

✅ **Answer: B**


**1️⃣1️⃣ tf.random.normal generates values from which distribution?**

A) Uniform distribution  
B) Normal distribution  
C) Bernoulli distribution  
D) Poisson distribution  

✅ **Answer: B**


**1️⃣2️⃣ What does tf.reduce_mean calculate?**

A) Maximum value  
B) Minimum value  
C) Mean value  
D) Median  

✅ **Answer: C**


**1️⃣3️⃣ What is tf.Tensor?**

A) Mutable variable  
B) Immutable data structure  
C) File format  
D) Model optimizer  

✅ **Answer: B**


**1️⃣4️⃣ What is tf.RaggedTensor used for?**

A) Fixed-size arrays  
B) Variable-length sequences  
C) Images only  
D) Scalars only  

✅ **Answer: B**


**1️⃣5️⃣ What is tf.SparseTensor used for?**

A) Dense matrices  
B) Mostly zero data  
C) Text only  
D) Images  

✅ **Answer: B**


**1️⃣6️⃣ What does @tf.function do?**

A) Slows execution  
B) Converts function into graph  
C) Deletes function  
D) Prints tensor  

✅ **Answer: B**


**1️⃣7️⃣ Why use @tf.function?**

A) Debugging  
B) Performance optimization  
C) Data cleaning  
D) Visualization  

✅ **Answer: B**


**1️⃣8️⃣ What does tf.GradientTape do?**

A) Saves model  
B) Computes gradients  
C) Loads data  
D) Plots graphs  

✅ **Answer: B**


**1️⃣9️⃣ Gradients are used for?**

A) Sorting data  
B) Optimization  
C) Printing values  
D) File handling  

✅ **Answer: B**


**2️⃣0️⃣ What is backpropagation?**

A) Forward prediction  
B) Gradient computation process  
C) Data loading  
D) Plotting graphs  

✅ **Answer: B**


**2️⃣1️⃣ What shape does tf.constant support?**

A) Only 1D  
B) Only 2D  
C) Multi-dimensional  
D) Only scalars  

✅ **Answer: C**


**2️⃣2️⃣ What is eager execution?**

A) Delayed execution  
B) Immediate execution  
C) Parallel execution  
D) Random execution  

✅ **Answer: B**


**2️⃣3️⃣ What happens inside @tf.function?**

A) Python runs slowly  
B) Code is optimized into a graph  
C) Memory is deleted  
D) Data is lost  

✅ **Answer: B**


**2️⃣4️⃣ What is tf.Variable best for?**

A) Static data  
B) Trainable weights  
C) Images  
D) Strings  

✅ **Answer: B**


**2️⃣5️⃣ What does matmul require?**

A) Scalars  
B) Shape-compatible tensors  
C) Strings  
D) Lists only  

✅ **Answer: B**


**2️⃣6️⃣ Random tensors are used for?**

A) Debugging  
B) Weight initialization  
C) Printing  
D) Sorting  

✅ **Answer: B**


**2️⃣7️⃣ tf.reduce_mean reduces which dimension?**

A) Rows only  
B) Columns only  
C) Any specified dimension  
D) Variables  

✅ **Answer: C**


**2️⃣8️⃣ TensorFlow supports which hardware?**

A) Only CPU  
B) CPU and GPU  
C) Only GPU  
D) Cloud only  

✅ **Answer: B**


**2️⃣9️⃣ GradientTape is used in?**

A) Data loading  
B) Training models  
C) Visualization  
D) File storage  

✅ **Answer: B**


**3️⃣0️⃣ Which TensorFlow object is immutable?**

A) tf.Variable  
B) tf.Tensor  
C) Python list  
D) NumPy array  

✅ **Answer: B**


**3️⃣1️⃣ What does assign_add() do?**

A) Multiply  
B) Add a value to the variable  
C) Delete variable  
D) Normalize  

✅ **Answer: B**


**3️⃣2️⃣ TensorFlow computation is based on?**

A) Trees  
B) Graphs  
C) Lists  
D) Files  

✅ **Answer: B**


**3️⃣3️⃣ What improves performance in TensorFlow?**

A) tf.constant  
B) tf.function  
C) tf.print  
D) tf.list  

✅ **Answer: B**


**3️⃣4️⃣ Which is NOT a TensorFlow data structure?**

A) Tensor  
B) Variable  
C) DataFrame  
D) SparseTensor  

✅ **Answer: C**


**3️⃣5️⃣ What does tf.constant store?**

A) Trainable weights  
B) Fixed values  
C) Graphs  
D) Models  

✅ **Answer: B**


**3️⃣6️⃣ Matrix multiplication output depends on?**

A) Shape compatibility  
B) Random seed  
C) Loss function  
D) Dataset size  

✅ **Answer: A**


**3️⃣7️⃣ tf.random.normal is useful for?**

A) Loss calculation  
B) Weight initialization  
C) File reading  
D) Sorting  

✅ **Answer: B**


**3️⃣8️⃣ Gradient descent uses?**

A) Random search  
B) Gradients  
C) Sorting  
D) Clustering  

✅ **Answer: B**


**3️⃣9️⃣ What is the main goal of TensorFlow training?**

A) Increase loss  
B) Minimize loss  
C) Delete model  
D) Freeze weights  

✅ **Answer: B**


**4️⃣0️⃣ Which concept links GradientTape to model training?**

A) Data loading  
B) Backpropagation  
C) Visualization  
D) Reshaping  

✅ **Answer: B**


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

**1️⃣ What is the main goal of linear regression?**

A) Classification  
B) Clustering  
C) Predict continuous values  
D) Image detection  

✅ **Answer: C**


**2️⃣ Linear regression models the relationship between?**

A) Images and labels  
B) Inputs and continuous outputs  
C) Text and speech  
D) Pixels only  

✅ **Answer: B**


**3️⃣ What is the general equation of linear regression?**

A) y = x²  
B) y = wx + b  
C) y = log(x)  
D) y = sin(x)  

✅ **Answer: B**


**4️⃣ In y = wx + b, what is w?**

A) Bias  
B) Weight  
C) Loss  
D) Gradient  

✅ **Answer: B**


**5️⃣ In y = wx + b, what is b?**

A) Bias  
B) Batch  
C) Gradient  
D) Feature  

✅ **Answer: A**


**6️⃣ What is used to measure error in this lab?**

A) Accuracy  
B) Cross entropy  
C) Mean Squared Error  
D) Precision  

✅ **Answer: C**


**7️⃣ MSE stands for?**

A) Mean Standard Error  
B) Mean Squared Error  
C) Model Score Estimation  
D) Mean Sample Error  

✅ **Answer: B**


**8️⃣ What does MSE measure?**

A) Model speed  
B) Difference between predictions and true values  
C) Data size  
D) Learning rate  

✅ **Answer: B**


**9️⃣ What is gradient descent used for?**

A) Data visualization  
B) Minimizing loss  
C) Increasing error  
D) Sorting data  

✅ **Answer: B**


**🔟 Gradient descent updates?**

A) Data  
B) Model parameters  
C) Labels  
D) Files  

✅ **Answer: B**


**1️⃣1️⃣ tf.data.Dataset is used for?**

A) Plotting  
B) Data pipeline creation  
C) Model saving  
D) Loss calculation  

✅ **Answer: B**


**1️⃣2️⃣ What does dataset.shuffle() do?**

A) Sorts data  
B) Randomizes order  
C) Deletes data  
D) Normalizes data  

✅ **Answer: B**


**1️⃣3️⃣ What does dataset.batch() do?**

A) Merges models  
B) Splits data into batches  
C) Removes samples  
D) Sorts features  

✅ **Answer: B**


**1️⃣4️⃣ Why do we use batching?**

A) Slower training  
B) Faster training and memory efficiency  
C) Increase loss  
D) Random output  

✅ **Answer: B**


**1️⃣5️⃣ tf.GradientTape is used for?**

A) Visualization  
B) Automatic differentiation  
C) Data cleaning  
D) Plotting graphs  

✅ **Answer: B**


**1️⃣6️⃣ What does GradientTape compute?**

A) Loss  
B) Gradients  
C) Accuracy  
D) Predictions  

✅ **Answer: B**


**1️⃣7️⃣ tf.Module is used to?**

A) Store datasets  
B) Build custom models  
C) Plot graphs  
D) Load CSV files  

✅ **Answer: B**


**1️⃣8️⃣ In linear regression, the prediction is?**

A) Sum of random values  
B) Weighted sum of inputs  
C) Image classification  
D) Clustering result  

✅ **Answer: B**


**1️⃣9️⃣ What does assign_sub() do?**

A) Adds value  
B) Subtracts value  
C) Multiplies value  
D) Resets variable  

✅ **Answer: B**


**2️⃣0️⃣ assign_sub() is used for?**

A) Forward pass  
B) Updating weights  
C) Data loading  
D) Visualization  

✅ **Answer: B**


**2️⃣1️⃣ Training loss represents?**

A) Model accuracy  
B) Error on training data  
C) Test speed  
D) Dataset size  

✅ **Answer: B**


**2️⃣2️⃣ Test loss measures?**

A) Performance on unseen data  
B) Training speed  
C) Memory usage  
D) Gradient size  

✅ **Answer: A**


**2️⃣3️⃣ One training iteration includes?**

A) Only prediction  
B) Forward + loss + backward + update  
C) Only plotting  
D) Only loading data  

✅ **Answer: B**


**2️⃣4️⃣ Gradient descent tries to?**

A) Increase loss  
B) Minimize loss  
C) Randomize weights  
D) Delete model  

✅ **Answer: B**


**2️⃣5️⃣ Synthetic data means?**

A) Real-world data  
B) Generated artificial data  
C) Missing data  
D) Image data  

✅ **Answer: B**


**2️⃣6️⃣ tf.stack() is used to?**

A) Remove tensors  
B) Combine tensors  
C) Sort tensors  
D) Normalize tensors  

✅ **Answer: B**


**2️⃣7️⃣ tf.squeeze() removes?**

A) Large values  
B) Dimensions of size 1  
C) Data samples  
D) Loss values  

✅ **Answer: B**


**2️⃣8️⃣ Learning rate controls?**

A) Dataset size  
B) Step size in gradient descent  
C) Number of features  
D) Model architecture  

✅ **Answer: B**


**2️⃣9️⃣ If learning rate is too high?**

A) Fast convergence always  
B) Model may diverge  
C) No change  
D) Perfect training  

✅ **Answer: B**


**3️⃣0️⃣ If learning rate is too low?**

A) Very fast training  
B) Slow convergence  
C) No training  
D) Random loss  

✅ **Answer: B**


**3️⃣1️⃣ What is the purpose of the forward pass?**

A) Update weights  
B) Make predictions  
C) Compute gradients  
D) Shuffle data  

✅ **Answer: B**


**3️⃣2️⃣ Backpropagation is used to?**

A) Compute gradients  
B) Load data  
C) Plot graphs  
D) Normalize inputs  

✅ **Answer: A**


**3️⃣3️⃣ Linear regression is a?**

A) Classification model  
B) Regression model  
C) Clustering model  
D) Reinforcement model  

✅ **Answer: B**


**3️⃣4️⃣ Which TensorFlow feature improves performance?**

A) tf.constant  
B) tf.function  
C) tf.print  
D) tf.list  

✅ **Answer: B**


**3️⃣5️⃣ Dataset.from_tensor_slices is used to?**

A) Create dataset from tensors  
B) Plot graphs  
C) Train model directly  
D) Save model  

✅ **Answer: A**


**3️⃣6️⃣ The goal of training is to?**

A) Increase loss  
B) Minimize loss  
C) Randomize outputs  
D) Freeze model  

✅ **Answer: B**


**3️⃣7️⃣ Which function calculates gradients automatically?**

A) tf.constant  
B) tf.GradientTape  
C) tf.stack  
D) tf.data  

✅ **Answer: B**


**3️⃣8️⃣ Model parameters in this lab are?**

A) Images  
B) Weights and bias  
C) Labels only  
D) Dataset  

✅ **Answer: B**


**3️⃣9️⃣ What indicates good training?**

A) Increasing loss  
B) Decreasing loss  
C) Random loss  
D) Constant loss  

✅ **Answer: B**


**4️⃣0️⃣ The main output of a linear regression model is?**

A) Class label  
B) Continuous value  
C) Image  
D) Text  

✅ **Answer: B**


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

**1️⃣ What is the basic unit of a neural network?**

A) Dataset  
B) Neuron  
C) Matrix  
D) Loss function  

✅ **Answer: B**


**2️⃣ A neuron computes?**

A) y = x²  
B) y = wx + b  
C) y = log(x)  
D) y = sin(x)  

✅ **Answer: B**


**3️⃣ What activation function is commonly used in this lab?**

A) ReLU  
B) Sigmoid  
C) Tanh  
D) Softmax  

✅ **Answer: B**


**4️⃣ Sigmoid output range is?**

A) (-∞, ∞)  
B) (0, 1)  
C) (-1, 1)  
D) (0, 10)  

✅ **Answer: B**


**5️⃣ Sigmoid formula is?**

A) 1 / (1 + eˣ)  
B) 1 / (1 + e⁻ˣ)  
C) eˣ  
D) x²  

✅ **Answer: B**


**6️⃣ What is the main use of sigmoid?**

A) Sorting  
B) Probability output  
C) Image resizing  
D) File handling  

✅ **Answer: B**


**7️⃣ Which logic gate can a single neuron learn easily?**

A) XOR  
B) AND  
C) Complex circuits only  
D) Memory gates  

✅ **Answer: B**


**8️⃣ Which gate is NOT linearly separable?**

A) AND  
B) OR  
C) XOR  
D) NAND  

✅ **Answer: C**


**9️⃣ Why can't a single neuron learn XOR?**

A) Too slow  
B) Not enough data  
C) Not linearly separable  
D) Too many parameters  

✅ **Answer: C**


**🔟 XOR requires?**

A) One neuron  
B) Multiple layers  
C) No training  
D) No activation  

✅ **Answer: B**


**1️⃣1️⃣ A neural network is made of?**

A) Files  
B) Neurons  
C) Tables  
D) Images  

✅ **Answer: B**


**1️⃣2️⃣ What does bias do?**

A) Controls output shift  
B) Removes data  
C) Sorts inputs  
D) Normalizes images  

✅ **Answer: A**


**1️⃣3️⃣ Which loss function is used?**

A) Cross entropy  
B) MSE  
C) Hinge loss  
D) KL divergence  

✅ **Answer: B**


**1️⃣4️⃣ MSE stands for?**

A) Mean Signal Error  
B) Mean Squared Error  
C) Model Score Estimation  
D) Matrix Sum Error  

✅ **Answer: B**


**1️⃣5️⃣ MSE measures?**

A) Accuracy  
B) Difference between prediction and target  
C) Data size  
D) Gradient value  

✅ **Answer: B**


**1️⃣6️⃣ What is used to compute gradients?**

A) tf.constant  
B) tf.GradientTape  
C) tf.stack  
D) tf.sigmoid  

✅ **Answer: B**


**1️⃣7️⃣ Gradient descent is used to?**

A) Increase loss  
B) Minimize loss  
C) Shuffle data  
D) Store variables  

✅ **Answer: B**


**1️⃣8️⃣ Weight update rule is?**

A) w = w + gradient  
B) w = w - learning_rate × gradient  
C) w = random  
D) w = 0  

✅ **Answer: B**


**1️⃣9️⃣ A neuron output after sigmoid is?**

A) Always 0  
B) Always 1  
C) Between 0 and 1  
D) Between -1 and 1  

✅ **Answer: C**


**2️⃣0️⃣ AND gate outputs 1 when?**

A) Any input is 1  
B) Both inputs are 1  
C) Inputs are 0  
D) Inputs differ  

✅ **Answer: B**


**2️⃣1️⃣ OR gate outputs 1 when?**

A) Both inputs are 0  
B) At least one input is 1  
C) Both inputs are 1  
D) Inputs are negative  

✅ **Answer: B**


**2️⃣2️⃣ XOR outputs 1 when?**

A) Inputs are equal  
B) Inputs differ  
C) Inputs are both 0  
D) Inputs are both 1  

✅ **Answer: B**


**2️⃣3️⃣ A single neuron is also called?**

A) Deep model  
B) Perceptron  
C) CNN  
D) RNN  

✅ **Answer: B**


**2️⃣4️⃣ Activation function introduces?**

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise only  

✅ **Answer: B**


**2️⃣5️⃣ Without activation function, model becomes?**

A) Non-linear  
B) Linear  
C) Random  
D) Deep  

✅ **Answer: B**


**2️⃣6️⃣ Multi-layer networks are needed for?**

A) Simple tasks  
B) Non-linear problems  
C) Sorting only  
D) File reading  

✅ **Answer: B**


**2️⃣7️⃣ XOR problem shows?**

A) Overfitting  
B) Linear limitation of single neurons  
C) Data imbalance  
D) Memory issue  

✅ **Answer: B**


**2️⃣8️⃣ tf.equal() is used for?**

A) Addition  
B) Comparison  
C) Loss calculation  
D) Training  

✅ **Answer: B**


**2️⃣9️⃣ tf.round() does?**

A) Floors values  
B) Rounds values  
C) Normalizes data  
D) Sorts tensors  

✅ **Answer: B**


**3️⃣0️⃣ tf.cast() is used to?**

A) Train model  
B) Change data type  
C) Plot graphs  
D) Compute loss  

✅ **Answer: B**


**3️⃣1️⃣ A neuron is inspired by?**

A) CPU  
B) Human brain  
C) Database  
D) Compiler  

✅ **Answer: B**


**3️⃣2️⃣ Input of neuron is multiplied by?**

A) Bias  
B) Weight  
C) Loss  
D) Gradient  

✅ **Answer: B**


**3️⃣3️⃣ Output of neuron before activation is?**

A) Random value  
B) Linear combination  
C) Image  
D) Label  

✅ **Answer: B**


**3️⃣4️⃣ Training improves?**

A) Loss increases  
B) Loss decreases  
C) Data size  
D) Model randomness  

✅ **Answer: B**


**3️⃣5️⃣ Gradient tells?**

A) Direction of improvement  
B) Data size  
C) Model type  
D) Output label  

✅ **Answer: A**


**3️⃣6️⃣ XOR requires?**

A) Linear model  
B) Non-linear model  
C) No model  
D) Random model  

✅ **Answer: B**


**3️⃣7️⃣ Neural networks solve XOR using?**

A) One layer  
B) Multiple layers  
C) No activation  
D) Sorting  

✅ **Answer: B**


**3️⃣8️⃣ Sigmoid is?**

A) Linear  
B) Non-linear  
C) Constant  
D) Random  

✅ **Answer: B**


**3️⃣9️⃣ Output of sigmoid is interpreted as?**

A) Loss  
B) Probability  
C) Gradient  
D) Weight  

✅ **Answer: B**


**4️⃣0️⃣ Main idea of this lab is?**

A) File handling  
B) Logic gates using neurons  
C) Image processing  
D) Database design  

✅ **Answer: B**


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

**1️⃣ MNIST dataset is used for?**

A) Text classification  
B) Image segmentation  
C) Handwritten digit classification  
D) Speech recognition  

✅ **Answer: C**


**2️⃣ Each MNIST image size is?**

A) 16×16  
B) 28×28  
C) 32×32  
D) 64×64  

✅ **Answer: B**


**3️⃣ MNIST labels represent?**

A) Animals  
B) Digits 0–9  
C) Letters A–Z  
D) Colors  

✅ **Answer: B**


**4️⃣ tfds.load() is used to?**

A) Build models  
B) Load datasets  
C) Plot graphs  
D) Train optimizer  

✅ **Answer: B**


**5️⃣ as_supervised=True returns?**

A) Images only  
B) Labels only  
C) (input, label) pairs  
D) Random tensors  

✅ **Answer: C**


**6️⃣ Main purpose of flatten layer?**

A) Increase image size  
B) Convert 2D to 1D  
C) Normalize data  
D) Add noise  

✅ **Answer: B**


**7️⃣ Flatten input 28×28 becomes?**

A) 28  
B) 784  
C) 1024  
D) 256  

✅ **Answer: B**


**8️⃣ A Dense layer computes?**

A) x²  
B) Wx + b  
C) log(x)  
D) sin(x)  

✅ **Answer: B**


**9️⃣ Activation function adds?**

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise  

✅ **Answer: B**


**🔟 ReLU function is?**

A) max(0, x)  
B) min(0, x)  
C) x²  
D) log(x)  

✅ **Answer: A**


**1️⃣1️⃣ Softmax outputs?**

A) Binary values  
B) Probabilities  
C) Gradients  
D) Loss values  

✅ **Answer: B**


**1️⃣2️⃣ Softmax ensures?**

A) Negative values  
B) Sum = 1  
C) Sum = 0  
D) Random output  

✅ **Answer: B**


**1️⃣3️⃣ Loss function used in this lab?**

A) MSE  
B) Cross entropy  
C) Hinge loss  
D) MAE  

✅ **Answer: B**


**1️⃣4️⃣ Sparse categorical cross entropy is used for?**

A) Regression  
B) Integer labels  
C) Images only  
D) Clustering  

✅ **Answer: B**


**1️⃣5️⃣ Output layer activation is?**

A) ReLU  
B) Softmax  
C) Sigmoid  
D) Linear  

✅ **Answer: B**


**1️⃣6️⃣ Optimizer used in lab?**

A) SGD only  
B) Adam  
C) RMSProp only  
D) None  

✅ **Answer: B**


**1️⃣7️⃣ Adam optimizer is?**

A) Random  
B) Adaptive learning rate optimizer  
C) Sorting algorithm  
D) Data loader  

✅ **Answer: B**


**1️⃣8️⃣ Training goal is to?**

A) Increase loss  
B) Minimize loss  
C) Randomize weights  
D) Reduce dataset  

✅ **Answer: B**


**1️⃣9️⃣ GradientTape is used for?**

A) Data loading  
B) Automatic differentiation  
C) Image resizing  
D) Sorting  

✅ **Answer: B**


**2️⃣0️⃣ Gradient descent updates?**

A) Data  
B) Weights  
C) Labels  
D) Images  

✅ **Answer: B**


**2️⃣1️⃣ MLP stands for?**

A) Multi Layer Perceptron  
B) Matrix Linear Process  
C) Multi Label Prediction  
D) Memory Learning Program  

✅ **Answer: A**


**2️⃣2️⃣ Hidden layers typically use?**

A) Softmax  
B) ReLU  
C) Argmax  
D) Sigmoid only  

✅ **Answer: B**


**2️⃣3️⃣ Output layer predicts?**

A) Images  
B) Classes  
C) Gradients  
D) Weights  

✅ **Answer: B**


**2️⃣4️⃣ argmax returns?**

A) Minimum value  
B) Index of max value  
C) Mean value  
D) Loss  

✅ **Answer: B**


**2️⃣5️⃣ argmax(axis=1) means?**

A) Across rows  
B) Across columns  
C) Entire tensor  
D) Random axis  

✅ **Answer: A**


**2️⃣6️⃣ Dataset pipeline includes?**

A) Train only  
B) Shuffle and batch  
C) Plot only  
D) Normalize only  

✅ **Answer: B**


**2️⃣7️⃣ batch() function is used to?**

A) Resize images  
B) Group data  
C) Delete data  
D) Sort labels  

✅ **Answer: B**


**2️⃣8️⃣ shuffle() function?**

A) Sorts data  
B) Randomizes order  
C) Deletes samples  
D) Normalizes data  

✅ **Answer: B**


**2️⃣9️⃣ Model input is?**

A) 1D vector  
B) 2D image  
C) Text  
D) Audio only  

✅ **Answer: B**


**3️⃣0️⃣ First step in model?**

A) Softmax  
B) Flatten  
C) Loss  
D) Optimization  

✅ **Answer: B**


**3️⃣1️⃣ Neural network learns by?**

A) Hard coding  
B) Gradient updates  
C) Random guessing  
D) Sorting data  

✅ **Answer: B**


**3️⃣2️⃣ Loss measures?**

A) Accuracy  
B) Error  
C) Speed  
D) Memory  

✅ **Answer: B**


**3️⃣3️⃣ Backpropagation uses?**

A) argmax  
B) Gradients  
C) Sorting  
D) Convolution  

✅ **Answer: B**


**3️⃣4️⃣ Weights represent?**

A) Data size  
B) Importance of inputs  
C) Labels  
D) Loss  

✅ **Answer: B**


**3️⃣5️⃣ Bias helps?**

A) Shift activation  
B) Normalize data  
C) Sort data  
D) Delete noise  

✅ **Answer: A**


**3️⃣6️⃣ Neural network output is?**

A) Random  
B) Prediction  
C) Dataset  
D) Image  

✅ **Answer: B**


**3️⃣7️⃣ MNIST classification has?**

A) 2 classes  
B) 10 classes  
C) 100 classes  
D) 50 classes  

✅ **Answer: B**


**3️⃣8️⃣ One-hot encoding is replaced by?**

A) Sparse labels  
B) Text labels  
C) Images  
D) Noise  

✅ **Answer: A**


**3️⃣9️⃣ Training improves?**

A) Loss increases  
B) Accuracy increases  
C) Data disappears  
D) Model resets  

✅ **Answer: B**


**4️⃣0️⃣ Main goal of Lab 5?**

A) Image generation  
B) Digit classification  
C) Speech recognition  
D) Object detection  

✅ **Answer: B**


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

**1️⃣ Keras Sequential model is used for?**

A) Complex graph models  
B) Linear stack of layers  
C) Databases  
D) Image storage  

✅ **Answer: B**


**2️⃣ Sequential model means?**

A) Random structure  
B) Layers connected one after another  
C) Parallel networks  
D) No layers  

✅ **Answer: B**


**3️⃣ Best use case for Sequential model?**

A) CNN with branches  
B) Simple feedforward networks  
C) Graph models  
D) Reinforcement learning  

✅ **Answer: B**


**4️⃣ First layer in Sequential model defines?**

A) Loss  
B) Input shape  
C) Optimizer  
D) Output  

✅ **Answer: B**


**5️⃣ keras.Input() is used to?**

A) Train model  
B) Define input shape  
C) Normalize data  
D) Compute loss  

✅ **Answer: B**


**6️⃣ Dense layer is?**

A) Convolution layer  
B) Fully connected layer  
C) Pooling layer  
D) Dropout layer  

✅ **Answer: B**


**7️⃣ Dense layer computes?**

A) x²  
B) Wx + b  
C) log(x)  
D) Sorting  

✅ **Answer: B**


**8️⃣ Activation function adds?**

A) Linearity  
B) Non-linearity  
C) Sorting  
D) Noise only  

✅ **Answer: B**


**9️⃣ ReLU stands for?**

A) Random Linear Unit  
B) Rectified Linear Unit  
C) Regression Linear Unit  
D) Reduced Linear Unit  

✅ **Answer: B**


**🔟 ReLU function is?**

A) max(0, x)  
B) min(0, x)  
C) x²  
D) log(x)  

✅ **Answer: A**


**1️⃣1️⃣ Softmax output represents?**

A) Loss  
B) Probabilities  
C) Gradients  
D) Images  

✅ **Answer: B**


**1️⃣2️⃣ Softmax ensures?**

A) Negative outputs  
B) Sum of probabilities = 1  
C) Random output  
D) Zero output  

✅ **Answer: B**


**1️⃣3️⃣ MNIST dataset contains?**

A) Images of animals  
B) Handwritten digits  
C) Text documents  
D) Audio signals  

✅ **Answer: B**


**1️⃣4️⃣ MNIST classes are?**

A) 5  
B) 10  
C) 100  
D) 2  

✅ **Answer: B**


**1️⃣5️⃣ Flatten layer does?**

A) Expands image  
B) Converts 2D to 1D  
C) Adds noise  
D) Normalizes labels  

✅ **Answer: B**


**1️⃣6️⃣ MNIST image size is?**

A) 32×32  
B) 28×28  
C) 64×64  
D) 16×16  

✅ **Answer: B**


**1️⃣7️⃣ model.summary() shows?**

A) Loss values  
B) Model architecture  
C) Dataset info  
D) Predictions  

✅ **Answer: B**


**1️⃣8️⃣ model.compile() is used for?**

A) Building dataset  
B) Configuring training  
C) Plotting graphs  
D) Saving model  

✅ **Answer: B**


**1️⃣9️⃣ Loss function used in MNIST?**

A) MSE  
B) Cross entropy  
C) MAE  
D) Hinge loss  

✅ **Answer: B**


**2️⃣0️⃣ Sparse categorical crossentropy is used for?**

A) Regression  
B) Integer labels  
C) Images only  
D) Clustering  

✅ **Answer: B**


**2️⃣1️⃣ Optimizer commonly used?**

A) SGD only  
B) Adam  
C) Random optimizer  
D) None  

✅ **Answer: B**


**2️⃣2️⃣ model.fit() is used for?**

A) Testing model  
B) Training model  
C) Saving model  
D) Loading dataset  

✅ **Answer: B**


**2️⃣3️⃣ model.evaluate() returns?**

A) Dataset  
B) Loss and accuracy  
C) Weights  
D) Gradients  

✅ **Answer: B**


**2️⃣4️⃣ model.predict() gives?**

A) Labels only  
B) Probabilities  
C) Loss  
D) Gradients  

✅ **Answer: B**


**2️⃣5️⃣ argmax is used to?**

A) Find minimum  
B) Find class index  
C) Normalize data  
D) Train model  

✅ **Answer: B**


**2️⃣6️⃣ argmax returns?**

A) Value  
B) Index  
C) Loss  
D) Image  

✅ **Answer: B**


**2️⃣7️⃣ Input to model must be?**

A) String  
B) Tensor  
C) File  
D) Image only  

✅ **Answer: B**


**2️⃣8️⃣ Hidden layers usually use?**

A) Softmax  
B) ReLU  
C) Argmax  
D) Loss  

✅ **Answer: B**


**2️⃣9️⃣ Output layer for MNIST uses?**

A) ReLU  
B) Softmax  
C) Sigmoid  
D) Linear  

✅ **Answer: B**


**3️⃣0️⃣ Training goal is?**

A) Increase loss  
B) Minimize loss  
C) Shuffle data  
D) Delete data  

✅ **Answer: B**


**3️⃣1️⃣ Backpropagation uses?**

A) Gradients  
B) Sorting  
C) Random values  
D) Images  

✅ **Answer: A**


**3️⃣2️⃣ Weights represent?**

A) Output labels  
B) Importance of features  
C) Loss values  
D) Images  

✅ **Answer: B**


**3️⃣3️⃣ Bias helps?**

A) Shift activation  
B) Normalize data  
C) Delete data  
D) Sort inputs  

✅ **Answer: A**


**3️⃣4️⃣ Learning improves?**

A) Accuracy  
B) Data size  
C) Image resolution  
D) Memory  

✅ **Answer: A**


**3️⃣5️⃣ Each MNIST prediction gives?**

A) One value  
B) 10 probabilities  
C) 28 values  
D) Random output  

✅ **Answer: B**


**3️⃣6️⃣ Sequential model is?**

A) Graph model  
B) Stack of layers  
C) Tree model  
D) Database  

✅ **Answer: B**


**3️⃣7️⃣ Training uses?**

A) Forward only  
B) Forward + backward pass  
C) Sorting  
D) Clustering  

✅ **Answer: B**


**3️⃣8️⃣ Loss measures?**

A) Accuracy  
B) Error  
C) Speed  
D) Memory  

✅ **Answer: B**


**3️⃣9️⃣ MNIST classification type?**

A) Regression  
B) Multi-class classification  
C) Clustering  
D) Detection  

✅ **Answer: B**


**4️⃣0️⃣ Main goal of Lab 6?**

A) Image generation  
B) Build MNIST classifier using Keras  
C) Object detection  
D) Text classification  

✅ **Answer: B**


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

**1️⃣ CIFAR-10 dataset contains how many classes?**

A) 5  
B) 10  
C) 100  
D) 20  

✅ **Answer: B**


**2️⃣ Each CIFAR-10 image size is?**

A) 28×28  
B) 64×64  
C) 32×32  
D) 128×128  

✅ **Answer: C**


**3️⃣ CIFAR-10 images are?**

A) Grayscale  
B) RGB color images  
C) Binary images  
D) Text images  

✅ **Answer: B**


**4️⃣ CIFAR-10 is mainly used for?**

A) NLP  
B) Image classification  
C) Speech recognition  
D) Reinforcement learning  

✅ **Answer: B**


**5️⃣ CIFAR-10 dataset has approximately?**

A) 10,000 images  
B) 50,000 images  
C) 60,000 images  
D) 1 million images  

✅ **Answer: C**


**6️⃣ Which of the following is NOT a CIFAR-10 class?**

A) Cat  
B) Dog  
C) Car  
D) Lion  

✅ **Answer: D**


**7️⃣ CIFAR-10 images are typically used in?**

A) CNN models  
B) RNN models  
C) LSTM models  
D) GAN only  

✅ **Answer: A**


**8️⃣ Input shape of CIFAR-10 images is usually?**

A) (28, 28, 1)  
B) (32, 32, 3)  
C) (64, 64, 1)  
D) (10, 10, 3)  

✅ **Answer: B**


**9️⃣ Conv2D layer is used for?**

A) Text processing  
B) Feature extraction from images  
C) Sorting data  
D) Normalization only  

✅ **Answer: B**


**🔟 MaxPooling is used to?**

A) Increase image size  
B) Reduce spatial dimensions  
C) Add noise  
D) Convert labels  

✅ **Answer: B**


**1️⃣1️⃣ Flatten layer converts?**

A) 1D → 2D  
B) 2D → 1D  
C) RGB → grayscale  
D) Labels → images  

✅ **Answer: B**


**1️⃣2️⃣ CNN is especially good for?**

A) Tabular data  
B) Image data  
C) Audio only  
D) Text only  

✅ **Answer: B**


**1️⃣3️⃣ Conv2D learns?**

A) Weights manually  
B) Filters automatically  
C) Labels  
D) Loss only  

✅ **Answer: B**


**1️⃣4️⃣ Pooling layer helps in?**

A) Overfitting  
B) Feature reduction  
C) Increasing noise  
D) Label encoding  

✅ **Answer: B**


**1️⃣5️⃣ MaxPooling selects?**

A) Average value  
B) Minimum value  
C) Maximum value  
D) Random value  

✅ **Answer: C**


**1️⃣6️⃣ CNN first learns?**

A) Shapes  
B) Edges  
C) Labels  
D) Loss  

✅ **Answer: B**


**1️⃣7️⃣ Flatten layer is required before?**

A) Conv layer  
B) Dense layer  
C) Pooling layer  
D) Input layer  

✅ **Answer: B**


**1️⃣8️⃣ CNN stands for?**

A) Central Neural Network  
B) Convolutional Neural Network  
C) Core Neural Node  
D) Complex Numeric Network  

✅ **Answer: B**


**1️⃣9️⃣ Keras Sequential model is?**

A) Graph-based model  
B) Linear stack of layers  
C) Tree model  
D) Rule-based system  

✅ **Answer: B**


**2️⃣0️⃣ First layer in CNN must define?**

A) Loss  
B) Input shape  
C) Optimizer  
D) Labels  

✅ **Answer: B**


**2️⃣1️⃣ Softmax activation is used for?**

A) Regression  
B) Classification probabilities  
C) Feature extraction  
D) Pooling  

✅ **Answer: B**


**2️⃣2️⃣ ReLU activation outputs?**

A) Negative values only  
B) 0 or positive values  
C) Only 1 and 0  
D) Probabilities  

✅ **Answer: B**


**2️⃣3️⃣ model.compile() is used to?**

A) Build model layers  
B) Train model  
C) Configure training  
D) Predict output  

✅ **Answer: C**


**2️⃣4️⃣ model.fit() is used to?**

A) Train model  
B) Test model  
C) Save model  
D) Compile model  

✅ **Answer: A**


**2️⃣5️⃣ model.evaluate() is used to?**

A) Train model  
B) Test model  
C) Build model  
D) Predict labels  

✅ **Answer: B**


**2️⃣6️⃣ model.predict() is used for?**

A) Training  
B) Evaluation  
C) Inference  
D) Compilation  

✅ **Answer: C**


**2️⃣7️⃣ Loss function in CIFAR-10 is usually?**

A) MSE  
B) MAE  
C) Cross-entropy  
D) Hinge loss  

✅ **Answer: C**


**2️⃣8️⃣ Optimizer commonly used is?**

A) SGD only  
B) Adam  
C) Random  
D) None  

✅ **Answer: B**


**2️⃣9️⃣ Data augmentation helps to?**

A) Increase dataset size artificially  
B) Reduce model size  
C) Remove labels  
D) Remove images  

✅ **Answer: A**


**3️⃣0️⃣ Which is NOT an augmentation technique?**

A) Rotation  
B) Flipping  
C) Normalization  
D) Shuffling labels  

✅ **Answer: D**


**3️⃣1️⃣ ImageDataGenerator is used for?**

A) Model saving  
B) Real-time augmentation  
C) Loss calculation  
D) Feature extraction  

✅ **Answer: B**


**3️⃣2️⃣ Horizontal flip means?**

A) Rotate image  
B) Flip left-right  
C) Flip up-down  
D) Crop image  

✅ **Answer: B**


**3️⃣3️⃣ Augmentation helps prevent?**

A) Underfitting  
B) Overfitting  
C) Training  
D) Evaluation  

✅ **Answer: B**


**3️⃣4️⃣ Zooming is used to?**

A) Remove images  
B) Change image scale  
C) Convert labels  
D) Normalize data  

✅ **Answer: B**


**3️⃣5️⃣ CNN learns features in order?**

A) Random → shapes → edges  
B) Edges → textures → shapes  
C) Shapes → edges → labels  
D) Labels → edges → textures  

✅ **Answer: B**


**3️⃣6️⃣ CNN is inspired by?**

A) Human brain visual cortex  
B) Database systems  
C) Sorting algorithms  
D) Decision trees  

✅ **Answer: A**


**3️⃣7️⃣ CIFAR-10 output layer has?**

A) 2 neurons  
B) 5 neurons  
C) 10 neurons  
D) 100 neurons  

✅ **Answer: C**


**3️⃣8️⃣ One-hot encoding converts?**

A) Images → numbers  
B) Labels → vectors  
C) Pixels → labels  
D) Loss → accuracy  

✅ **Answer: B**


**3️⃣9️⃣ CNN reduces parameters using?**

A) Fully connected layers  
B) Weight sharing  
C) Random weights  
D) Label smoothing  

✅ **Answer: B**


**4️⃣0️⃣ Main advantage of CNN over Dense network is?**

A) Faster text processing  
B) Better image feature learning  
C) Lower memory always  
D) No training needed  

✅ **Answer: B**


---
---

---
---

# 🟣 Lab 8: Transfer Learning with DenseNet121 (CIFAR-10)

> Instead of training from scratch, **reuse a model pre-trained on ImageNet** and adapt it to CIFAR-10 — saving time, compute, and achieving higher accuracy.

![Transfer Learning](https://img.shields.io/badge/Technique-Transfer%20Learning-purple?style=for-the-badge)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-blueviolet?style=for-the-badge)
![Keras](https://img.shields.io/badge/API-Keras-red?style=for-the-badge&logo=keras&logoColor=white)

---

## 🧠 Lab Summary

| Component | Detail |
|-----------|--------|
| 🏗️ Base Model | DenseNet121 (pre-trained on ImageNet) |
| 📊 Dataset | CIFAR-10 (32×32 RGB, 10 classes) |
| 🎯 Strategies | Feature Extraction, Fine-Tuning |
| 📉 Loss | Categorical Cross-Entropy |
| ⚙️ Optimizer | Adam |

---

## 🔄 What is Transfer Learning?

Transfer Learning reuses a model already trained on a large dataset (ImageNet) and adapts it to a new task.

```
Without Transfer Learning:  Train from random weights  →  Slow, needs huge data
With Transfer Learning:     Reuse ImageNet weights    →  Fast, fewer data needed
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| ⏱️ Saves time | No need to train deep layers from scratch |
| 💻 Saves compute | Fewer epochs needed |
| 📈 Better accuracy | Pre-trained features are already powerful |

The pre-trained model already knows: edges → textures → shapes → objects.  
We only teach it: **the 10 new CIFAR-10 classes**.

---

## 🏗️ What is DenseNet121?

**DenseNet = Dense Convolutional Network**

Key idea: **every layer connects to all previous layers** (not just the one before).

```
Layer 1 → Layer 2
Layer 1 → Layer 3
Layer 1 → Layer 2 → Layer 3
```

| Advantage | Description |
|-----------|-------------|
| Better feature reuse | Each layer sees all previous outputs |
| Stronger gradient flow | Prevents vanishing gradient |
| Fewer parameters | Efficient use of features |

### Loading DenseNet121

```python
base_model = tf.keras.applications.DenseNet121(
    include_top=False,       # Remove ImageNet classifier (1000 classes)
    weights='imagenet',      # Load pre-trained ImageNet weights
    input_shape=(32, 32, 3)  # CIFAR-10 image size
)
```

| Parameter | Meaning |
|-----------|---------|
| `include_top=False` | Remove original 1000-class head |
| `weights='imagenet'` | Use pre-trained weights |
| `weights=None` | Random initialization (from scratch) |
| `input_shape=(32,32,3)` | CIFAR-10 RGB image dimensions |

---

## 🔒 Feature Extraction Strategy

**Freeze** all pre-trained layers → train only the new classification head.

```python
base_model.trainable = False   # Freeze all DenseNet layers
```

### Adding a Custom Classification Head

```python
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),   # Feature maps → vector
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### What is GlobalAveragePooling2D?

Converts feature maps into a 1D vector by averaging each feature map spatially.

```
Input:   (batch, 4, 4, 8)  →  Feature maps
Output:  (batch, 8)         →  One value per channel
```

| Layer | Purpose |
|-------|---------|
| `GlobalAveragePooling2D` | Reduce 2D maps → 1D vector |
| `Dense(128, relu)` | Learn CIFAR-specific patterns |
| `Dense(10, softmax)` | Output 10 class probabilities |

---

## 🔥 Fine-Tuning

After training the classifier head, **unfreeze some (or all) pre-trained layers** and continue training with a very small learning rate.

```python
# Unfreeze all layers
base_model.trainable = True

# Or unfreeze only later layers
for layer in base_model.layers[:100]:
    layer.trainable = False   # Freeze early layers

for layer in base_model.layers[100:]:
    layer.trainable = True    # Unfreeze later layers

# Use very small learning rate
optimizer = tf.keras.optimizers.Adam(1e-5)
```

**Why small learning rate?**
```
✔ Prevents destroying the pre-trained knowledge
✔ Makes small targeted adjustments
```

### Why freeze early layers?

| Layer Type | Learns | Keep? |
|------------|--------|-------|
| Early CNN layers | General edges & textures | ✅ Keep frozen |
| Later CNN layers | Dataset-specific shapes | 🔥 Fine-tune |
| Classifier head | New class labels | 🔥 Train |

---

## 📊 Three Training Strategies

| Strategy | What Happens | Speed | Accuracy |
|----------|-------------|-------|----------|
| **Feature Extraction** | Freeze base, train head only | ⚡ Fast | Good |
| **Fine-Tuning** | Unfreeze some layers, continue training | Medium | Better ✅ |
| **From Scratch** | Train everything randomly | 🐢 Slow | Needs lots of data |

---

## ⚙️ Training Process

```python
# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train (Feature Extraction)
model.fit(train_images, train_labels, epochs=5)

# Evaluate
model.evaluate(test_images, test_labels)
```

---

## 🎯 Key Takeaways — Lab 8

```
✔ Transfer learning = reuse ImageNet knowledge for a new task
✔ include_top=False = remove old 1000-class head
✔ weights='imagenet' = load pre-trained weights
✔ base_model.trainable=False = freeze (feature extraction)
✔ GlobalAveragePooling2D = bridge between base and classifier
✔ Fine-tuning = unfreeze + low learning rate
✔ Early layers = general features (keep frozen)
✔ Later layers = task-specific (fine-tune)
```

---

## ❓ Lab 8 MCQs (40 Questions)

**Q1. What is the main goal of transfer learning?**

A) Delete model weights  
B) Reuse knowledge from a pre-trained model  
C) Reduce image size  
D) Increase dataset labels  

✅ **Answer: B**

---

**Q2. Which pre-trained model was used in this lab?**

A) AlexNet  
B) VGG16  
C) DenseNet121  
D) LeNet  

✅ **Answer: C**

---

**Q3. DenseNet121 is pre-trained on which dataset?**

A) CIFAR-10  
B) MNIST  
C) Fashion-MNIST  
D) ImageNet  

✅ **Answer: D**

---

**Q4. What does include_top=False mean?**

A) Remove convolution layers  
B) Remove pooling layers  
C) Remove the original classifier head  
D) Freeze all layers  

✅ **Answer: C**

---

**Q5. Why do we use include_top=False in transfer learning?**

A) To add a custom classifier  
B) To reduce image size  
C) To remove activations  
D) To normalize data  

✅ **Answer: A**

---

**Q6. What is the purpose of freezing layers?**

A) Increase batch size  
B) Prevent weight updates  
C) Remove gradients  
D) Duplicate layers  

✅ **Answer: B**

---

**Q7. Which attribute freezes a model layer?**

A) layer.freeze=True  
B) layer.fixed=True  
C) layer.trainable=False  
D) layer.stop=True  

✅ **Answer: C**

---

**Q8. Feature extraction means?**

A) Training the whole model from scratch  
B) Freezing the base model and training only the classifier  
C) Removing convolution layers  
D) Using only pooling layers  

✅ **Answer: B**

---

**Q9. Fine-tuning means?**

A) Freezing all layers forever  
B) Training only the output layer  
C) Unfreezing some layers and continuing training  
D) Removing DenseNet layers  

✅ **Answer: C**

---

**Q10. Why is a low learning rate used during fine-tuning?**

A) To prevent destroying learned features  
B) To increase overfitting  
C) To reduce epochs  
D) To remove gradients  

✅ **Answer: A**

---

**Q11. Which dataset was used in this lab?**

A) MNIST  
B) CIFAR-10  
C) COCO  
D) Fashion-MNIST  

✅ **Answer: B**

---

**Q12. CIFAR-10 images have how many color channels?**

A) 1  
B) 2  
C) 3  
D) 4  

✅ **Answer: C**

---

**Q13. What is the image size in CIFAR-10?**

A) 28×28  
B) 64×64  
C) 32×32  
D) 128×128  

✅ **Answer: C**

---

**Q14. Which layer converts feature maps into a vector by averaging spatial dimensions?**

A) Flatten  
B) Conv2D  
C) GlobalAveragePooling2D  
D) MaxPooling2D  

✅ **Answer: C**

---

**Q15. The output shape of GlobalAveragePooling2D keeps?**

A) Width and height only  
B) Channels only  
C) Batch size only  
D) Pixels only  

✅ **Answer: B**

---

**Q16. Which parameter loads ImageNet weights?**

A) weights='imagenet'  
B) dataset='imagenet'  
C) model='imagenet'  
D) train='imagenet'  

✅ **Answer: A**

---

**Q17. What happens if weights=None?**

A) Pre-trained weights are loaded  
B) Random initialization is used  
C) Model freezes automatically  
D) Images are normalized  

✅ **Answer: B**

---

**Q18. Transfer learning is especially useful when?**

A) Dataset is small  
B) Dataset is extremely large  
C) No GPU exists  
D) Using text data only  

✅ **Answer: A**

---

**Q19. Which layer is commonly added after DenseNet base?**

A) Conv3D  
B) GlobalAveragePooling2D  
C) Embedding  
D) LSTM  

✅ **Answer: B**

---

**Q20. Which activation is commonly used for CIFAR-10 output layer?**

A) ReLU  
B) Tanh  
C) Softmax  
D) Sigmoid  

✅ **Answer: C**

---

**Q21. CIFAR-10 contains how many classes?**

A) 5  
B) 8  
C) 10  
D) 100  

✅ **Answer: C**

---

**Q22. What is the purpose of the classifier head?**

A) Feature extraction  
B) Predict final classes  
C) Normalize images  
D) Resize input  

✅ **Answer: B**

---

**Q23. Which statement about DenseNet is true?**

A) Layers are disconnected  
B) Layers are densely connected  
C) Uses no convolutions  
D) Has only one layer  

✅ **Answer: B**

---

**Q24. Fine-tuning all layers may cause?**

A) Faster convergence only  
B) Overfitting if dataset is small  
C) No gradients  
D) No learning  

✅ **Answer: B**

---

**Q25. What does model.trainable=False do?**

A) Deletes weights  
B) Freezes the entire model  
C) Changes activation functions  
D) Increases image resolution  

✅ **Answer: B**

---

**Q26. Which optimizer is commonly used in transfer learning?**

A) Adam  
B) Bubble Sort  
C) DFS  
D) BFS  

✅ **Answer: A**

---

**Q27. During feature extraction, which layers are trained?**

A) All DenseNet layers  
B) Only newly added classifier layers  
C) No layers  
D) Only pooling layers  

✅ **Answer: B**

---

**Q28. Fine-tuning starts after?**

A) Data normalization  
B) Feature extraction training  
C) Removing DenseNet  
D) Deleting classifier head  

✅ **Answer: B**

---

**Q29. Why compare validation accuracy between strategies?**

A) To check model performance  
B) To resize images  
C) To freeze data  
D) To remove labels  

✅ **Answer: A**

---

**Q30. Which layer reduces overfitting by randomly dropping neurons?**

A) Dense  
B) Conv2D  
C) Dropout  
D) Flatten  

✅ **Answer: C**

---

**Q31. What is the main advantage of transfer learning?**

A) Requires no data  
B) Faster training and better accuracy  
C) Removes CNN layers  
D) Prevents gradients  

✅ **Answer: B**

---

**Q32. Which type of learning trains the whole model from scratch?**

A) Feature extraction  
B) Fine-tuning  
C) Random initialization training  
D) Freezing  

✅ **Answer: C**

---

**Q33. In custom freezing, early layers are usually?**

A) Deleted  
B) Frozen  
C) Randomized  
D) Flattened  

✅ **Answer: B**

---

**Q34. Why freeze early layers?**

A) They learn general features  
B) They reduce dataset size  
C) They remove gradients  
D) They classify directly  

✅ **Answer: A**

---

**Q35. Which of the following is NOT a transfer learning strategy?**

A) Feature extraction  
B) Fine-tuning  
C) Training from scratch  
D) K-means clustering  

✅ **Answer: D**

---

**Q36. What does input_shape=(32,32,3) represent?**

A) Width only  
B) Height only  
C) RGB image dimensions  
D) Number of classes  

✅ **Answer: C**

---

**Q37. DenseNet121 belongs to which model type?**

A) Recurrent Neural Network  
B) Convolutional Neural Network  
C) Decision Tree  
D) Transformer only  

✅ **Answer: B**

---

**Q38. What is the output of GlobalAveragePooling2D for shape (1,4,4,8)?**

A) (1,4,4)  
B) (1,8)  
C) (4,4,8)  
D) (8,8)  

✅ **Answer: B**

---

**Q39. Why use pre-trained ImageNet weights?**

A) They already learned useful visual features  
B) They reduce image size  
C) They remove labels  
D) They disable training  

✅ **Answer: A**

---

**Q40. Which statement best describes transfer learning?**

A) Reusing a trained model for a new task  
B) Training without labels  
C) Using only linear regression  
D) Removing neural networks  

✅ **Answer: A**

---
---

# ⚫ Lab 9: LSTM on 20 Newsgroups Dataset

> Use **LSTM** and **Bidirectional LSTM** networks to classify news articles from the **20 Newsgroups** dataset — a complete NLP pipeline from raw text to predictions.

![NLP](https://img.shields.io/badge/Task-NLP%20Classification-black?style=for-the-badge)
![LSTM](https://img.shields.io/badge/Model-LSTM%20%2F%20BiLSTM-darkblue?style=for-the-badge)
![Keras](https://img.shields.io/badge/API-Keras-red?style=for-the-badge&logo=keras&logoColor=white)

---

## 🧠 Lab Summary

| Component | Detail |
|-----------|--------|
| 📰 Dataset | 20 Newsgroups (thousands of text articles) |
| 🎯 Task | Multi-class text classification (20 categories) |
| 🧠 Model | LSTM / Bidirectional LSTM |
| 📉 Loss | Categorical Cross-Entropy |
| ⚙️ Optimizer | Adam |

---

## 📰 20 Newsgroups Dataset

A collection of ~18,000 news articles divided into **20 categories**:

| Category | Examples |
|----------|---------|
| ⚽ Sports | Baseball, hockey |
| 🏛️ Politics | Government, guns |
| 💻 Technology | Hardware, software |
| 🔬 Science | Medicine, space |
| ⛪ Religion | Christianity, atheism |

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train')
# Returns: text articles + integer labels
```

---

## 🔤 Text Preprocessing

Neural networks cannot understand raw text — we must convert words to numbers.

### Step 1 — Tokenization

Convert words → unique integers.

```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
    num_words=5000,       # Keep only top 5000 frequent words
    oov_token="<unk>"    # Replace unknown words with this token
)
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)
# {'hello': 1, 'world': 2, 'keras': 3, ...}
```

### Step 2 — Convert to Sequences

```python
sequences = tokenizer.texts_to_sequences(texts)
# "hello world" → [1, 2]
```

### Step 3 — Padding

All sequences must be the **same length** for batching.

```python
from keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(
    sequences,
    maxlen=100,          # Max sequence length
    padding='post',      # Add zeros AFTER sequence
    truncating='post'    # Cut long sequences from the END
)
# [1, 2, 3] → [1, 2, 3, 0, 0]  (maxlen=5)
```

| Parameter | Options | Effect |
|-----------|---------|--------|
| `padding` | `'pre'` / `'post'` | Zeros before / after |
| `truncating` | `'pre'` / `'post'` | Cut from start / end |
| `maxlen` | integer | Fixed sequence length |

### Step 4 — Label Encoding

```python
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y = lb.fit_transform(labels)
# Sports → [1, 0, 0, ...]   Politics → [0, 1, 0, ...]
```

---

## 📦 Embedding Layer

Transforms integer word IDs into **dense semantic vectors**.

```
word ID: 5  →  [0.2, 0.7, -0.1, 0.5, ...]   (128-dim vector)
```

```python
layers.Embedding(
    input_dim=5000,      # Vocabulary size
    output_dim=128,      # Embedding vector size (each word → 128 values)
    input_length=100     # Sequence length
)
```

| Parameter | Meaning |
|-----------|---------|
| `input_dim` | Vocabulary size |
| `output_dim` | Size of each word vector |
| `input_length` | Length of input sequences |

**Output shape:**
```
Input:   (batch_size, 100)          # word IDs
Output:  (batch_size, 100, 128)     # word vectors
```

---

## 🧠 LSTM & Bidirectional LSTM

### Why LSTM?

Normal RNNs forget long-range context. LSTM fixes this with **memory cells**.

```
"dog bites man"   vs   "man bites dog"
→ Same words, different meaning — LSTM tracks the ORDER
```

| Model | Problem Solved |
|-------|---------------|
| Simple RNN | Basic sequence, forgets long dependencies |
| LSTM | Remembers long-range dependencies |
| Bidirectional LSTM | Reads both left→right AND right→left |

### LSTM Layer

```python
layers.LSTM(
    units=64,                  # Number of memory cells
    return_sequences=True      # Return output at every timestep
                               # False = return only final output
)
```

| `return_sequences` | Output Shape | Use When |
|--------------------|-------------|---------|
| `False` (default) | (batch, units) | Last layer before Dense |
| `True` | (batch, seq_len, units) | Stacking LSTM layers |

### Bidirectional LSTM

Reads the sequence in **both directions** simultaneously:

```
Forward:   I → love → deep → learning  →  state_f
Backward:  learning → deep → love → I  →  state_b
Combined:  [state_f + state_b]  (concat, default)
```

```python
layers.Bidirectional(layers.LSTM(64))
# Output size = 64 × 2 = 128  (forward + backward)
```

---

## 🏗️ Model Architecture

```
Raw Text
    ↓
Tokenizer       → words → integers
    ↓
pad_sequences   → equal-length sequences
    ↓
Embedding       → (batch, 100) → (batch, 100, 128)
    ↓
Bidirectional(LSTM(64))  → (batch, 128)
    ↓
Dense(20, softmax)       → 20 class probabilities
    ↓
Predicted Category
```

```python
model = keras.Sequential([
    layers.Embedding(5000, 128, input_length=100),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(20, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 📊 LSTM vs Bidirectional LSTM

| Model | Reads | Accuracy |
|-------|-------|---------|
| LSTM | Left → Right only | Good |
| Bidirectional LSTM | Both directions | Better 🚀 |

---

## 🎯 Key Takeaways — Lab 9

```
✔ Tokenization = convert words → integers
✔ pad_sequences = make all sequences equal length
✔ oov_token = handle unknown words
✔ Embedding = convert word IDs → dense vectors
✔ LSTM = remembers sequence order (solves vanishing gradient)
✔ Bidirectional = reads text in both directions
✔ return_sequences=True = outputs at every timestep
✔ LabelBinarizer = one-hot encode category labels
✔ Adam + categorical_crossentropy = standard NLP training setup
```

---

## ❓ Lab 9 MCQs (40 Questions)

**Q1. What is the main goal of Lab 9?**

A) Image segmentation  
B) Text classification  
C) Object detection  
D) Regression  

✅ **Answer: B**

---

**Q2. Which dataset is used in this lab?**

A) CIFAR-10  
B) MNIST  
C) 20 Newsgroups  
D) COCO  

✅ **Answer: C**

---

**Q3. Which library provides the 20 Newsgroups dataset?**

A) TensorFlow  
B) NumPy  
C) Scikit-learn  
D) Pandas  

✅ **Answer: C**

---

**Q4. What type of neural network is mainly used in this lab?**

A) CNN  
B) GAN  
C) LSTM  
D) Autoencoder  

✅ **Answer: C**

---

**Q5. What does LSTM stand for?**

A) Long Short-Term Memory  
B) Linear Statistical Tensor Model  
C) Large Sequential Training Model  
D) Long Simple Tensor Machine  

✅ **Answer: A**

---

**Q6. Why are LSTMs useful for text?**

A) They process images  
B) They remember sequence information  
C) They remove labels  
D) They reduce vocabulary  

✅ **Answer: B**

---

**Q7. Which preprocessing step converts text into integers?**

A) Pooling  
B) Padding  
C) Tokenization  
D) Flattening  

✅ **Answer: C**

---

**Q8. Which class is used for tokenization?**

A) Dense()  
B) Tokenizer()  
C) Embedding()  
D) LSTM()  

✅ **Answer: B**

---

**Q9. What does Tokenizer.fit_on_texts() do?**

A) Removes labels  
B) Builds the vocabulary index  
C) Pads sequences  
D) Creates embeddings  

✅ **Answer: B**

---

**Q10. What is the purpose of texts_to_sequences()?**

A) Convert numbers to text  
B) Convert text into integer sequences  
C) Normalize data  
D) Remove stop words  

✅ **Answer: B**

---

**Q11. What does OOV mean in NLP?**

A) Out Of Vocabulary  
B) Output Of Vector  
C) Order Of Value  
D) Object Output Variable  

✅ **Answer: A**

---

**Q12. Which parameter handles unknown words?**

A) padding  
B) input_dim  
C) oov_token  
D) return_sequences  

✅ **Answer: C**

---

**Q13. Why is padding needed?**

A) To normalize labels  
B) To make sequences equal length  
C) To resize images  
D) To freeze layers  

✅ **Answer: B**

---

**Q14. Which function performs sequence padding?**

A) pad_sequences()  
B) fit_on_texts()  
C) Embedding()  
D) LabelBinarizer()  

✅ **Answer: A**

---

**Q15. What does padding='post' mean?**

A) Add zeros before sequence  
B) Add zeros after sequence  
C) Remove zeros  
D) Reverse sequence  

✅ **Answer: B**

---

**Q16. What does truncating='pre' do?**

A) Removes values from beginning  
B) Removes values from end  
C) Adds zeros at beginning  
D) Adds zeros at end  

✅ **Answer: A**

---

**Q17. What is the role of the Embedding layer?**

A) Convert text into images  
B) Convert word IDs into dense vectors  
C) Reduce batch size  
D) Remove vocabulary  

✅ **Answer: B**

---

**Q18. Which parameter specifies vocabulary size in Embedding?**

A) units  
B) input_length  
C) input_dim  
D) output_dim  

✅ **Answer: C**

---

**Q19. What does output_dim represent in Embedding?**

A) Number of classes  
B) Embedding vector size  
C) Batch size  
D) Sequence length  

✅ **Answer: B**

---

**Q20. What does input_length represent?**

A) Number of epochs  
B) Sequence length  
C) Vocabulary size  
D) Batch size  

✅ **Answer: B**

---

**Q21. What is the output shape of an Embedding layer?**

A) (batch_size, embedding_dim)  
B) (batch_size, seq_len, embedding_dim)  
C) (embedding_dim,)  
D) (seq_len,)  

✅ **Answer: B**

---

**Q22. Which layer processes sequential data?**

A) Dense  
B) Conv2D  
C) LSTM  
D) Flatten  

✅ **Answer: C**

---

**Q23. What does the units parameter in LSTM control?**

A) Number of memory cells  
B) Batch size  
C) Sequence length  
D) Vocabulary size  

✅ **Answer: A**

---

**Q24. What does return_sequences=True do?**

A) Returns only last output  
B) Returns all timestep outputs  
C) Removes outputs  
D) Reverses sequence  

✅ **Answer: B**

---

**Q25. What is Bidirectional LSTM?**

A) Two Dense layers  
B) Reads sequences in both directions  
C) Removes embeddings  
D) Performs pooling  

✅ **Answer: B**

---

**Q26. Which wrapper creates a Bidirectional LSTM?**

A) Sequential()  
B) Flatten()  
C) Bidirectional()  
D) Dropout()  

✅ **Answer: C**

---

**Q27. What is the default merge mode in Bidirectional()?**

A) sum  
B) average  
C) concat  
D) multiply  

✅ **Answer: C**

---

**Q28. If an LSTM has 8 units, Bidirectional output size is usually?**

A) 4  
B) 8  
C) 16  
D) 32  

✅ **Answer: C**

---

**Q29. Why are Bidirectional LSTMs often better?**

A) They reduce memory usage  
B) They process context from both directions  
C) They remove padding  
D) They train without labels  

✅ **Answer: B**

---

**Q30. Which encoding method converts labels into one-hot vectors?**

A) Tokenizer()  
B) LabelBinarizer()  
C) pad_sequences()  
D) Embedding()  

✅ **Answer: B**

---

**Q31. Which activation is commonly used in the output layer for classification?**

A) ReLU  
B) Sigmoid  
C) Softmax  
D) Tanh  

✅ **Answer: C**

---

**Q32. Which optimizer is commonly used in this lab?**

A) SGD only  
B) Adam  
C) RMS only  
D) Adagrad only  

✅ **Answer: B**

---

**Q33. What problem does LSTM solve better than simple RNN?**

A) Image resizing  
B) Vanishing gradients  
C) Over-padding  
D) Label encoding  

✅ **Answer: B**

---

**Q34. Which type of data is processed in this lab?**

A) Audio  
B) Video  
C) Text  
D) Tabular  

✅ **Answer: C**

---

**Q35. What happens if sequences are too long?**

A) They may be truncated  
B) They are duplicated  
C) Labels disappear  
D) Vocabulary resets  

✅ **Answer: A**

---

**Q36. Which layer usually comes before LSTM in NLP models?**

A) Conv2D  
B) Embedding  
C) Flatten  
D) Pooling  

✅ **Answer: B**

---

**Q37. What is the purpose of num_words in Tokenizer?**

A) Number of labels  
B) Maximum vocabulary size  
C) Number of epochs  
D) Batch size  

✅ **Answer: B**

---

**Q38. Which of the following is true about embeddings?**

A) They are sparse vectors  
B) They capture semantic meaning  
C) They remove sequences  
D) They freeze training  

✅ **Answer: B**

---

**Q39. Which model generally performs better for text understanding?**

A) Standard LSTM  
B) Bidirectional LSTM  
C) Linear Regression  
D) K-Means  

✅ **Answer: B**

---

**Q40. What is the final goal of the model in this lab?**

A) Predict image pixels  
B) Generate text  
C) Classify news articles into categories  
D) Detect objects in images  

✅ **Answer: C**

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
| `GlobalAveragePooling2D` | Keras | Feature maps → 1D vector |
| Cross-entropy | Loss Function | Classification error measure |
| Adam | Optimizer | Adaptive learning rate |
| Data Augmentation | Technique | Prevent overfitting |
| `DenseNet121` | Keras Applications | Pre-trained CNN (ImageNet) |
| `include_top=False` | Transfer Learning | Remove original classifier |
| Fine-Tuning | Transfer Learning | Unfreeze + low LR training |
| `Tokenizer` | Keras / NLP | Convert words → integers |
| `pad_sequences` | Keras / NLP | Equalize sequence lengths |
| `Embedding` | Keras / NLP | Word IDs → dense vectors |
| `LSTM` | Keras / NLP | Sequential memory network |
| `Bidirectional` | Keras / NLP | Forward + backward LSTM |
| `LabelBinarizer` | Scikit-learn | One-hot encode labels |
| `GroupBy` | Pandas | Grouping rows by shared value |
| Histogram | Matplotlib | Data distribution plot |

---

## 🌍 Why These Labs Matter

These labs build the **complete Deep Learning pipeline** from foundations to NLP:

| Lab | Topic | Builds Toward |
|-----|-------|---------------|
| Lab 1 | NumPy / Pandas / Matplotlib | Data handling & visualization |
| Lab 2 | TensorFlow Tensors & Variables | Model architecture basics |
| Lab 3 | Linear Regression | Optimization & custom training |
| Lab 4 | Neurons & Logic Gates | Perceptron & activation functions |
| Lab 5 | MNIST MLP (custom loop) | Full neural network pipeline |
| Lab 6 | Keras Sequential MNIST | High-level model building |
| Lab 7 | CNN CIFAR-10 | Image classification & CNNs |
| Lab 8 | Transfer Learning DenseNet121 | Pre-trained models & fine-tuning |
| Lab 9 | LSTM 20 Newsgroups | NLP text classification pipeline |

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


