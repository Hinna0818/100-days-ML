# Machine Learning and Deep Learning
## Introduction
  
## 1. linear regression
### 1. 简单线性回归模型

考虑一个简单线性回归模型：

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$


### 2. 矩阵表示

对于多个观测值，模型可以表示为矩阵形式：

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

对于简单线性回归（一个自变量），设计矩阵如下：

$$
\mathbf{X} =
\begin{bmatrix}
1 & X_1 \\
1 & X_2 \\
\vdots & \vdots \\
1 & X_n \\
\end{bmatrix}
$$

### 3. 最小二乘法（Ordinary Least Squares, OLS）

目标是最小化残差平方和（RSS）：

$$
RSS = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

### 4. 正规方程（Normal Equation）

通过对 RSS 关于回归系数求偏导并令其为零，可以得到正规方程：

$$
\frac{\partial RSS}{\partial \beta} = -2\mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = 0
$$

整理得：

$$
\mathbf{X}^\top \mathbf{Y} = \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}
$$

解得回归系数：

$$
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}
$$

<mark>具体的python函数如下：  
```{python}
import numpy as np
x = np.array([xxx])
y = np.array([xxx]).reshape(-1,1) ## y是一个列向量
x_transpose = x.T ## 计算x的转置

## 使用np.linalg.inv()计算矩阵的逆，.dot()计算矩阵乘法
theta = np.linalg.inv(x_transpose.dot(x)).dot(x_transpose).dot(y)

## 也可以用@来计算矩阵乘法
theta = np.linalg.inv(x_transpose@x)@x_transpose@y
```
<br>

## 2. gradient descent 梯度下降
---
## 梯度下降法在线性回归中的推导

梯度下降是一种优化算法，用于通过迭代的方法最小化目标函数。在**线性回归**中，梯度下降被用来最小化**均方误差（Mean Squared Error, MSE）**，从而找到最佳的回归系数。

### 1. 线性回归模型

考虑一个简单的线性回归模型：

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$


### 2. 均方误差（MSE）

目标是最小化预测值与实际值之间的残差平方和（Residual Sum of Squares, RSS）：

$$
RSS = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

均方误差（MSE）定义为：

$$
MSE = \frac{1}{n} RSS = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

### 3. 梯度下降法

梯度下降通过迭代更新回归系数，以最小化MSE。更新规则基于MSE对回归系数的偏导数（梯度）。

#### 3.1 计算梯度

对回归系数求偏导：

$$
\frac{\partial MSE}{\partial \beta_0} = -\frac{2}{n} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)
$$

$$
\frac{\partial MSE}{\partial \beta_1} = -\frac{2}{n} \sum_{i=1}^{n} X_i (Y_i - \beta_0 - \beta_1 X_i)
$$

#### 3.2 更新规则

根据梯度下降的原理，回归系数的更新规则如下：

$$
\beta_0 := \beta_0 - \alpha \frac{\partial MSE}{\partial \beta_0}
$$

$$
\beta_1 := \beta_1 - \alpha \frac{\partial MSE}{\partial \beta_1}
$$

其中，alpha是学习率（Learning Rate），决定了每次更新的步长。

### 4. 向量化表示

为了提高计算效率，梯度下降的更新可以使用向量和矩阵运算来表示。

#### 4.1 向量化模型

对于多个回归系数，线性回归模型可以表示为：

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

#### 4.2 均方误差的向量表示

$$
MSE = \frac{1}{n} (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
$$

#### 4.3 梯度的向量表示

梯度向量对应回归系数的偏导数：

$$
\nabla MSE = -\frac{2}{n} \mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
$$

#### 4.4 更新规则的向量表示

$$
\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla MSE
$$

即：

$$
\boldsymbol{\beta} := \boldsymbol{\beta} + \frac{2\alpha}{n} \mathbf{X}^\top (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
$$

### 5. 梯度下降算法步骤

1. **初始化**回归系数 \( \boldsymbol{\beta} \)（通常设为零或随机值）。
2. **计算预测值**：\( \hat{\mathbf{Y}} = \mathbf{X}\boldsymbol{\beta} \)。
3. **计算梯度**：\( \nabla MSE = -\frac{2}{n} \mathbf{X}^\top (\mathbf{Y} - \hat{\mathbf{Y}}) \)。
4. **更新回归系数**：
   $$
   \boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla MSE
   $$
5. **重复**步骤2-4，直到满足停止条件（如梯度足够小或达到最大迭代次数）。
<br>

```{python}
import numpy as np

## 假设x是自变量矩阵，y是一个因变量向量
x.shape = m,n  ## 求出x的维数
theta = np.zeros((n,1)) ## 初始化回归系数矩阵为零矩阵

## 开始迭代
for _ in range(iterations):
    prediction = x@theta ##求出预测的y值
    error = prediction - y.reshape(-1,1)
    updates = error@x.T / m ## 计算梯度
    theta -= updates * alpha ## 更新梯度，alpha为学习率
```
<br>

#### 补充：归一化（standardization）和标准化（min-max normalization）
```{python}
# standardization
## 假设data是一个array
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
standardized_data = (data - mean) / std

# min-max normalization
min_val = np.min(data, axis = 0)
max_val = np.max(data, axis = 0)
normalized_data = (data - min_val)/(max_val - min_val)
```
注意，这里要对列进行操作，因为矩阵里一个行是一个样本/观测，一个列是一个变量/特征。在numpy中，列用axis = 0表示，行用axis = 1表示。

<br>

## 3. 激活函数
### Sigmoid
Sigmoid 函数是一种广泛应用于机器学习和神经网络中的激活函数，尤其在二分类问题中扮演着重要角色。它能够将任意实数输入映射到介于0和1之间的输出，便于解释为概率。

通常，sigmoid函数会作为神经网络中神经元的处理函数，用于拟合所有可能的函数，因为很多个sigmoid函数加起来能够实现某个特定函数的输出。

具体的公式为：  
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

z是函数的输入，可以是任意实数。结果输出范围在0-1之间

<br>

### Softmax
Softmax 函数是一种广泛应用于多类别分类任务的激活函数，特别是在神经网络的输出层。它是Sigmoid 函数的推广，用于将任意实数向量转换为概率分布，使得每个类别的预测概率介于0和1之间，并且所有类别的概率之和为1。

公式如下：
$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, 2, \dots, K
$$

其中z是一个向量，输出结果为0-1的概率

```{python}
import numpy as np
import math

## 假设x是一个输入向量
upper = math.exp(x)
result = upper/sum(upper)
```
<br>

### logsoftmax
LogSoftmax 函数是 Softmax 函数的对数版本，它通过对 Softmax 的结果取对数来转换每个类别的输出。其数学公式为：
$$
LogSoftmax(z)_i = log(\frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}) = z_i - log(\sum_{j=1}^{k}e^{z_j})
$$

$$
z_i = x_i - max(x)
$$
其中，LogSoftmax直接返回每个类别的对数概率，$z_i$是每个x与x最大值的差。 

<mark>为什么使用 LogSoftmax？  

数值稳定性：直接计算 Softmax 和取对数时，可能会遇到数值不稳定的问题，特别是当 logits 值差距很大时。通过使用 LogSoftmax，我们可以避免这种问题，因为它将计算过程合并为一个表达式，避免了先计算 Softmax 再取对数的潜在数值不稳定性。

与交叉熵损失结合：在多分类任务中，常常使用交叉熵损失（cross-entropy loss）。交叉熵损失函数的定义通常包含 log(p)，即对概率分布取对数。为了高效计算，通常会将 LogSoftmax 和交叉熵损失结合在一起，避免对 Softmax 和对数分开计算。例如，PyTorch 和 TensorFlow 中的 LogSoftmax 函数就是用来与交叉熵损失配合使用的。

```{python}
## 对scores列表进行logsoftmax
scores = scores - max(scores)
res = scores - np.log(np.sum(np.exp(scores)))
```
<br>

### ReLU (Rectified Linear Unit)
ReLU是一种常用的激活函数，主要用于神经网络中，尤其是在深度神经网络的隐藏层中。
ReLU激活函数的数学表达式为：
$$
f(x) = max(0, x)
$$
也就是说，ReLU 会将输入中的所有负值变为 0，保留正值。这使得 ReLU 激活函数非常简单且有效，但也带来了一些问题，尤其是在网络训练过程中可能会出现“死亡神经元”现象：

死亡神经元问题：当 ReLU 的输入为负时，输出为 0。如果多个神经元的输入始终为负值，这些神经元的输出始终为 0，导致该神经元在训练过程中不再更新权重，从而变成“死神经元”，使得网络无法从这些神经元中获得任何有用信息。

<br>

### Leaky ReLU (Leaky Rectified Linear Unit)
为了解决 ReLU 中的“死亡神经元”问题，Leaky ReLU 对 ReLU 进行了改进，在负半轴上给出一个非常小的线性斜率。这使得即使输入为负值时，Leaky ReLU 也能有一个小的输出，而不是完全为 0，从而避免了“死神经元”的问题。
其数学公式如下：
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$
其中，$\alpha$是一个非常小的常数，通常为 0.01，控制负值区域的斜率。
```{python}
# 假设z是一个一维数组
## ReLU function
res = max(0,z) 

## Leaky ReLU
res = z if z > 0 else alpha * z
```
<br>

## 4. 特殊回归
### 1) Rridge Regression(岭回归)
Ridge Regression（岭回归）是一种用于解决多重共线性问题的线性回归方法。它通过在最小化普通最小二乘（OLS）损失函数的过程中，加入一个正则化项来限制模型复杂度，从而提高模型的泛化能力。
常见的OLS损失函数为：
$$
L_{\text{OLS}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = ||\mathbf{y} - X\beta||^2
$$

而Ridge Regression的损失函数，加上了一项正则化项来迫使模型的回归系数保持较小的值，从而减少过拟合的风险。
$$
L_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda ||\beta||^2
$$
其中，$\lambda$是超参数，控制正则化的强度。较大的λ值会导致回归系数趋于零，从而使模型更简单；较小的λ值则更接近于普通最小二乘回归。

```{python}
## 计算ridge regression的loss function
import numpy as np

## 假设X是一个feature matrix，w是一个coefficients matrix， y_true表示true labels，alpha表示regularization parameter.
ridge_loss = np.mean((y_true - X@w) ** 2) + np.sum(w**2) * alpha
```