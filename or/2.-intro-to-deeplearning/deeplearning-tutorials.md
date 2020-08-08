# Deeplearning Tutorials

이제 딥러닝을 통한 음성 데이터 처리 실습을 해보자. 

## FC Layer

뉴럴넷의 가장 간단한 아키텍쳐를 세팅해보려고 한다. 먼저, 레이어의 갯수와 뉴런의 갯수 그리고, 각 layer들의 연결 패턴을 고려하는 것이 필요하다. 이번 케이스에서는 Input, hidden, output layer를 각각 하나씩 가지고 있는 모듈을 만들어 보자.

가장  activation function을 의해보자. 앞서 본 activation function  sigmoid와 relu를 사용해볼 것이다.

![](../../.gitbook/assets/image%20%287%29.png)

```python
import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape) # assert를 통해, dimension을 맞
    cache = Z 
    return A, cache
```



딥러닝 학습에서 중요한 것은 학습해야 하는 parameter가 무엇인지를 이해하는 것이다.  다음 함수를 통해 weight와 bias parameter를 레이어의 dimension에 맞게 정의해보자.

```python
def initialize_parameters_deep(layer_dims):
    # dictionary 객체 생성
    parameters = {}
    # 총 layer들의 길이를 계산
    L = len(layer_dims)
    # 레이어들을 돌면서, 레이어들 간의 weight와 bias의 초기값의 난수 생성
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # assert를 통해, dimension을 맞추줍니다. 틀릴시 error 발생
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters  
```



위의 정의한 함수를 각각 5, 4, 3, 1차원으로 정의한 4개의 레이어를 input으로 넣어보면 다음과 같은 input 값에 따른   parameter 결과를 얻을 수 있다.  

```python
layer_dims = [5,4,3,1]
parameters = initialize_parameters_deep(layer_dims)
parameters
```

```aspnet
##  {'W1': array([[-0.19988052,  0.13950715,  0.19509749, -0.18316674,  0.09670186],
##         [ 0.31722227, -0.104757  ,  0.31510588, -0.02915611,  0.07131382],
##         [-0.94082019, -0.62775115, -0.15116581, -0.38276253,  0.54977618],
##         [-0.16486756, -0.46020127, -0.35513057,  0.50480488, -0.4672517 ]]),
##  'W2': array([[-0.32151525,  0.0840826 , -0.42605469,  0.19201324],
##         [-0.91137902, -0.98562382,  0.00887571, -0.34684142],
##         [-0.27606418,  0.65011575,  0.099086  , -0.14360026]]),
##  'W3': array([[-0.40648812,  0.99661933,  0.09316457]]),
##  'b1': array([[0.],
##         [0.],
##         [0.],
##         [0.]]),
##  'b2': array([[0.],
##         [0.],
##         [0.]]),
##  'b3': array([[0.]])}
```



다음으로 forward 함수와 activation 함수를 정의하는 함수를 만들어보자. 여기서 각각의 연산에 사용된 input value들은 cache에 저장한다.

```python
def linear_forward(A, W, b):
    # W에 A를 내적하게 됩니다. 그후에는 b를 더해줍니다.
    Z = W.dot(A) + b
    # Z의 shape이 input과 weight의 shape과 동일한지를 체크합니다.
    assert(Z.shape == (W.shape[0], A.shape[1]))
    # 계산단계에서 사용한 값을 cache에 저장해둡니다.
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    # Activation function의 종류에 따라서 값을 나누어 줍니다.
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    # Shape이 input과 weight와 동일한지 체크해줍니다.
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    # linear 연산과 activation 연산을 cache에 저장해둡니다.
    cache = (linear_cache, activation_cache)
    return A, cache
```



위에서 정의한 activation forward 함수를 바탕으로 FC Layer를 쌓아보자.

```python
def L_model_forward(X, parameters):
    # cache 들의 list입니다.
    caches = []
    A = X
    # weight와 bias가 저장되어 있기 때문에 //2 를 해주어야 layer의 사이즈가 됩니다.
    L = len(parameters) // 2
    
    # hidden layersms relu를 통과
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # output layer는 sigmoid를 통과하게 한다
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches
```



만든 모델에 난수 X  넣어 결과값을 도출해보자.

실제 Y 레이블과의 차이가 어떠한가?

```python
X = np.random.randn(5,4)
Y = np.array([[0, 1, 1, 0]])
AL, caches = L_model_forward(X, parameters)
AL
```

```text
##  array([[0.57649229, 0.58858761, 0.50625841, 0.67020139]])
```



### \| Cost Function

우리는 신경망을 통과한 y^값을 찾을 수 있었다. 하지만 우리의 실제 y 레이블과는 다른 값일 가능성이 매우 크기 떄문에 이를 반영하여 학습을 시켜야한다. Cost function은 여러가지 종류가 있는데 이번에는 가장 많이 쓰이는 cost function 중에 하나인 cross-entropy 함수를 사용해보자.

cross-entropy 함수는 다음과 같다.

$$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))$$

이를 코드로 구현해보자. 

{% hint style="success" %}
**Arguments**

*   ****AL : 뉴럴넷을 통과해서 나오게된 y^ . shape \(1, number of examples\) 
*   Y -- 실제 "label" vector. \(for example: containing 0 if non-cat, 1 if cat\), shape \(1, number of examples\)
{% endhint %}

{% hint style="success" %}
**Returns**

* cost : cross-entropy cost
{% endhint %}

```python
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1.0/m)*np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost
    
cost = compute_cost(AL, Y)
print("cost = " + str(cost))
```

```text
##  cost = 0.7947985413241794
```





### **\| Backpropagation Process**

cost를 줄이기 위한 학습 파라미터를 업데이트 하기 위해서는 Backpropagation\(역전파\) 과정이 필요하다. Backpropagation은 다음과 같은 process를 가지게 된다.

* LINEAR backward
* LINEAR -&gt; ACTIVATION backward
* Layer -&gt; Layer backward

#### 1. Linear backward

Linear 한 영역에서 backward 과정은 다음과 같은 인자를 받게 된다.

{% hint style="success" %}
**Arguments**

*  dZ : Z의 변화량. linear 부분에서 ouput이 cost function 에 대한 gradient를 나타낸다.
*  cache : forward과정에서 필요한 값을 받아옵니다. tuple 형태의 \(A\_prev, W, b\) 값들을 받아온다.
{% endhint %}

{% hint style="success" %}
**Returns**

*  dA\_prev : Linear 구간의 input으로 들어왔었던, 지난 레이어의 activation 을 통과한 A가 cost function에 대한 변화량.
*  dW : Linear 구간의 weight의 cost function에 대한 변화량.
*  db : Linear 구간의 bias의 cost function 에 대한 변화량.
{% endhint %}

\*\*\*\*

#### 2. Linear-Activation backward

Activation function g\(.\) 에 대해서 Linear-activate backward는 다음과 같이 계산된다.

$$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$$.

{% hint style="success" %}
**Arguments**

*  dA : 현재 layer의 gradient값이 인자로 들어온다.

cache : forward pass에서 계산했던 linear\(Z\) 부분과 activation\(A\) 부분의 계산값들을 받는다.
{% endhint %}

{% hint style="success" %}
**Returns**

*  dA\_prev : Linear 구간의 input으로 들어왔었던, 지난 레이어의 activation 을 통과한 A가 cost function에 대한 변화량입니다. 

  $$dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}$$

* dW : Linear 구간의 weight의 cost function에 대한 변화량.$$dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$
*  db : Linear 구간의 bias의 cost function 에 대한 변화량.

  $$db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}$$
{% endhint %}

```python
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,cache[0].T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
    
def L_model_backward(AL, Y, caches):
    grads = {} # 빈 dictionary 호출
    L = len(caches) # 레이어의 갯수를 caches로 부터 받아옵니다.
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Shape을 AL과 동일하게 해줍니다.
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y,AL)- np.divide(1-Y, 1-AL))
    # caches index를 잡아둡니다.
    current_cache = caches[L-1] 
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    for l in reversed(range(L-1)):
        # indexing입니다.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads    
    
grads = L_model_backward(AL, Y, caches)
grads 
```

```text
##  {'dA0': array([[ 0.22615928,  0.05056479,  0.03725571,  0.26606754],
##          [-0.08893561,  0.06033779,  0.00678003,  0.08820048],
##          [-0.31533332, -0.00864554,  0.03630182, -0.37436958],
##          [ 0.18297927, -0.11513727, -0.06175443,  0.00178919],
##          [-0.22823877,  0.09102261,  0.02246496, -0.08833891]]),
##   'dA1': array([[ 0.24583103, -0.17543675, -0.21054402,  0.28579099],
##          [-0.25786842,  0.1840272 ,  0.22085354, -0.29978506],
##          [ 0.31603321, -0.22553637, -0.27066926,  0.36740456],
##          [ 0.31965621, -0.22812192, -0.27377221,  0.37161649]]),
##   'dA2': array([[ 0.78789736, -0.56228112, -0.67480121,  0.91597046],
##          [-0.51572388,  0.36804515,  0.44169598, -0.59955504],
##          [ 0.13185238, -0.09409615, -0.11292606,  0.15328505]]),
##   'dW1': array([[ 0.01718012,  0.07410485, -0.07740157, -0.01062174, -0.08069423],
##          [ 0.00538101, -0.10324898,  0.0618198 ,  0.01496212,  0.01577951],
##          [-0.03120275,  0.06252994, -0.13021064, -0.04486433, -0.07889032],
##          [ 0.00794814,  0.15945834, -0.08395172,  0.04680436, -0.01708135]]),
##   'dW2': array([[ 0.04299149, -0.09749935,  0.31746886, -0.12159809],
##          [-0.02814039,  0.0638189 , -0.20780153,  0.0795929 ],
##          [ 0.        ,  0.        ,  0.        ,  0.        ]]),
##   'dW3': array([[0.11232412, 0.03374054, 0.        ]]),
##   'db1': array([[0.0802695 ],
##          [0.0460068 ],
##          [0.17085944],
##          [0.02288357]]),
##   'db2': array([[ 0.11669637],
##          [-0.07638445],
##          [ 0.        ]]),
##   'db3': array([[0.08538493]])}
```



### **\|** Update parameter

파라미터를 업데이트 하는 규칙은 생각보다 간편합니다. Learning rate인 α 에 Gradient를 곱해서 현재의 parameter에 빼주면 새로운 parameter가 됩니다.

$$W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}$$ $$b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}$$

{% hint style="success" %}
**Arguments**

parameters : 파라미터들이 담겨져 있는 parameter dictionary.

 grads : Gradient들이 담겨있는 dictionaty.
{% endhint %}

{% hint style="success" %}
**Returns**

parameters : 업데이트되어있는 파라미터들이 담긴 dictionary.

* parameters\["W" + str\(l\)\] = ...
* parameters\["b" + str\(l\)\] = ...
{% endhint %}



```python
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # 레이어의 갯수입니다.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters
```

```text
parameters = update_parameters(parameters, grads, 0.05)
parameters
```

```text
##  {'W1': array([[-0.17780875, -0.03590766, -0.16854908,  0.29383995, -0.10266489],
##          [ 0.26400491, -0.30606321, -0.02412317,  0.07721398, -0.09659762],
##          [ 0.86338446,  0.26198627, -0.87832808, -0.22104154, -0.15349807],
##          [-0.00886301, -0.52353836,  0.02512958,  0.56527036, -0.47544392]]),
##   'W2': array([[ 0.47492277, -0.14788143,  0.5092166 ,  0.40566695],
##          [ 0.25358267,  0.26344798,  0.19980122, -0.01333085],
##          [-0.39912022, -0.24331496,  0.01081425, -0.71950724]]),
##   'W3': array([[ 1.36109307, -0.8962764 ,  0.22871491]]),
##   'b1': array([[-0.00401348],
##          [-0.00230034],
##          [-0.00854297],
##          [-0.00114418]]),
##   'b2': array([[-0.00583482],
##          [ 0.00381922],
##          [ 0.        ]]),
##   'b3': array([[-0.00426925]])}
```



## CNN

### CNN 

CNN 은 이미지의 **특징을 추출** 하는 부분과 **클래스를 분류** 하는 부분으로 나눌 수 있다.  


**1\) 특징 추출** : Convolution Layer,Pooling Layer

입력 데이터를 필터가 순회하며 합성곱을 계산하고, 그 계산 결과를 이용하여 Feature map을 만**든다.** Feature map sub-sampled 를 통해서 차원을 줄여주는 효과를 가지게 됩된다. Convolution Layer는 Filter 크기, Stride, Padding 적용여부, Max Pooling의 크기에 따라서 출력 데이터의 Shape이 결정된다.

* Convolution Layer : 입력데이터에 필터\(Filter or Weight\)를 적용 후 활성함수를 반영하는 요소.
* Pooling Layer\(Subsampling\) : spatial 차원의 다운샘플링을 책임짐.

\*\*\*\*

**2\) 클래스 분류** : Fully Connected Layer

#### 

### \| Convolve Window

이번에는 Filter를 이동시키며 convolution 연산하는 과정을 구해보자. input의 volume을 받아서\(3차원\), 모든 position의 input에 filter를 적용한다. Convolution 연산은 element wise multiplication\(아다마르 곱\)으로 이루어진다.

{% hint style="success" %}
**Argument**

* a\_slice\_prev : Filter가 적용될 Input. \(f, f, n\_C\_prev\)
* W : Filter의 사이즈.\(f, f, n\_C\_prev\)
* b : Bais - matrix of shape \(1, 1, 1\)
{% endhint %}

{% hint style="success" %}
**Returns**

* Z : Convolution 연산의 결과로 나오는 값입니다.
{% endhint %}

```python
#zero-padding하는 함수 
def zero_pad(X,pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=0)
    return X_pad

x = np.random.randn(4,3,3,2)
x_pad = zero_pad(x, 2)
print(x.shape, x_pad.shape)

#Convolution 연
def conv_single_step(a_slice_prev, W, b):
    # Element-wise product
    s = a_slice_prev * W
    # 채널을 기반으로 모두 더해줍니다.
    Z = np.sum(s)
    # Bias b를 더해줍니다.
    Z = Z + np.float(b)
    return Z
```



### \| Convolutional Neural Networks - Forward pass

Forward pass에서는 다양한 필터를 통해서, 구현을 위해 2D input의 horizental과 vertial index를 계산하면서 filter를 적용해본다. stack이 되는 output을 계산해보자.

Convolution의 output shape을 결정하는 식은 다음과 같습니다.

$$ nH = \lfloor \frac{n{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1$$

$$nW = \lfloor \frac{n{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1$$

$$n_C = \text{number of filters used in the convolution}$$

{% hint style="success" %}
**Arguments**

* A\_prev : Input으로 들어가는 Matrix입니다. 데이터의 batch m, Hight, Width, Channel이 포함되어 있습니다. \(m, n\_H\_prev, n\_W\_prev, n\_C\_prev\)
* W : Weights, Filter입니다. \(f, f, n\_C\_prev, n\_C\)
* b : Biases \(1, 1, 1, n\_C\)
* hparameters : "stride" 와 "pad"를 결정하는 python dictionary입니다.
{% endhint %}

{% hint style="success" %}
**Returns**

* Z : conv output입니다. \(m, n\_H, n\_W, n\_C\)
* cache : conv\_backward\(\) 에 도움을 줄 cache입니다.
{% endhint %}



```text
def pool_forward(A_prev, hparameters, mode="max"):
    # input의 shape을 받아옵니다
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # filter size와 stride size를 받아옵니다.
    f = hparameters["f"]
    stride = hparameters["stride"]
    # ouput dimension을 잡아줍시다
    n_H = int(1+(n_H_prev-f)/stride)
    n_W = int(1+(n_W_prev-f)/stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_prev_slice)
                        
    cache = (A_prev, hparameters)
    assert(A.shape ==(m,n_H,n_W,n_C))
    
    return A, cache
```

```text
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}
A, cache = pool_forward(A_prev, hparameters)
print(A)
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print(A)
```

```text
##  [[[[1.74481176 0.86540763 1.13376944]]]
##  
##  
##   [[[1.13162939 1.51981682 2.18557541]]]]
##  [[[[ 0.02105773 -0.20328806 -0.40389855]]]
##  
##  
##   [[[-0.22154621  0.51716526  0.48155844]]]]
```



## RNN Forward Pass

실제 RNN 모델의 경우에는 단일한 time-step이 적용되는 경우는 거의 없**다.** 이번에는 RNN cell이 10개가 붙어있다고 생각해 봅시다.

{% hint style="success" %}
**Arguments**

* x -- 모든 time-step의 input 데이터입니다. shape은 \(n\_x, m, T\_x\)결정됩니다.
* a0 -- 초기 hidden state입니다. hidden state의 갯수와, 데이터의 갯수로 shape이 결정됩니다.\(n\_a, m\)
* parameters -- python dictionary로 다음과 같은 정보가들어옵니다.
  * Waa -- hidden state에 대한 Weight matrix입니다.\(n\_a, n\_a\)
  * Wax -- input에 대한 Weight matrix입니다.\(n\_a, n\_x\)
  * Wya -- hidden state에서 output으로 가는 Weight matrix 입니다. \(n\_y, n\_a\)
  * ba -- hidden state에 대한 Bias입니다. \(n\_a, 1\)
  * by -- output에 대한 Bias입니다 \(n\_y, 1\)
{% endhint %}

{% hint style="success" %}
**Returns**

* a -- 모든 time step에 대한 hidden state 백터입니다. \(n\_a, m, T\_x\)
* y\_pred -- 예측된 Output입니다. \(n\_y, m, T\_x\)
* caches -- Backprop에 필요한 Caches입니다. \(list of caches, x\)
{% endhint %}

{% hint style="success" %}
**Task**

1. a인 hidden state vector의 공간을 zero vector로 만들어 줍니다.
2. a0 \(initial hidden state\)을 초기화 합니다.
3. Time step을 기반으로 for loop를 통해서 RNN cell 을 돌려줍니다. :
   * a \(tth position\)를 계산합니다. 즉 이전 스탭에서 현재 스탭으로 업데이트하는 것이죠.
   * a \(tth position\)를 캐시에 저장해 줍니다.
   * ypred를 다시 업데이트 해줍니다.
   * 캐시를 저장합니다.
4. 마지막 step의 a, y와 caches를 저장해줍니다.
{% endhint %}

```text
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def rnn_cell_forward(xt, a_prev, parameters):
    
    # parameter의 dict에서 데이터를 호출합니다.
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # 1. hidden state를 구현해봅시다.
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # 2. Predict Output를 구현해봅니다.
    yt_pred = softmax(np.dot(Wya, a_next)+by)
    # 3. Cache에 저장합시다.
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache
    
def rnn_forward(x, a0, parameters):
    # caches라는 cache를 저장할 list를 선언합니다.
    caches = []
    
    # Dimension을 맞추기 위해 input sequence 가준으로 unpacking 해줍니다.
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # a와 y를 초기화 합니다.
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    # a_next를 초기화 합니다.
    a_next = a0
    
    # time step을 돌면서 rnn cell을 작동 시킵니다.
    for t in range(T_x):
        # 1. hidden step을 계산해 줍니다.
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # 2. 새로운 hidden step을 a에 반영해 줍니다.
        a[:,:,t] = a_next
        # y의 값 역시 업데이트 해줍니다.
        y_pred[:,:,t] = yt_pred
        
        # 결과값을 저장해 줍니다.
        caches.append(cache)
        
    caches = (caches, x)
    return a, y_pred, caches
```

```text
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
```

```text
a
```

```text
array([[[-0.94679727,  0.99998902,  0.99859532,  0.99998339],
        [ 0.52566384,  0.99993012, -0.99996484,  0.99999942],
        [ 0.84483137,  0.99971338,  0.63006186,  0.99999504],
        [ 0.95268814,  0.9993585 ,  0.87825787,  0.9999979 ],
        [ 0.99996124,  0.9999908 ,  0.99897665,  0.6007902 ],
        [ 0.94354992, -0.99516219,  0.99987056,  0.99902443],
        [-0.31399689,  0.90494133,  0.99964112,  0.999997  ],
        [ 0.9985362 , -0.95921363,  0.97076661,  0.99792727],
        [ 0.99995626,  0.99994879,  0.55718656,  0.97797982],
        [ 0.99981346, -0.99139889, -0.90908533,  0.99994617]],

       [[ 0.9017533 , -0.0035545 , -0.40146936,  0.47240999],
        [-0.64008899, -0.99808521,  0.90937915,  0.99308063],
        [-0.61107796, -0.93987579, -0.82797531, -0.99944897],
        [ 0.69254271,  0.70004749,  0.95560602,  0.03494921],
        [ 0.99323355,  0.98511719,  0.93041097,  0.99371087],
        [-0.97376282,  0.89291419,  0.9777595 ,  0.68670555],
        [ 0.96905989,  0.84821902,  0.99428756,  0.91339115],
        [ 0.63387486, -0.0561147 , -0.06557296, -0.0515541 ],
        [ 0.34250436,  0.76229843,  0.89552076, -0.60056774],
        [-0.83173001,  0.94604565,  0.99882607,  0.98957886]],

       [[ 0.97087407,  0.96868192,  0.9860278 ,  0.62437977],
        [ 0.32586574, -0.91931824,  0.62536152,  0.83109594],
        [ 0.94009082,  0.79972708,  0.98280633, -0.9114071 ],
        [ 0.97102425,  0.69671275,  0.99918672, -0.81446397],
        [ 0.99869819,  0.81461615, -0.34958752,  0.98390801],
        [-0.9227938 ,  0.99784354,  0.99857354,  0.94312789],
        [-0.97880697,  0.62394864,  0.99397484, -0.99894842],
        [-0.78819779,  0.19186314,  0.91860743,  0.9916753 ],
        [ 0.99957809, -0.91253018,  0.71732866, -0.45986869],
        [-0.84758466, -0.98924985,  0.99999082,  0.99746386]],

       [[-0.9104497 ,  0.99927595,  0.94217573, -0.98743686],
        [-0.96081056,  0.99726769, -0.98947737, -0.97175622],
        [-0.93837279, -0.99812032, -0.99997534,  0.9759714 ],
        [ 0.9957971 ,  0.98744174, -0.91907333,  0.30870646],
        [ 0.84483456,  0.05888194,  0.57284256, -0.99798536],
        [ 0.98777081, -0.99999738, -0.91229958, -0.77235035],
        [-0.73832733,  0.84553649, -0.98818114,  0.08833992],
        [-0.99876665,  0.81798993,  0.99999724,  0.73642847],
        [ 0.41236695,  0.75086186, -0.36929754,  0.99998852],
        [ 0.93310421, -0.01108915, -0.99769046, -0.94005036]],

       [[-0.99935897, -0.57882882,  0.99953622,  0.99692362],
        [-0.99999375,  0.77911235, -0.99861469, -0.99833267],
        [ 0.98895163,  0.9905525 ,  0.87805502,  0.99623046],
        [ 0.9999802 ,  0.99693738,  0.99745184,  0.97406138],
        [-0.9912801 ,  0.98087418,  0.76076959,  0.54482277],
        [ 0.74865774, -0.59005528, -0.97721203,  0.92063859],
        [-0.96279238, -0.99825059,  0.95668547, -0.76146336],
        [-0.99251598, -0.95934467, -0.97402324,  0.99861032],
        [ 0.93272501,  0.81262652,  0.65510908,  0.69252916],
        [-0.1343305 , -0.99995298, -0.9994704 , -0.98612292]]])
```

