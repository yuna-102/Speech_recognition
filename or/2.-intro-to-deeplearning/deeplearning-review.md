# Deeplearning review

### Connectivity Pattern

딥러닝은 여러 레이어를 쌓아나가는 구조이다. 뉴런은 각 레이어에 있으며, 레이어들간의 연결관계에 따라서 다음과 같이 패턴이 나눠진다.

* Fully-Connected
* Convolutional
* Dilated
* Recurrent
* Skip / Residual
* Random

 이 중 FC Layer, CNN, RNN, LSTM, Attention에 대해 리뷰를 해보자.

## Fully-Connected Layer \(FC Layer\)

FC Layer는 Multi-Layer Perceptron \(MLP\)에서 non-linear function module을 뺀 레이어이다.  즉,  MLP는 다양한\(최소 3개의\) Fully connected layers로 구성된다고 볼 수 있다.

그렇다면 MLP란 무엇인가?

* MLP 

![](../../.gitbook/assets/image%20%289%29.png)

![](../../.gitbook/assets/image%20%2815%29.png)

## Convolutional Neural Networks \(CNN\)

Convolution Layer는 입력을 필터를 사용하여 Convolution연산을 하게 된다. 하이퍼 파라미터에는 필터 크기 F 및 보폭 S가 포함됩니다. 결과 출력 O를 feature map or activation map 이라고 부른다.

Pooling Layer 은 다운 샘플링 작업으로, 일반적으로 Convolution Layer 이후에 적용되며 spatial invariance을 수행한다. 특히 Max pool과 Average Pool 은 각각 최대 값과 평균값을 취하는 특수 종류의 풀링을 말한다.

![](../../.gitbook/assets/image%20%2834%29.png)

![](../../.gitbook/assets/image%20%2831%29.png)

![](../../.gitbook/assets/image%20%2832%29.png)

### CNN in Audio

* **1-D CNN in Spectrogram**

  Convolution filter의 크기가 frequency 영역대는 고정되어 있으며, Time에 따라서 진행된다.

  -Advantage : 1D feature map significantly reduce the number of paramerters-Fast to train-Small dataset   
  -Disadvantage : Not invariant to pitch shifting

> A 1D CNN is very effective when you expect to derive interesting features from shorter \(fixed-length\) segments of the overall data set and where the location of the feature within the segment is not of high relevance.

![](../../.gitbook/assets/image%20%288%29.png)

![](../../.gitbook/assets/image%20%2828%29.png)

![](../../.gitbook/assets/image%20%2825%29.png)



* **2-D CNN**

  Convolution filter의 크기가 frequenc과 Time에 따라서 진행된다.

  -Advantage : Time-frequency 두가지 영역에서 pattern을 찾게  된다. \(즉, flexibility를 가지게 되었다고 해석할 수 있다. 2D CNN이 1D CNN에 비해 큰 데이터 셋에 대해서 높은 성능을 보인다.

![](../../.gitbook/assets/image%20%2829%29.png)

![](../../.gitbook/assets/image%20%2833%29.png)

### 

### Sample CNN

Sample level CNN 의 가장 큰 특징은 바로 input데이터를 waveform 그 자체로 사용할 수 있다는 점이다.

Advantage 

*  CNN이 "phase-invariant" representation을 반영한다,
* 커널이 input signal에 대한 spectral bandwidth를 계산해준다.

![](../../.gitbook/assets/image%20%2835%29.png)

![](../../.gitbook/assets/image%20%2820%29.png)



## Recurrent Neural Network \(RNN\)

[  
](https://dreamgonfly.github.io/blog/understanding-rnn/)RNN은 Hidden State 를 유지하면서 이전 출력을 입력으로 사용할 수 있는 신경망이다.

• 어떠한 input length든 커버할 수 있다.

• 입력 크기에 따라 모델 Size가 증가하지 않는다.

• Historical Information을 잘 활용한다.

• 시간축에 따른 Weights Sharing이 진행된다.

![](../../.gitbook/assets/image%20%2830%29.png)

![](../../.gitbook/assets/image%20%2816%29.png)



## Long Short-Term Memory Unit \(LSTM\)

GRU \(Gated Recurrent Unit\) 및 LSTM \(Long Short-Term Memory Unit\)은 기존 RNN에서 vanishing gradient 를 처리하며 LSTM은 GRU의 일반화된 모델이다.

![](../../.gitbook/assets/image%20%2824%29.png)

![](../../.gitbook/assets/image%20%2810%29.png)

![](../../.gitbook/assets/image%20%2823%29.png)



## Attention

RNN계열 encoding 방식에서는 계속 마지막 hidden state까지 학습을 하면서 연산을 해야했다. 이러한 문제를 해결하는 것이 바로 attention이다. Attention은 input source 와 hidden state의 관계를 학습시키는 추가적인 Network를 만들게 된다. 이 Attention은 output에 의해서 weight를 학습한다.

![](../../.gitbook/assets/image%20%2814%29.png)

![](../../.gitbook/assets/image%20%2817%29.png)

![](../../.gitbook/assets/image%20%2819%29.png)

![](../../.gitbook/assets/image%20%2818%29.png)

![](../../.gitbook/assets/image%20%2812%29.png)

![](../../.gitbook/assets/image%20%2821%29.png)

