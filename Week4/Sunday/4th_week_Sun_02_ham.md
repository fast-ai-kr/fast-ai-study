참고 : https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb

# 04. Under the Hood: Training a Digit Classifier
# 04. 내부 살펴보기: 숫자 분류 학습 

## Computing Metrics Using Broadcasting
## 브로드캐스팅을 이용한 메트릭 계산

### 만들어진 모델 성능 측정하기
- 앞에서 봤던 두 가지 함수 - 평균 제곱 오차, 평균 절대 오차로 모델이 얼마나 잘 훈련되었는지 확인할 수 있다.
- 하지만 사람들은 `정확도`로 구분하기를 좋아하여 이를 모델의 성능 측정 기준으로 사용한다.
- 검증셋(validation set)을 사용하여 모델의 성능을 측정한다. 
- 검증셋을 따로 두는 이유는 기존 데이터만을 사용하면 오버핏(overfit)되어 학습 데이터에만 적합한 결과가 되기 쉽기 떄문이다.
- 하지만 이번 픽셀 유사 모델은 학습한 수가 적어서 오버핏 가능성은 낟다.
- 다향히 MNIST 데이터셋 제작자들이 준비해둔 검증셋을 사용한다. valid라는 디렉토리에 들어있다.

### 검증셋 준비
우선, 3과 7의 디렉토리로 텐서를 만들고 `이상적인 이미지`와의 차이를 통해 모델의 품질을 측정해본다.

```python
valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape, valid_7_tens.shape
```
```python
(torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))
```

- 작업할 때마다 shape를 체크한다. 

### is_3() 함수 만들기
임의의 이미지를 넣었을 때 3인지 7인지 알려주는 is_3 함수를 만든다.
```python
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
```
```python
tensor(0.1114)
```

위 결과는 이전에 두 이미지들의 거리를 계산한 것과 같은 결과이다. 
이 두 이미지들 이상적인 3인 `mean_3`과 임의의 샘플 3인 `a_3`은 `[28,28]` shape의 단일 이미지 텐서이다.

### 브로드캐스팅
그냥 함수에 텐서를 넣으면 브로드캐스팅되어 계산이 된다.

```python
valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```
```python
(tensor([0.1290, 0.1223, 0.1380,  ..., 0.1337, 0.1132, 0.1097]), torch.Size([1010]))
 ```
 인자가 매개변수 shape에 맞지않다는 에러 메시지 대신 모든 단일 이미지의 거리를 1,010개의 벡터(rank-1 텐서라고도 한다.)로 돌려준다.
  `mnist_distance` 함수를 다시 살펴보면 빼기 `(a-b)`가 있음을 알 수 있다. 마술은 PyTorch가 다른 순위의 두 텐서 사이에서 간단한 빼기 연산을 수행하려고 할 때 broadcasting 이 이루어진다는 것에 있다. 즉, 랭크가 작은 텐서를 자동으로 확장하여 랭크가 큰 텐서를 같은 크기로 만든다. 브로드캐스팅은 텐서 코드를 훨씬 쉽게 작성할 수 있도록하는 중요한 기능이다.
 
 두 개의 인수 텐서가 동일한 rank를 가지도록 브로드 캐스팅 한 후, PyTorch는 동일한 rank의 두 텐서에 대해 일반적인 로직을 적용한다. 즉, 두 개의 텐서의 각 해당 요소에 대해 연산을 수행하고 텐서 결과를 반환하는 것이다. 예를 들면 다음과 같다.
 
 ```python
 tensor([1,2,3]) + tensor([1,1,1])
 ```
 ```python
 tensor([2, 3, 4])
 ```
 
- 위~의 경우 경우 PyTorch는 단일 이미지를 나타내는 rank-2 텐서인 'mean3'을 동일한 이미지의 1,010 개 사본 인 것처럼 처리 한 다음 검증셋 각각의 3 이미지에서 각 사본을 뺐다. 
 
```python
(valid_3_tens-mean3).shape
```
```python
torch.Size([1010, 28, 28])
```
- 브로드캐스팅은 PyTorch에 의해 실행되며 성능 및 메모리 사용에서도 우수하다.
- 브로드캐스팅에 대해서는 차후 `17장`에서 다양하게 실습할 예정이다.

### is_3(x) 구현
```python
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
```
```python
is_3(a_3), is_3(a_3).float()
```
```python
(tensor(True), tensor(1.))
```

블린을 실수로 변환하여 `1.0`은 True로, `0.0`은 `False`로 바꿔서 쓴다.
```python
is_3(valid_3_tens)
```
```python
tensor([True, True, True,  ..., True, True, True])
```

이제 3과 7의 정확도를 계싼해본다. 7은 1에서 3의 정확도를 뺀 것으로 만든다.
```python
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
```python
(tensor(0.9168), tensor(0.9854), tensor(0.9511))
```

보면 90% 이상 나와서 꽤 괜찮은 성능으로 보이며 브로드캐스팅을 사용하여 어떻게 구하는지 확인해 보았다.
하지만 사실 3과 7은 매우 다른 형태이며 겨우 2가지만 비교한 거다. 그래서 개선이 필요하다. 
개선하기 위해서는 좀 더 실제 학습을 위한 시스템을 시도해야하는데, 자동으로 성능을 개선해주는 무엇인가가 필요하다.
이제 이를 위한 학습 방법인 SGD에 대해서 알아보자.

## Stochastic Gradient Descent (SGD)
## 확률적 경사 하강법 (SGD)

### Calculating Gradients
### 그라디언트 계산하기


