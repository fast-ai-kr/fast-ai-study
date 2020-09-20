참고 : https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb

# 04. Under the Hood: Training a Digit Classifier
# 04. 내부 살펴보기: 숫자 분류 훈련

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

앞의 방법으로는 학습을 통해 점차 성능이 개성되어야 하는데 우리가 만든 모델은 그렇지 않았다.
이미지와 `이상적인 이미지` 간의 거리를 통해 유사도를 찾는 것 대신 다른 방법이 필요하다.
각 부분마다 다른 가중치(weight)를 가지는데 이를 이용하여 성능을 개선하는 함수를 다음과 같이 나타낼 수 있다.

```python
def pr_eight(x,w) = (x*w).sum()
```


우리는 주어진 이미지가 특정 것에 가까우는지(예를 들어 8) 구분하는데 적절한 w 값을 찾기를 원한다.

이러한 것들을 위해 머신 러닝 분류는 다음과 같은 단계를 거친다.
1. Initialize the weights.
1. For each image, use these weights to predict whether it appears to be a 3 or a 7.
1. Based on these predictions, calculate how good the model is (its loss).
1. Calculate the gradient, which measures for each weight, how changing that weight would change the loss
1. Step (that is, change) all the weights based on that calculation.
1. Go back to the step 2, and repeat the process.
1. Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).

![image](https://user-images.githubusercontent.com/1307187/93710455-eafca400-fb81-11ea-99d3-dbc0ec962909.png)

* 초기화 :: 랜덤 값으로 초기화한다. 어차피 반복되면 최적값을 찾아낼 것이기에 초기값은 중요치 않다.
* Loss :: weight에 대한 모델의 성능을 수치적으로 얻을 수 있는 함수가 있어야 한다. 일반적으로 작을 수록 좋고 클수록 나쁘다.
* 스텝 :: weight가 증가하거나 감소하게 하는 단계. 너무 작으면 너무 느려진다. 경사(gradient)를 계산하는 방식으로 성능 최적화를 통하여 적은 작업으로 빗슷한 결과를 얻을 수 있다.
* 종료 :: 모델 학습을 위하여 얼마나 많이 반복(epoch)할지 여부를 정해야 한다.  

```python
def f(x): return x**2
plot_function(f, 'x', 'x**2')
```
![image](https://user-images.githubusercontent.com/1307187/93710572-cb19b000-fb82-11ea-9581-485c5ad7b2ac.png)


어느 위치의 한 값(여기서는 -1.5)를 정하고 loss를 구한다. 
```python
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');
plot_function(f, 'x', 'x**2')
```
![image](https://user-images.githubusercontent.com/1307187/93710597-e8e71500-fb82-11ea-91a6-393c7ca09d58.png)

여기서 각 포인트에 따라 따라서 경사 값이 달라진다는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/1307187/93710603-f1d7e680-fb82-11ea-8888-8e40d2386ab2.png)

아래와 같이 반복할 수록 경사도는 점점 작아져서 loss의 최저 포인트에 이르게 된다.

![image](https://user-images.githubusercontent.com/1307187/93710609-f7cdc780-fb82-11ea-9585-9a0536a0a7e6.png)

이런식으로 최적의 값을 계산해낼 수 있다. 이에 대해서는 추후 장에서 다시 다룰 것이다.

### Calculating Gradients
### 그라디언트 계산하기

고등학교 때 미분 공부가 필요하다면 칸 아카데미 가설 들어봐라. (https://www.khanacademy.org/math/differential-calculus/dc-diff-intro)
PyTorch 쓰면 간단하게 미분 계산 된다. 아래와 같이 require_grad_()를 추가해주고 .backward() 메소드를 실행해서 계산해주면 된다.
```python
xt = tensor(3.).requires_grad_()
yt = f(xt)
yt
```
```python
tensor(9., grad_fn=<PowBackward0>)
```
```python
yt.backward()
```
`backward`는 역전파(*backpropagation*)에서 온 것으로 jargon 용어이다. `calculate_grad`라 이해하며 된다.

```python
xt.grad
```
```python
tensor(6.)
```

```python
xt = tensor([3.,4.,10.]).requires_grad_()
xt
```
```python
tensor([ 3.,  4., 10.], requires_grad=True)
```
`sum`을 넣어서 함수를 벡터(rank-1 tensor)를 처리할 수 있도록 하여 스칼라(rank-0 tensor)를 받는다.

def f(x): return (x**2).sum()

```python
yt = f(xt)
yt
```
```python
tensor(125., grad_fn=<SumBackward0>)
```
우리의 그라디언트는 `2*xt`였다. 예상대로다.

```python
yt.backward()
xt.grad
```
```python
tensor([ 6.,  8., 20.])
```

그라디언트는 단지 우리 함수의 경사(slope)일 뿐이다. 정확히 최적 파라미터로부터 얼마나 떨어져있는지 정확히 알려주지는 못한다. 하지만 아이디어는 얻을 수 있으며 경사가 매우 크고 작음에 따라 최적화된 값에 접근했는지 여부를 추측할 수 있다.
