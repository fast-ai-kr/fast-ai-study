참고 : https://github.com/fastai/fastbook/blob/master/03_ethics.ipynb 

# 03. Data Ethics
데이터를 현장에 적용을 해야할 때 연관되는 사항들 및 윤리적인 부분을 접근해야 하는 부분들에 대한 내용

## Integrating Machine Learning with Product Design
머신러닝을 프로덕트 디자인에 적용할 때 수많은 `결정`을 내려야 한다.
- 데이터를 어떤 수준으로 집계하여 저장 할 것인지
- 어떤 손실 함수를 사용할 것인지
- 어떤 검증 및 훈련 세트를 사용해할지? 
- 어디에 중점을 둘지? : 구현의 단순성, 추론 속도 또는 모델의 정확성
- 모델이 도메인 밖 데이터들을 어떻게 처리할지?
- 미세 조정할 수 있는지, 또는 시간이 지남에 따라 처음부터 다시 훈련해야 하는지?

또한 여러분의 모델을 사용하는 사람들이 만들어진 모델을 적절하게 사용되지 못할 수 있다

![wrongly matched to criminal mugshots](https://github.com/fastai/fastbook/raw/c6322c68a53c38c26fa2fd0a5898f2afcbbb721b/images/ethics/image4.png)

* 아마존 안면 인식 소프트웨어 예 : 개발자(아마존), 연구자, 경찰 
* [아마존 안면 인식 소프트웨어 오류](https://www.google.co.kr/search?source=hp&ei=3u9dX52vMsHGmAWGhZrABw&q=%EC%95%84%EB%A7%88%EC%A1%B4+%EC%95%88%EB%A9%B4+%EC%9D%B8%EC%8B%9D+%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4+%EC%98%A4%EB%A5%98&oq=%EC%95%84%EB%A7%88%EC%A1%B4+%EC%95%88%EB%A9%B4+%EC%9D%B8%EC%8B%9D+%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4+%EC%98%A4%EB%A5%98&gs_lcp=CgZwc3ktYWIQAzIFCAAQzQI6BQgAELEDOgIIADoECAAQCjoICAAQsQMQgwE6BAgAEAM6CwgAELEDEIMBEIsDOggIABCxAxCLAzoECAAQHjoGCAAQCBAeOgUIIRCgAToECCEQFToECCEQClDeB1jENWD_NmgCcAB4AIABggGIAZsjkgEEMC4zOZgBAaABAqABAaoBB2d3cy13aXqwAQC4AQI&sclient=psy-ab&ved=0ahUKEwidlofr8eXrAhVBI6YKHYaCBngQ4dUDCAc&uact=5)
* [주의원 80명 중 26명 범죄자](https://biz.chosun.com/site/data/html_dir/2019/08/15/2019081501854.html)
* [아마존 안면인식 서비스 사업 중지](https://www.msn.com/ko-kr/news/techandscience/%EC%95%84%EB%A7%88%EC%A1%B4%EB%8F%84-%EC%95%88%EB%A9%B4%EC%9D%B8%EC%8B%9D-%EA%B8%B0%EC%88%A0-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%A4%91%EC%A7%80/ar-BB15ka7k)

그래서 위와 같은 경우를 최소화 하기 위해서는...
* 데이터 과학자는 연구자와 긴밀하게 협업해야 한다.
* 도메인 전문가(실무자)는 모델을 직접 훈련하고 디버깅 할 수 있을 정도로 배워야 한다.
* 현대 직장은 매우 전문화되어 자신만의 일을 하기 때문에 전체를 보기 힘들게 만든다.
* 그래서 앞의 것들에 대해서 대응하는 것은 매우 힘들지만 최선을 다해야 한다.
* 여러 부서에 걸쳐서 개선하려고 시도하는 사람들은 - 중간 경영진은 불편하게 생각하겠지만 - 고위 경영진은 높게 평가할 수도 있다.


### Topics in Data Ethnics
* 의지와 책임감
* 피드백 루프
* 편견
* 잘못된 정보

#### Recourse and accountability
복잡한 시스템 내에서는 결과에 대한 책임감이 약해질 수 있다. 또한 데이터에 근본적으로 오류가 있을 수 있다. 머신러닝 실무자는 알고리즘 동작 과정의 이해와 구현 뿐만 아니라 미치는 결과에 대하여 책임감을 가져야 한다. 
* 아칸소 헬스케어 시스템 문제 - 버그로 인한 뇌성마비 치료 누락 책임 문제
* [캘리포니아 갱 수배자 데이터베이스 문제](https://ko.livingorganicnews.com/is-gang-activity-rise-movement-abolish-gang-databases-makes-it-hard-tell-313704)
* 2012 년 연방 거래위원회 (FTC)의 대규모 신용보고 연구
* 공영 라디오 리포터 Bobby Allyn의 총기 유죄 판결 오기재 사건 

#### Feedback Loops
알고리즘은 환경과 상호작용하여 피드백 루프를 만들 수 있으며 실제 세계에서 수행되는 작업을 강화할 수 있다. 
하지만 알고리즘은 숫자를 최적화하기 위해 할 수 있는 모든 것을 하게된다. 사람과 알고리즘 간의 상호작용해서 피드백 루프가 만들어질 수 있으며 심지어 사람이 없는 피드백 루프도 만들어 질 수 있다. 따라서 피드백 루프를 방지하거나 첫 징후가 보일 때 개입해야 한다. 또한 편견(편향)도 조심해야 하는데 피드백 루프와 상호작용하여 매우 큰 문제를 만들 수 있다. 

* 구글 유튜브 추천 시스템의 강화학습 도입 문제 - 체류시간 및 조회수 기반 알고리즘의 문제
* ['소아성애자 놀이터 됐다' 폭로에 유튜브 화들짝…진화 나서](https://www.yna.co.kr/view/AKR20190222053500009)
* [On YouTube’s Digital Playground, an Open Gate for Pedophiles](https://www.nytimes.com/2019/06/03/world/americas/youtube-pedophiles.html)
* [옛 유튜브 알고리즘 담당자가 밝힌 추천 시스템의 비밀](http://www.bloter.net/archives/301890)
* [How an ex-YouTube insider investigated its secret algorithm](https://www.theguardian.com/technology/2018/feb/02/youtube-algorithm-election-clinton-trump-guillaume-chaslot)
* 유튜브 영상 속 카메라 등장을 바탕으로 한 분류 문제 - 카메라가 있는 영상은 사진 채널일까? 한번 사진 채널로 분류되면 그 이후로 분류될 확률이 지속적으로 올라간다.
* Meetup 남성/여성에게 기술 모임 추천 안내 - 모델을 짤 때 성별을 명시적으로 사용 안하여 피드백 루프 생성을 회피
* 반면 Facebook은 이러한 점 고려 안함 : 안티 백신 그룹 가입 -> 안티GMO, 켐트레일 감시(음모론), 지구 평면설 등등의 음모론 코스 제공



