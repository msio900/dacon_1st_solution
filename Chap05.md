# ⚾Chapter05 외국인 투수 스카우팅 최적화
## 📑contents<a id='contents'></a>
1. 문제 정의 [👉🏻](#1)
2. 탐색적 데이터 분석 [👉🏻](#2)
3. 데이터 전처리[👉🏻](#3)
4. 모델 구축과 검증[👉🏻](#4)
4. 성능 향상을 위한 방법[👉🏻](#5)
6. 정리[👉🏻](#6)

## 1_ 문제 정의[📑](#contents)<a id='1'></a>

### 1_1 경진대회 소개

| 주최자      | 데이콘, 서울대학교 통계연구소 한국야구학회 |
| ---------- | ------------------------------------------------------------ |
| 총 상금     | 200만원                                    |
| 문제 유형   | Time-Series(시계열), Recommendation(추천)  |
| 평가척도    | 전달력, 논리성, 실용성, 간결성             |
| 대회 기간   | 2019년 3월 26일 ~ 2019년 5월 20일          |
| 대회 참여팀 | 80팀                                       |

### 1_2 평가 척도

* 제출 내용은 기존의 데이콘 경진대회와 달리 모델의 예측값을 제출하는 것이 아닌 최종적으로 스카우팅하고자 하는 외국인 선수 2명을 선정하고 그 과정을 정리한 인사이트 보고서를 제출

  > * **전달력** : 코드 또는 시각화에 대한 설명이 얼마나 이해하기 쉬운지에 관한 가중치 0.5
  > * **논리력** : 어떤 선수를 스카우트할 것인지에 대한 논리력 가중치 0.4
  > * **실용성** : 인사이트와 결과물이 실제 현업에서 쓸 수 있는 것인지에 관한 가중치 0.1
  > * **간결성** : 중요한 인사이트를 얼마나 간결하고 압축적으로 잘 표현했는지(동점 발생 시)

* 각각 척도는 5점 만점

* (전달력) * 0.5 +  (논리성) * 0.4 + (실용성) * 0.1

* 동점자 발생 시 간결성 점수를 기준으로 최종 등수를 산출

### 1_3 도메인 조사

* **이닝(inning)** : 두 팀이 한경기에 참여하고 두 팀은 공격과 수비를 번갈아가면서 하는데 양 팀이 한 번식 공수를 번갈아 하는 것을 말함
  * 야구는 9 이닝 내에 최대한 많은 점수를 얻는 것을 목표로 함
* **베이스** : 야구 경기장에는 4개의 판이 다이아몬드 형태로 놓여 있는 판
* **진루** : 각각의 베이스에 도착하는 것을 말함
  * 4개의 베이스를 모두 돌아 출발점으로 돌아왔을 때 공격팀은 1점을 얻음
* 수비팀의 역할은 공격팀이 진루를 못하도록 막는 것임
  * 첫 번째는 타자가 친 공이 땅에 떨어지기 전에 수비수가 잡는 것임. 수비수가 공을 먼저 잡으면 타자는 아웃이 되어 진루할 기회가 주어지지 않음.
  * 두 번째는 투수가 스트라이크를 3번 만들어 타자를 아웃시키는 것임. 스트라이크는 크게 2개의 형태로 이루어지는데, 타자가 스윙하지 않은 상태로 투수가 던진 공이 일정한 스트라이크 존을 통과했을 때나 투수가 던진 공을 타자가 스윙했지만 배트에 공을 맞추지 못했을 때 스트라이크로 판정됨
* 타자가 출루하기 위해 가장 먼저 맞서는 상대가 투수이기 때문에 수비팀에서 투수의 역할은 매우 중요
* 이 때문에 프로구단은 선발투수를 유능한 외국인 선수로 채워 넣는 경향이 있으며, 유능한 외국인 선발투수를 스카우트하는 것이 큰 관심사 임
* 외국 리그와 국내 프로야구 리그의 환경이 달라서 단순히 외국 리그의 성적만으로는 국내 프로야구에서의 활약을 예측하기 어려우므로, 이 예측을 용이하게 해줄 인사이트 발굴을 위해 본 대회의 목적임
* **세이버메트릭스(sabermetrics)** : 미국에서 데이터 분석을 스포츠에 접목하여 생긴 학문
* 단순 경기별 지표인 평균자책점, 안타 수, 홈런 수와 투구별 데이터의 경기별 지표가 주어짐

### 1_4 문제 해결을 위한 접근 방식 소개

* 스카우팅을 하는 주된 이유
  * 선수를 영입한 첫 시즌부터 좋은 성적을 보여줄 수 있는 투수를 확보
* 스카우팅의 목적
  * 선수 육성 차원보다는 영입 직후 팀 성적 향상에 기여할 수 있는 선수를 선발
* 주어진 데이터로부터 KBO 영입 후 첫 번째 시즌에 좋은 활약을 보인 외국인 투수 그룹을 선정
* 2011년부터 2018년 사이에 활약한 외국인 투수 중 첫 번째 시즌의 평균 자책점(Earned Run Average； ERA)이 상위 50%에 속하면서, 상대 타자 수 또한 상위 50%에 속하는 그룹을 KBO에서 좋은 활약을 보인 '엘리트 투수'라고 정의
* 엘리트 투수들이 KBO에 영입되기 이전에 MLB에서의 제구력을 평가
  * 제구력이 높을수록 KBO 첫 번째 시즌의 ERA가 유의미하게 줄어든다는 점을 선형회귀분석을 통해 검증
  * 제구력은 스트라이크 존 가장자리에 분포한 투구 중 타자가 스윙하지 않은 상태 에서 스트라이크 판정된 투구 비중이 10% 이상인 구종의 개수로 정의
  * 스트라이크 판정된 가장자리 투구는 공이 날아오는 궤적이 스트라이크 존에서 벗어나 보이지만, 마지막에 궤도가 변하여 스트라이크 존에 걸친 변화구가 포함돼 있다고 볼 수 있음
  * 타자가 볼(ball)로 판정될 것이라 판단하여 스윙을 안 했지만 최종적으로 스트라이크 판정이 난 것으로 볼 수 있음
  * 이러한 투구는 투수가 목표로 하는 투구라고 가정했고, 해당 투구를 활용해 제구력을 평가
* MLB에서의 제구력이 높을수록 KBO 첫 번째 시즌의 ERA가 유의미하게 줄어든다는 인사이 트를 밝힌 후, 2019년도 신입 외국인 투수 중 MLB에서의 제구력이 가장 높은 선수를 스카우팅
* 제어 능력을 측정할 때 사용하는 구종 개수가 동등한 선수가 3명 이상일 때는 해당 선수가 타자를 아웃시킬 확률을 구해 순위를 매긴 후 상위 2명을 스카우팅할 예정
* 전체 과정을 도식화하면 다음 그림 처럼 나타낼 수 있습니다.

| 2011~2018년 데이터로 학습                                    |                                                              |                           | 2019년 데이터 적용             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- | ------------------------------ |
| 1                                                            | 2                                                            | 3                         | 4                              |
| KBO 첫번째 시즌에서 '좋은 활약'을 보인 투수 선정<br />(엘리트 투수) | MLB 구종 증가할 수록 KBO 첫 번째 시즌 ERA 감소<br />(엘리트 투수) | 데이터 기반 인사이트 형성 | 인사이트<br />+<br />아웃 확률 |

> **그림** 분석과정 요약 이미지

* 5.2절에서는 제공된 데이터에 대한 탐색적 데이터 분석을 진행합니다. 변수별 분포와 특성을 확인함
* 5.3절에서는 분석을 위한 데이터 전처리를 진행합니다. 통계적으로 유의미한 데이터만 걸러내 통곗값에 신뢰성을 부여함
* 5.4절에서는 모델 구축 작업을 진행합니다. 선형회귀분석을 통해 가설을 검증하고 음이항 분포를 사용해 투수가 타자를 아웃시킬 확률을 추정함
* 5.5 절에서는 모델의 성능 향상 방법을 탐구합니다. 투수 평가 시 사용 가능한 추가 지표를 알아 보고 해당 지표를 기존 모델에 추가하여 스카우팅 모델을 구축해 봄

### 1_5 분석환경 구축

| 언어        | python 3 |
| ----------- | -------- |
| ide         | anaconda |
| matplotlib  | 3.3.3    |
| numpy       | 1.18.5   |
| pandas      | 1.0.5    |
| seaborn     | 0.10.1   |
| statsmodels | 0.12.1   |

## 2_ 탐색적 데이터 분석[📑](#contents)<a id='2'></a>

* 데이터를 목적에 맞게 활용하기 위해서는 데이터에 대한 이해가 선행되어야 함
* 주어진 데이터가 어떻게 구성되는지 알아야 데이터에 최적화된 모델링을 할 수 있는데 이 장에서는 데이터를 파악하기 위한 탐색적 데이터 분석을 실습함.

> 2011년부터 2018년까지 KBO에서 활약한 외국인 투수

| 파일명                                    | 내용                                     |
| ----------------------------------------- | ---------------------------------------- |
| kbo_yearly_foreigners_2011_2018. csv      | 역대 KBO 정규시즌 성적                   |
| fangraphs_foreigners_2011_2018. csv       | KBO 입성 전 MLB에서의 역대 정규시즌 성적 |
| baseball_savant_foreigners_2011_2018. csv | KBO 입성 전 MLB에서의 스탯캐스트 데이터  |

> 2019년 신규 KBO 외국인 투수

| 파일명                               | 내용                         |
| ------------------------------------ | ---------------------------- |
| fangraphs_foreigners_2019. csv       | MLB에서의 역대 정규시즌 성적 |
| baseball_savant_foreigners_2019 .csv | MLB에서의 스탯캐스트 데이터  |

* 2011년도부터 2018년도까지 KBO에서 활약한 외국인 투수에게는 KBO와 MLB에서의 성적이 모두 주어지는 반면, 2019년도 신규 KBO 외국인 투수에게는 MLB에서의 성적만 있음
* 2018년도까지 기록된 KBO, MLB 데이터를 통해 인사이트를 형성하고, 해당 인사이트를 2019년도 MLB 데이터에 적용해 스카우팅을 실시
  * KBO 데이터 : 투수의 정규시즌별 기록
  * MLB 데이터 : 투수의 정규 시즌별 기록과 더불어 경기별 기록
  * MLB의 정규시즌별 기록은 팬그래프 (FanGraphs) 웹사이트에서 제공한 데이터이고 MLB의 경기별 기록 데이터는 최신 추적 기술 시스템인 스탯캐스트(StatCast)를 활용해 측정된 데이터
  * 경기별 투수가 던진 투구의 구속, 위치, 투구 결과, 분당 회전수 등의 정보가 기록됨

### 2_1 KBO/팬그래프 데이터에 기록된 야구 지표 분석

* 우선 KBO 정규시즌 데이터와 MLB 정규시즌 데이터에 기록된 야구 지표를 분석

  > 데이터셋 불러오기

  ``` python
  import pandas as pd
  
  #데이터셋 불러오기 
  atKbo_11_18_KboRegSsn = pd.read_csv("./data/chap05/kbo_yearly_foreigners_2011_2018.csv")
  
  atKbo_11_18_MlbTot = pd.read_csv("./data/chap05/fangraphs_foreigners_2011_2018.csv")
  
  atKbo_19_MlbTot = pd.read_csv("./data/chap05/fangraphs_foreigners_2019.csv")
  ```

  > 데이터 크기 확인

  ```python
  print(atKbo_11_18_KboRegSsn.shape)
  print(atKbo_11_18_MlbTot.shape)
  print(atKbo_19_MlbTot.shape)
  
  # 실행 결과
  (105, 11)
  (205, 19)
  (41, 19)
  ```

* atKbo_11_18_KboRegSsn 데이터에는 105개의 행이 존재하며 11 개의 열이 존재

* atKbo_11_18_MlbTot 데이터에는 205개의 행이 존재하며 19개의 열이 존재

* atKbo_19_MlbTot 데이터에는 41 개의 행이 존재하며 19개의 열이 존재

* 해당 데이터 에는 2019년에 KBO에 입성한 외국인 투수들의 MLB 기록만 있어 atKbo_11_18_MlbTot 데이터보다 행의 개수가 줄어든 것을 확인 가능

* 하지만 수집된 MLB 기록은 같으므 로 열의 개수는 동일한 것을 확인 가능

  > `pd. DataFrame.columns`를 통해 변수 확인

  ```python
  print(atKbo_11_18_KboRegSsn.columns)
  print(atKbo_11_18_MlbTot.columns)
  print(atKbo_19_MlbTot.columns)
  
  # 실행 결과
  Index(['pitcher_name', 'year', 'team', 'ERA', 'TBF', 'H', 'HR', 'BB', 'HBP',
         'SO', 'year_born'],
        dtype='object')
  Index(['pitcher_name', 'year', 'ERA', 'WAR', 'TBF', 'H', 'HR', 'BB', 'HBP',
         'SO', 'WHIP', 'BABIP', 'FIP', 'LD%', 'GB%', 'FB%', 'IFFB%', 'SwStr%',
         'Swing%'],
        dtype='object')
  Index(['pitcher_name', 'year', 'ERA', 'WAR', 'TBF', 'H', 'HR', 'BB', 'HBP',
         'SO', 'WHIP', 'BABIP', 'FIP', 'LD%', 'GB%', 'FB%', 'IFFB%', 'SwStr%',
         'Swing%'],
        dtype='object')
  ```

* KBO/팬그래프 데이터 칼럼별 상세 설명

  | 칼럼명       | 상세                                                         |
  | ------------ | ------------------------------------------------------------ |
  | pitcher_name | 투수이름                                                     |
  | year         | 연도                                                         |
  | team         | 소속팀                                                       |
  | ERA          | 평균자책점(Earned run average)                               |
  | WAR          | 대체선수 대비 승리기여도(Wins above replacement)             |
  | TBF          | 상대한 타자 수(Total battlers faced)                         |
  | H            | 피안타 수(Hits)                                              |
  | HR           | 피홈런 수(Home runs)                                         |
  | BB           | 피볼넷 수(Bases on balls)                                    |
  | HBP          | 피사구 수(Hit by pitch)                                      |
  | SO           | 삼진 수(Strikeouts)                                          |
  | year_born    | 생년월일                                                     |
  | WHIP         | 이닝당 출루 허용률(Walks plus hits divided by innings pitched) |
  | BABIP        | 인플레이 타구 안타 비율(Batting average on balls in play)    |
  | FIP          | 수비 무관 자책점(Fielding independent pitching)              |
  | LD%          | 라인드라이브 비율(Line-drive rate)                           |
  | GB%          | 땅볼 비율 (Ground-ball rate)                                 |
  | FB%          | 플라이볼 비율(Fly-ball rate)                                 |
  | IFFB%        | 플라이볼 중 인필드 플라이볼 비율(Infield fly ball rate)      |
  | SwSTR%       | 헛스윙 비율(Swinging strike rate                             |
  | Swing%       | 스윙 비율(Swing rate)                                        |

* 총 21개의 칼럼이 KBO 데이터와 팬그래프 데이터에 나뉘어 기록

* 투수의 인적 사항, 투구 결과에 대한 여러 지표들 기록

* 안타 및 볼넷에서 파생된 지표 : WHIP, BABIP 등

* 라인드라이브 : 타구의 궤적에 따라 직선으로 날아가는 방향

* 땅볼 : 땅에 맞고 흘러가는 볼

* 플라이볼 : 공중으로 높이 솟아오르는 볼

* 이번 실습에서는 추후 분석에 이용될 주요 칼럼인 ERA와 TBF에 대한 데이터 탐색을 진행

  > KBO 정규시즌의 ERA와 TBF의 분포 및 기초통계량

  ```python
  atKbo_11_18_KboRegSsn[['ERA', 'TBF']].hist()
  print(atKbo_11_18_KboRegSsn[['ERA', 'TBF']].describe())
  
  # 실행 결과
                ERA         TBF
  count  105.000000  105.000000
  mean     4.714286  558.980952
  std      1.733072  224.436816
  min      2.380000   40.000000
  25%      3.740000  349.000000
  50%      4.420000  622.000000
  75%      5.230000  758.000000
  max     15.750000  870.000000
  ```

  ![](./image/5_2_1-1.png)

* ERA는 평균값이 4.71 이며 분포가 왼쪽으로 쏠려있는 것을 확인 가능

* TBF 는 평균값이 558이고 분포가 오른쪽으로 쏠려있음

  > MLB에서 ERA와 TBF의 분포와 기초통계량

  ```python
  atKbo_11_18_MlbTot[['ERA', 'TBF']].hist()
  print(atKbo_11_18_MlbTot[['ERA', 'TBF']].describe())
  
  # 실행 결과
                ERA         TBF
  count  205.000000  205.000000
  mean     5.750585  193.160976
  std      3.404285  186.000312
  min      0.000000    4.000000
  25%      4.050000   57.000000
  50%      5.200000  126.000000
  75%      6.750000  263.000000
  max     27.000000  827.000000
  ```

  ![](./image/5_2_1-2.png)

* ERA는 평균값이 5.75이며 분포가 왼쪽으로 쏠려있는 것을 확인 가능

* 반면 TBF는 평균값이 193이고 분포는 왼쪽으로 쏠려있음

* MLB에서의 TBF 값의 평균이 KBO보다 낮고 분포 또한 왼쪽으로 몰려있는 이유는 대부분의 KBO 외국인 투수들이 MLB에서는 선발 투수가 아니었기 때문

* 투수는 크게 선발투수와 불펜투수로 나뉘는데, 선발투수가 불펜투수보다 투구할 기회가 더 많이 주어짐.

* KBO에서 외국인 투수들은 대부분 선발투수 로 배정되어 상대한 타자 수가 많아 TBF 값이 높지만, MLB에서는 대부분 불펜투수로 활동 하여 TBF 값이 낮게 측정됐다고 볼 수 있음

* 투수를 평가할 때 사용하는 대표적인 지표로는 ERA가 있음

* MLB에서 ERA가 낮을수록 KBO에서도 낮을 것이라는 것이 증명된다면 외국인 투수를 스카우팅할 때 ERA가 낮은 투수 위주로 스카우트하면 됨

* 해당 가설을 확인해 보기 위해 다음 코드를 수행

  > 투수별 각 리그에서의 ERA 평균값

  ```python
  m_mean = (atKbo_11_18_MlbTot.groupby('pitcher_name')['ERA'].
            mean().
            reset_index().
            rename(columns={'ERA':'MLB_mean'}))
  k_mean = (atKbo_11_18_KboRegSsn.groupby('pitcher_name')['ERA'].
            mean().
            reset_index().
            rename(columns={'ERA':'KBO_mean'}))
  
  df = pd.merge(m_mean, k_mean, how = 'inner', on= 'pitcher_name')
  
  df.head()
  ```

  |      | pitcher_name |  MLB_mean | KBO_mean |
  | ---: | -----------: | --------: | -------- |
  |    0 |       니퍼트 |  4.290000 | 3.6875   |
  |    1 |   다이아몬드 | 10.262500 | 4.4200   |
  |    2 |     듀브론트 |  5.106667 | 4.9200   |
  |    3 |       레나도 |  7.306667 | 6.8000   |
  |    4 |         레온 |  6.050000 | 11.2500  |

* `groupby` 함수를 통해 `pitcher_name`별로 묶어준 후 ERA에 대한 평균값을 구함

  * 그리고 `pitcher_name`을 기준으로 `merge`함수를 적용하면 각 리그에 공통적으로 등장한 투수들의 ERA 평균값만 추출되어 데이터프레임 형태로 저장
  * 투수별로 해당 리그에 머문 시간이 다르므로 하나의 값으로 변환하여 비교하기 위해 평균값을 구함 

* 다음 코드를 통해 시각화해 데이터 간의 관계를 확인

  > MLB에서의 ERA 평균과 KBO에서의 ERA 평균 간 산점도

  ```python
  df.plot(kind = 'scatter', x = 'MLB_mean', y = 'KBO_mean')
  print(df.corr())
  ```

  ![](./image/5_2_1-3.png)

* MLB에서의 ERA 평균과 KBO에서의 ERA 평균 간 산점도
  * ERA는 작을수록 좋은 수치인데, MLB에서의 성적이 좋다고 해서 KBO에서의 성적이 좋아지는 경향을 찾기가 어려움
  * 두 개의 변수 간의 상관계수 生한 0.107로 수치가 낮은 편임을 알 수 있음
  * 이러한 분석을 통해 MLB에서의 ERA만으로 KBO에서의 투수 활약을 예측하기는 어려운 것 을 알 수 있음
  * ERA 지표가 투수 능력뿐만 아니라 투수가 속한 팀의 수비 능력 및 리그의 전반적인 수준에도 영향을 받는 지표이기 때문에 이런 결과가 나온 것으로 보임
* 5.4절에서는 온전히 투수의 능력만을 반영한 새로운 지표를 구축해봄

### 2_2 스탯캐스트 데이터에 기록된 야구 지표 분석

* 스탯캐스트 데이터에 대한 탐색적 데이터 분석을 실시

  > 스탯캐스트 데이터를 불러오고 데이터 크기를 확인

  ```python
  atKbo_11_18_StatCast = pd.read_csv("data/chap05/baseball_savant_foreigners_2011_2018.csv")
  
  atKbo_19_StatCast = pd.read_csv("data/chap05/baseball_savant_foreigners_2019.csv")
  
  print(atKbo_11_18_StatCast.shape)
  print(atKbo_19_StatCast.shape)
  
  # 실행 결과
  (135753, 24)
  (21903, 24)
  ```

* 행의 갯수가 상당히 많음.

* 스탯 캐스트 데이터는 경기에서 해당 투수가 던진 모든 투구별 데이터를 기록한 것이므로 관측치 많음

  > 스탯캐스트 데이터 칼럼명 확인

  ```python
  print(atKbo_11_18_StatCast.columns)
  print(atKbo_19_StatCast.columns)
  
  # 실행 결과
  Index(['game_date', 'release_speed', 'batter', 'pitcher', 'events',
         'description', 'zone', 'stand', 'p_throws', 'bb_type', 'balls',
         'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'ax', 'ay', 'az',
         'launch_speed', 'launch_angle', 'release_spin_rate', 'pitch_name',
         'pitcher_name'],
        dtype='object')
  Index(['game_date', 'release_speed', 'batter', 'pitcher', 'events',
         'description', 'zone', 'stand', 'p_throws', 'bb_type', 'balls',
         'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'ax', 'ay', 'az',
         'launch_speed', 'launch_angle', 'release_spin_rate', 'pitch_name',
         'pitcher_name'],
        dtype='object')
  ```

* 스탯캐스트 데이터 칼럼별 상세 설명

  | 칼럼명            | 상세                                           |
  | ----------------- | ---------------------------------------------- |
  | game_date         | 경기 날짜                                      |
  | release_speed     | 구속(miles/h)                                  |
  | batter            | 타자의 고유 아이디                             |
  | pitcher           | 투수의 고유 아이디                             |
  | event             | 해당 타석의 결과                               |
  | description       | 해당 공의 결과                                 |
  | zone              | 공이 홈 플레이트를 지날 때의 위치              |
  | stand             | 타자의 손잡이                                  |
  | p_throws          | 투수의 손잡이                                  |
  | bb_type           | 타구의 유형                                    |
  | balls             | 공을 던지기 직전 볼카운트중볼수                |
  | strikes           | 공을 던지기 직전 볼카운트 중 스트라이크 수     |
  | pfx_x             | 공의 수평 움직임（피트）                       |
  | pfx_z             | 공의 수직 움직임（피트）                       |
  | plate_x           | 공이 홈 플레이트를 지날 때의 수평 위치（피트） |
  | plate_z           | 공이 홈 플레이트를 지날 때의 수직 위치（피트） |
  | ax                | 공의 가속도의 X 성분                           |
  | ay                | 공의 가속도의 y 성분                           |
  | az                | 공의 가속도의 z 성분                           |
  | launch_speed      | 타구의 속도（2015년부터 기록）                 |
  | launch_angle      | 타구의 발사각도（2015년부터 기록）             |
  | release_spin_rate | 투수가 던진 공의 회전율（2015년부터 기록）     |
  | pitch_name        | 구종                                           |
  | pitcher_name      | 투수이름                                       |

* 24개의 칼럼이 스탯캐스트 데이터에 기록

* `events`, `description`, `pitch_name`에 대한 데이터 탐색 진행

  > 스탯캐스트 주요 칼럼 내용 확인

  ```python
  atKbo_11_18_StatCast[['events', 'description', 'pitch_name']]
  ```

  |        |    events |         description | pitch_name      |
  | -----: | --------: | ------------------: | --------------- |
  |      0 | field_out |       hit_into_play | 4-Seam Fastball |
  |      1 |  home_run | hit_into_play_score | 4-Seam Fastball |
  |      2 |       NaN |                foul | 4-Seam Fastball |
  |      3 |       NaN |        blocked_ball | 4-Seam Fastball |
  |      4 |       NaN |                ball | 4-Seam Fastball |
  |    ... |       ... |                 ... | ...             |
  | 135748 |       NaN |                ball | 4-Seam Fastball |
  | 135749 | field_out |       hit_into_play | Cutter          |
  | 135750 |       NaN |       called_strike | Cutter          |
  | 135751 |       NaN |                ball | 4-Seam Fastball |
  | 135752 |       NaN |                ball | 4-Seam Fastball |

  135753 rows × 3 columns

* 해당 칼럼 모두 문자열 값이 들어 있는 것을 확인 가능

  * `pitch.name` 칼럼에는 투수 가 던진 투구의 구종명이 기록
  * `description`에는 해당 공이 스트라이크인지, 볼로 판정됐는지, 또는 타자가 안타를 쳤는지 등의 공에 대한 결과가 기록
  * `events` 칼럼에는 해당 타석의 결과가 기록

* 결과가 일어난 후에는 타자가 바뀌기 때문에 타석에 대한 결과는 타자별로 한 개만 가짐

  * 가령 타자가 출루하거나 아웃된 것 을 `events` 칼럼에 기록
  * 다음 코드로 칼럼별 값의 분포를 확인
  * 먼저 `events` 칼럼에 기록된 값들의 분포를 확인

  > dacon/ch05/eda.ipynb - events 칼럼에 기록된 값들의 분포

  ```python
  (atKbo_11_18_StatCast['events'].
   value_counts().
   sort_values(ascending = True).
   plot(kind = 'barh', figsize=(8, 8)))
  ```

  ![](./image/5_2_2-1.png)

* 타석의 결과로 `field_out`, `single`, `strikeout` 등의 문자열이 기록된 것을 확인 가능

* 그 중 `field_out`의 기록이 약 1.5만 개로 가장 많음

* `description` 칼럼의 값들을 확인

  > description 칼럼의 값

  ```python
  (atKbo_11_18_StatCast['description'].
   value_counts().
   sort_values(ascending = True).
   plot(kind = 'barh', figsize=(8, 8)))
  ```

  ![](./image/5_2_2-2.png)

* `description` 칼럼에는 공의 결과인 `ball`, `called_strike`, `foul` 등의 값이 기록

* 볼(ball)의 비중이 약 5만 개로 가장 많은 것을 확인 가능

* 다음으로는 `pitch_name` 칼럼의 값들을 확인

  > pitch_name 칼럼의 값

  ```python
  (atKbo_11_18_StatCast['pitch_name'].
   value_counts().
   sort_values(ascending = True).
   plot(kind = 'barh', figsize=(8, 8)))
  ```

  ![](./image/5_2_2-3.png)

* `pitch_name`칼럼에는 `Fastball`, `Slider`, `Changeup` 등의 다양한 구종이 기록

* `4-Seam Fastball`이 가장 많이 기록됐으며 `Slider`, `Changeup` 순으로 가장 많이 던진 것을 알 수 있음

* 지금까지 대회에서 제공한 데이터에 기록된 주요 칼럼의 값들을 살펴보았고 이러한 방법으로 다른 칼럼 또한 확인해 볼 것을 권장

* 5.3절에서는 데이터 전처리를 통해 질 좋은 데이터를 추출하는 방법을 확인
