# Model Optimization

## 1. Introduction

<p align="center">
    <img src='https://github.com/boostcampaitech2/image-classification-level1-08/raw/master/_img/AI_Tech_head.png' height=50% width=50%></img>
</p>

<img src='https://github.com/boostcampaitech2/image-classification-level1-08/blob/master/_img/value_boostcamp.png?raw=true'>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 이 곳에는 그 세번 째 대회인 `Open-Domain Question Answering` 과제에 대한 **Level2-nlp-14조** 의 문제 해결 방법을 기록합니다.

### Team KiYOUNG2

_"Korean is all YOU Need for dialoGuE"_

#### 🔅 Members

김대웅|김채은|김태욱|유영재|이하람|진명훈|허진규|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/41335296?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/60843683?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/47404628?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/53523319?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/35680202?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/37775784?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/88299729?v=4' height=80 width=80px></img>
[Github](https://github.com/KimDaeUng)|[Github](https://github.com/Amber-Chaeeunk)|[Github](https://github.com/taeukkkim)|[Github](https://github.com/uyeongjae)|[Github](https://github.com/hrxorxm)|[Github](https://github.com/jinmang2)|[Github](https://github.com/JeangyuHeo)

#### 🔅 Contribution  

- [`진명훈`](https://github.com/jinmang2) &nbsp; NAS • Analyzing hyperparameter search results
- [`김대웅`](https://github.com/KimDaeUng) &nbsp; Curriculum Learning • DPR • Question Embedding Vis • KoEDA • Context Summary • Post processing • Ensemble(hard voting)
- [`김태욱`](https://github.com/taeukkkim) &nbsp; NAS • Analyzing hyperparameter search results
- [`허진규`](https://github.com/JeangyuHeo) &nbsp; NAS • Analyzing hyperparameter search results
- [`이하람`](https://github.com/hrxorxm) &nbsp; NAS • Analyzing hyperparameter search results
- [`김채은`](https://github.com/Amber-Chaeeunk) &nbsp; NAS • Knowledge distillation • Loss function • Pretrained model
- [`유영재`](https://github.com/uyeongjae) &nbsp; NAS • Analyzing hyperparameter search results

## 2. Project Outline
![](https://i.imgur.com/60lPM8Z.png)


모델 경량화는 deploy 시점에 고려해야할 중요한 기법 중 하나입니다. 본 대회는 재활용 쓰레기 데이터셋에 대해서 이미지 분류를 수행하는 모델을 설계합니다.

이번 프로젝트를 통해서는 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델을 만들어볼 예정입니다.


### Final Score

![](https://i.imgur.com/8H4Ems2.png)

## 3. Solution

- [솔루션 정리](./assets/kiyoung2_optimization_wrapup.pdf)


## 4. How to Use
```
.
├── assets
│   └── kiyoung2_optimization.pdf
├── configs
│   ├── data/taco.yaml
│   └── model/mobilenetv3.yaml
├── src
│   ├── augmentation
│   │     ├── methods.py
│   │     ├── policies.py
│   │     └── transforms.py
│   ├── modules
│   │     ├── __init__.py
│   │     ├── activations.py
│   │     ├── base_generator.py
│   │     ├── bottleneck.py
│   │     ├── conv.py
│   │     ├── dwconv.py
│   │     ├── flatten.py
│   │     ├── invertedresidualv2.py
│   │     ├── invertedresidualv3.py
│   │     ├── linear.py
│   │     ├── mbconv.py
│   │     └── poolings.py
│   ├── utils
│   │     └── pytransform
│   │           ├── __init__.py
│   │           └── _pytransform.so
│   │     ├── common.py
│   │     ├── data.py
│   │     └── torch_utils.py
│   ├── __init__.py
│   ├── dataloader.py
│   ├── loss.py
│   ├── model.py
│   └── trainer.py
├── .gitignore
├── README.md
├── inference.md
├── train.py
└── tune.py
```

아래 명령어로 실행 가능합니다.

### 1. train
```bash
python train.py \
    --model_config ${path_to_model_config} \
    --data_config ${path_to_data_config}
```

### 2. inference(submission.csv)
```bash
python inference.py \
    --model_config configs/model/mobilenetv3.yaml \
    --weight exp/2021-05-13_16-41-57/best.pt \
    --img_root /opt/ml/data/test \
    --data_config configs/data/taco.yaml3
```

## 3. References

### Baseline
Our basic structure is based on [Kindle](https://github.com/JeiKeiLim/kindle)(by [JeiKeiLim](https://github.com/JeiKeiLim))

### Paper
- [Distilling the Knowledge in a Neural Network
](https://arxiv.org/abs/1503.02531)

