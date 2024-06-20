# 프로젝트 소개
- 노이즈 캔슬링 이어폰을 착용했을 때 차량 소리를 인식하지 못하고 발생하는 사고를 예방하기 위한 프로젝트.
- 외부 소리를 캡처하여 차량 소리가 인식되면 사용자에게 경고음을 울려 차량이 접근 중임을 알려 노이즈 캔슬링을 사용하면서도 사고를 방지할 수 있다.

# 프로젝트 배경
- 노이즈 캔슬링 이어폰을 착용한 보행자와 차량의 사고가 계속해서 증가하고 있다.
- 특히 골목길이나 좁은 보차혼용도로에서 사고가 많이 발생하고 있다.
- 도로교통공단에서 노이즈 캔슬링의 유무에 따른 차량 소리 인식 거리를 측정하는 실험을 진행한 결과, 노이즈 캔슬링을 사용하지 않으면 약 8.7m의 거리에서 차를 인식하고 사용하면 0.8m의 매우 근접한 거리에서 인식하는 것을 확인할 수 있었다.

# 소리 분류 모델 개발

![model](https://github.com/jihyuk02/vehicle-warning-system/blob/main/images/model.jpg)

- 차량 소리는 말이나 사이렌 소리처럼 짧은 길이의 소리가 아닌 지속적으로 발생하는 소리이기 때문에, 적당히 긴 길이인 4초를 기준으로 소리를 잘라 학습을 진행할 수 있도록 했다.
- 4초 길이로 자른 소리를 Mel Spectrogram으로 변환하여 특징을 추출해준 뒤, 추출된 특징을 CNN 모델에 입력 값으로 학습시켜 소리를 분류하는 모델을 개발했다.
- 실세계에서는 차 소리만 존재하는 것이 아니라, 다양한 잡음(말 소리, 바람 소리 등)이 있기 때문에, 자동차/이륜자동차 소리 외에 바람 소리, 발걸음 소리, 말 소리 등의 소리를 함께 학습시켜 현실에서도 제대로 인식할 수 있도록 하였다.

# 앱 개발

<img src="https://github.com/jihyuk02/vehicle-warning-system/blob/main/images/%EC%95%B1.jpg" width="200" height="400"/>

- 소리 분류 모델을 탑재한 앱을 개발했다. 시작 버튼을 누르면 실시간으로 외부의 소리를 녹음하게 되고, 4초마다 4초 길이의 소리를 캡처한다. 캡처한 소리를 전처리하고 분류 모델에 입력하여 결과를 예측한 다음 차량 소리가 인식되면 사용자의 이어폰에 경고음을 출력하여 차량이 접근중임을 인식할 수 있도록 구현했다.

# 시연 영상
1. 차량이 보행자의 후방에서 접근 중인 상황

https://youtube.com/shorts/aFl-wd_UsIs?feature=share

2. 차량이나 건물에 의해 시야 확보가 어려운 상황

https://youtube.com/shorts/GSJEQrGYmlI?feature=share
