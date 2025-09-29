# Real-time Pose Estimation & Analysis

이 프로젝트는 웹캠으로 사람의 자세를 실시간으로 추적하고, 그 데이터를 분석하는 시스템이다.

main.py는 웹 서버 역할로 웹캠 영상을 받아서 분석하고, 결과를 웹페이지에 바로 보여주는 역할을 한다.

mediapipe_to_kinect_match.py는 분석된 데이터를 더 전문적으로 평가하고 처리하는 역할이다.

# 1. 주요 기능

이 프로젝트의 기능은 다음과 같다.

1. 실시간 자세 추적<div>
웹캠 영상을 받아서 MediaPipe 기술로<div>
실시간으로 사람의 관절 위치를 파악한다.

2. 데이터 분석 및 평가<div>
MediaPipe로 얻은 데이터를 Kinect 데이터셋 형식에 맞춰 변환하고,<div>
속도를 계산하거나 RMSE/DTW 같은 지표로 데이터의 정확도를 평가한다.

3. 실시간 웹 통신<div>
main.py의 Socket.IO를 이용해 웹페이지와 서버가 실시간으로 소통한다.<div>
웹에서 영상을 보내면, 서버가 분석 후 결과를 다시 웹으로 보내줘서 웹페이지에서 포즈 추적 결과를 볼 수 있다.

4. 데이터 저장<div>
 분석된 관절 데이터를 CSV 파일로 저장한다.<div>
 나중에 이 데이터를 가지고 AI 모델을 학습시키거나 분석할 때 활용할 수 있다.

# 2. 프로젝트 구조
프로젝트는 두 개의 파이썬 파일로 구성되어 있다.

main.py<div>
- FastAPI와 Socket.IO 서버를 설정하고 실행한다.<div>
- 웹캠 영상을 처리하고, MediaPipe로 포즈를 분석한다.<div>
- 분석된 데이터를 실시간으로 웹페이지에 전송하고, CSV 파일에 저장한다.

mediapipe_to_kinect_match.py<div>
- MediaPipe의 랜드마크를 Kinect 관절에 맞게 변환하는 KINECT_JOINTS 및 KINECT_TO_MEDIAPIPE 딕셔너리가 들어있다.<div>
- 관절의 위치 변화를 기반으로 속도를 계산하는 함수(compute_velocity_from_positions)를 포함한다.<div>
- Kinect 데이터셋과 MediaPipe 데이터를 비교할 수 있도록 시간 동기화 및 RMSE/DTW 같은 평가 지표를 계산하는 함수들이 들어있다.
