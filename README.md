# FAICE 스피커
## 🎙️ 과제 개요 및 목표
- **1인 가구가 점차 증가**하고 있고, 그 수치를 따라 외로움을 느끼는 인구의 수도 증가하기 때문에 **외로움을 해결할 방법**으로 AI 스피커를 선정
- 기존 AI 스피커와 다르게 카메라로 **사용자를 인식**해 **시간대 별로 상황에 맞는 메시지를 출력**함으로써 메시지를 들은 사용자가 짧은 대화를 통해 **외로움 해소** 가능
- AI 스피커를 이용함으로써 **1인 가구원의 감정을 공유**할 수 있는 대상이 되어줄 수 있음

## 🪄 기술 스택
- OS: Debian
- Language: Python

## ⏰ 개발 기간
- 2023.03.06 ~ 2023.06.09

## ✨ 주요 기능
- 사용자 얼굴 인식 후 지정된 시간에 맞는 멘트 출력
- 기존 AI 스피커와 동일한 간단한 질문에 대한 답변 가능
- 외부인 출입 시 자동 녹화

### 수행 로직
**Case 1. 등록된 사용자인 경우** <br/>
- openCV의 haarcascade를 이용한 사용자의 얼굴 인식
- 사용자의 얼굴을 인식하면 espeak를 활용하여 시간대별 멘트 출력 <br/>
    ex. AM 08:00 ~ 10:00 : “좋은 아침이에요!” <br/>
    PM 05:00 ~ 07:00 : “고생하셨어요!”
- 사용자가 “오케이 구글”을 이용해 스피커 호출
- 사용자가 원하는 질문을 하면 AI 스피커가 질문에 대한 답변 출력

**Case 2. 등록되지 않은 외부인인 경우** <br/>
- 사용자가 아닌 외부인 탐지 시 카메라에 찍히는 외부인 촬영 후 동영상 파일로 저장 <br/>
  → 이후 촬영된 동영상으로 외부인 출입 여부 확인 가능

### 얼굴 인식률
- **안경 착용**: 사용자가 안경 착용 시, 기존 인식률에 비해 10% 낮은 인식률(70% → 60%)
- **모자 착용**: 사용자가 모자 착용 시, 명암과 조도 비교 불가해 인식 불가
- **다른 환경**: 기존의 사진 찍은 환경과 다를 경우, 사용자 인식 가능(조명 센 곳, 어두운 곳 제외)

⇒ **평균 인식률 : 70 ~ 80%**

## 📹 시연 영상

https://github.com/user-attachments/assets/832c7adb-280b-468f-8ed0-2c9792e0b07d

