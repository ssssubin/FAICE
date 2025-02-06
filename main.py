import cv2
import numpy as np
import os
import time
import datetime
import subprocess
#import speech_recognition as sr



#import pyttsx3
#import subprocess
#import skehql
#import pushtotalk as p
#text = "Hello Josubin"
#filename = 'skehql.py'

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/giga2/fdCam/trainer/trainer.yml')
cascadePath = "fdCam/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
a = 0
names = ['none','조수빈님', '마승우님','김정용님', '김정용님', '김정용', '김정용님']

cam = cv2.VideoCapture(0, cv2.CAP_V4L)
cam.set(3, 640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

def speak(option, msg):
	os.system("espeak {} '{}'".format(option, full_msg))
option = '-s 150 -p 95 -a 50 -v ko'

#video variable
is_recording = False
record_duration = 5000 #video recoding term(sec)
output_video = 'output.mp4' #save name
video_writer = None
now=datetime.datetime.now().time()


def time_speak():
	greeting=""
	
	if datetime.time(8,0)<=now<=datetime.time(10,0):
		greeting=f"좋 은 아 침 입 니 다  {names[int(id)]}"
	elif datetime.time(11,0)<=now<=datetime.time(14,0):
		greeting=f"점 심 은 드셨나요? {names[int(id)]}"
	elif datetime.time(14,1)<=now<=datetime.time(16,59):
		greeting=f"나른해지는 시간, 커피 한 잔 어때요? {names[int(id)]}"
	elif datetime.time(17,0)<=now<=datetime.time(19,0):
		greeting=f"오늘 하루도 고생했어요 {names[int(id)]}"
	elif datetime.time(22,0)<=now<=datetime.time(23,59):
		greeting=f"퇴근이 늦으셨네요. 오늘 하루도 정말 고생했어요 {names[int(id)]}"
	elif datetime.time(0,0)<=now<=datetime.time(2,0):
		greeting=f"퇴근이 늦으셨네요. 오늘 하루도 정말 고생했어요 {names[int(id)]}"
	return greeting
	
while True:
	
	ret, img = cam.read()
	#img = cv2.flip(img, -1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (int(minW), int(minH)),
	)
	

	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
		id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
		
		if (confidence < 60):
			id =str(id)
			confidence = "	{0}%".format(round(100-confidence))
		
			a += 1
			if(a==1):
				greeting=time_speak()
				full_msg=f"{greeting}"
				speak(option, full_msg)
				break
			
		else:
			id = "unknown"
			confidence = "   {0}%".format(round(100-confidence))
		
		cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
		cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

	cv2.imshow('camera', img)
	k= cv2.waitKey(10)&0xff
	if k==27:
		break
		
	if id == "unknown":
		if not is_recording:
			start_time = time.time()
			is_recording = True
			video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640,480))
		elif time.time() - start_time >= record_duration:
			is_recording = False
			video_writer.release()
	else:
		if is_recording:
			is_recording = False
			video_writer.release()
			
	if is_recording:
		video_writer.write(img)
		
		

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
