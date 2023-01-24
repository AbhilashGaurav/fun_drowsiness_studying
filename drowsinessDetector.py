import cv2
import dlib
import pyttsx3
import random
# In tag_activity.py
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from scipy.spatial import distance

engine = pyttsx3.init() # INITIALIZING THE pyttsx3 SO THAT ALERT AUDIO MESSAGE CAN BE DELIVERED

cap = cv2.VideoCapture(1)   # SETTING UP OF CAMERA TO 1 YOU CAN EVEN CHOSSE 0 IN PLACE OF 1

# FACE DETECTION OR MAPPING THE FACE TO GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()    
dlib_facelandmark = dlib.shape_predictor("C:\\Users\\Acer\\Desktop\\wakeupstudy\\shape_predictor_68_face_landmarks.dat")  # PUT THE LOCATION OF .DAT FILE (FILE FOR PREDECTING THE LANDMARKS ON FACE )

# FUNCTION CALCULATING THE ASPECT RATIO FOR THE Eye BY USING EUCLIDEAN DISTANCE FUNCTION
def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)

	return aspect_ratio_Eye


# working
# def play_videoFile(filePath,mirror=False):

#     cap = cv2.VideoCapture(filePath)
#     cv2.namedWindow('Video Life2Coding',cv2.WINDOW_AUTOSIZE)
#     while True:
#         ret_val, frame = cap.read()

#         if mirror:
#             frame = cv2.flip(frame, 0)

#         cv2.imshow('Video Life2Coding', frame)

#         if cv2.waitKey(1) == 50:
            # break  # esc to quit

    # cv2.destroyAllWindows()


# working
import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer
video_list=['1','2','3','4','5','6','7','8']
video_number = random.choice(video_list)
# choosing fro

video_path=str(video_number)+'.mp4'
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
# import tkinter as tk
# from tkVideoPlayer import TkinterVideo

# def play_me():
#     root = tk.Tk()

#     videoplayer = TkinterVideo(master=root, scaled=True)
#     videoplayer.load(r"1.mp4")
#     videoplayer.pack(expand=True, fill="both")

#     videoplayer.play() # play the video

#     root.mainloop()

#----------------------------------------------------------------
# MAIN LOOP IT WILL RUN ALL THE UNLESS AND UNTIL THE PROGRAM IS BEING KILLED BY THE USER
while True:
    null, frame = cap.read()    # READING AND STORING THE CAMERA PARAMETERS
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []    # ARRAY FOR THE LEFT EYES
        rightEye = []	# ARRAY FOR THE RIGHT EYES
        
        for n in range(42,48):    # THESE ARE THE POINTS ALLOCATION FOR THE LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        for n in range(36,42):      # THESE ARE THE POINTS ALLOCATION FOR THE RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36    
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(255,255,0),1)	
            
        # CALCULATING THE ASPECT RATIO FOR LEFT AND RIGHT EYE	
        right_Eye = Detect_Eye(rightEye);left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye+right_Eye)/2
        
        # NOW ROUND OF THE VALUE OF AVERAGE MEAN OF RIGHT AND LEFT EYES
        Eye_Rat = round(Eye_Rat,2)
        
        if Eye_Rat<0.15:    # THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT) WILL DECIDE WHETHER THE PERSONS'S EYES ARE CLOSE OR NOT
            cv2.putText(frame,"DROWSINESS DETECTED",(50,100),cv2.FONT_HERSHEY_PLAIN,2,(21,56,210),3)
            # cv2.putText(frame,"Alert!!!! WAKE UP DUDE",(50,450),cv2.FONT_HERSHEY_PLAIN,2,(21,56,212),3)
            
            # CALLING THE AUDIO FUCNTION OF TEXT TO AUDIO FOR ALERTING THE PERSON
            # play_videoFile('2.mp4',mirror=False)
            # play_me()
            video_list=['1','2','3','4','5','6','7','8']
            video_number = random.choice(video_list)
# choosing fro

            video_path=str(video_number)+'.mp4'
            PlayVideo(video_path)
            # engine.say("Alert!!!! WAKE UP DUDE")
            # engine.runAndWait()
            # print("Drowsiness detected")
        # print(Eye_Rat)
    cv2.imshow("Drowsiness DETECTOR FUNN IN OPENCV2", frame)
    key=cv2.waitKey(9)
    # print(key)
    if key == 20:
        break
cap.release()
cv2.destroyAllWindows()




# def main():
    

# if __name__ == '__main__':
#     main()