# Face detection using RetinaFace (serengil/retinaface)
# Licensed under the MIT License: https://github.com/serengil/retinaface/blob/master/LICENSE
import cv2
from retinaface import RetinaFace
from moviepy.editor import VideoFileClip
import os
import time

# Video input/output

def blur(video_path,output_path):
    temp_video_path = "temp_video.mp4"
    frame_count = 0
    cached_faces = []

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    start = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % 3 == 0:
            # Detect faces every 10th frame
            faces = RetinaFace.detect_faces(frame)
            cached_faces = [face["facial_area"] for face in faces.values()]
    
        for face in cached_faces:
            x1, y1, x2, y2 = map(int, face)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            
            frame[y1:y2, x1:x2] = cv2.resize(
                cv2.resize(frame[y1:y2, x1:x2], (6, 6), interpolation=cv2.INTER_AREA),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST)

        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()

    original_clip = VideoFileClip(video_path)
    new_clip = VideoFileClip(temp_video_path).set_audio(original_clip.audio)
    new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    original_clip.close()
    new_clip.close()
    os.remove(temp_video_path)

    end = time.time()
    print(start-end)


folder_path = ""

for i in range(1, 10):
    blur(folder_path + "/" + str(i) + ".mp4",folder_path + "/" + str(i) + "_blur.mp4")
