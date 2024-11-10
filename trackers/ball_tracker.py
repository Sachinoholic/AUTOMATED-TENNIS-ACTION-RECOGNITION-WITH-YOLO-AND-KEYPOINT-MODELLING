from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
class Balltracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions): #helps to detect the missing detection
        ball_positions = [x.get(1,[])for x in ball_positions]
        #convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns= ['x1','y1','x2','y2'])

        #interpolate the missing value
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections =[]
        # it helps to store our tracking in .pkl format, no need to run many times for tarcking
        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)

        return ball_detections
    
################ for each individual having unique id's ##################

    def detect_frame(self, frame):
        results = self.model.predict(frame,conf = 0.15)[0]
        
        ball_dict = {}
        for box in results.boxes:

            result = box.xyxy.tolist()[0]
            
            ball_dict[1] = result
            
        return ball_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID:{track_id}", (int(bbox[0]),int(bbox[1] -10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255),2)
            output_video_frames.append(frame)
        
        return output_video_frames
