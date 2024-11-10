from utils import (read_video,
                   save_video)
from trackers import Playertracker, Balltracker
from court_line_detector import CourtLineDetector
import cv2
def main():
    # read video
    input_video_path = "input_video.mp4"
    video_frames = read_video(input_video_path)
    
    #Detect players and balls
    player_tracker = Playertracker(model_path='yolov8x')
    ball_tracker = Balltracker(model_path='models/yolo5_last.pt') #we trained the model on specific datset for ball detection
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True, #for intial false then true to run the stored pickle file
                                                     stub_path='tracker_stubs/player_detections.pkl' )

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True, #for intial false then true to run the stored pickle file
                                                     stub_path='tracker_stubs/ball_detections.pkl' )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector model
    court_model_path = 'D:/DL project/models/new_keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #choose players
    player_detections =player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    # draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)

    # Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    #draw frame number on the top cornor
    for i , frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame {i}', (10, 120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # save output video
    save_video(output_video_frames, "output_videos/output.avi")
    #print(court_keypoints)
if __name__ == "__main__":
    main()