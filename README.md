# AUTOMATED-TENNIS-ACTION-RECOGNITION-WITH-YOLO-AND-KEYPOINT-MODELLING
"AI-Powered Tennis Analysis System: Leveraging YOLOv8, YOLOv5, and ResNet50 for Player Tracking, Ball Detection, and Court Keypoint Extraction"


This project presents an advanced AI/ML-based Tennis Analysis System that leverages cutting-edge object detection, tracking, and keypoint extraction techniques to analyze tennis matches. The system utilizes the YOLOv8 model for efficient person detection and tracking, assigning unique IDs to each individual in the video. For tennis ball detection, we developed a specialized model by fine-tuning the pretrained YOLOv5s architecture on a custom dataset, resulting in optimized weights and biases tailored for ball detection.  

To address instances where the ball detection model fails to identify the ball in certain frames, we implemented an interpolation method using the pandas library. This approach stores each frame's bounding box data in a DataFrame and estimates missing bounding box values, ensuring smooth and continuous ball detection throughout the video.  

The project also incorporates a novel approach for tennis court keypoint detection. A custom dataset of tennis court ground markings was used to train a new model based on the pretrained ResNet50 architecture. This enables precise detection of keypoints, which play a critical role in identifying and localizing the court within the video.  

Furthermore, while the YOLOv8 model tracks all individuals in the frame, we use the detected tennis court keypoints to filter out irrelevant persons and isolate the players of interest (Player 1 and Player 2). This integration ensures robust player tracking and accurate match analysis.  

The system demonstrates a comprehensive framework for analyzing tennis matches, showcasing the integration of state-of-the-art deep learning techniques for object detection, tracking, and keypoint extraction. It holds significant potential for applications in sports analytics, player performance evaluation, and match insights generation.
