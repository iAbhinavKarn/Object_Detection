from imageai.Detection import VideoObjectDetection
import os


def main():
    execution_path = os.getcwd()
    dic = {}

    def forFrame(frame_number, output_array, output_count):
        print("Frame Number : " , frame_number)
        dic[frame_number] = output_array

    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    video_detector.loadModel(detection_speed="fast")

    video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video.mp4"), save_detected_video=False,  frames_per_second=20, per_frame_function = forFrame, minimum_percentage_probability=30)
    for k,v in dic.items():
        print(str(k) + " : " + str(v))
    video_path = video_detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "video.mp4"), output_file_path=os.path.join(execution_path, "video_output"), frames_per_second=29,  minimum_percentage_probability=30)

main()
