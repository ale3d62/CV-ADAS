from waymo_extractor_utils import camera_front_extract, video_extractor
import os

#---------------PARAMETERS---------------
CAMERA_IMAGES_PATH = "./camera_images/"
VIDEOS_PATH = "./extracted_waymo_videos"
VIDEO_FRAMERATE = 10 #fps
#----------------------------------------


if __name__ == "__main__":

    #Export images
    camera_front_extract.extract_front_camera_images(CAMERA_IMAGES_PATH, CAMERA_IMAGES_PATH)

    #Generate videos
    if not os.path.exists(VIDEOS_PATH):
        os.makedirs(VIDEOS_PATH)

    video_extractor.create_videos_from_images(CAMERA_IMAGES_PATH, VIDEOS_PATH, VIDEO_FRAMERATE)
