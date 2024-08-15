#Returs true if the system can keep processing the input, false otherwise
def canProcessVideo(inputVideos, videoSource):
    if(videoSource == "screen" or videoSource == "camera"):
        return True
    elif(videoSource == "video"):
        return len(inputVideos) > 0
    else:
        return False