#include <iostream>
#include <opencv2/opencv.hpp>

#include <thread>
#include <chrono>

//To get terminal dimensions
#include <windows.h>
#include <mmsystem.h>

//To disable info messages
#include <opencv2/core/utils/logger.hpp>

//To run commands
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>


using namespace std;
using namespace cv;
using namespace chrono;



tuple<int, int> resDimensions(int resX, int resY, int termX, int termY){
    int newResX, newResY;

    termX/=2;

    if((termX/(double)termY) > (resX/(double)resY)){
        newResY = termY;
        newResX = termY*resX/(double)resY;
    }   
    else{
        newResX = termX;
        newResY = termX*resY/(double)resX;
    }

    tuple<int,int> newRes(newResX, newResY);
    return newRes;
}



const string chars = " `^”,:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const size_t nChars = chars.size();

char pixelToChar(int pixelVal){
    return chars[pixelVal/(double)256*nChars];
}



std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}



int main(int argc, char *argv[]){

    //CHECK ARGUMENTS
    if(argc < 2 || (strcmp(argv[1],"webcam") && strcmp(argv[1],"video") && strcmp(argv[1],"link"))){
        cout<<"ERROR - Expected one of the following arguments:"<<endl;
        cout<<" webcam"<<endl;
        cout<<" video"<<endl;
        cout<<" link"<<endl;
        return 1;
    }   


    //Disable info messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);


    //GET VIDEO SOURCE
    string videoFilename;
    VideoCapture cap;
    string audioLink = "";
    if(!strcmp(argv[1], "webcam")){
        cout<<"Loading webcam..."<<endl;
        cap = VideoCapture(0);
    }
    else if(!strcmp(argv[1], "video")){
        
        cout<<"Enter video filename: "<<endl;
        cin>>videoFilename;
        cap = VideoCapture(videoFilename);
    }
    else{
        string youtubeLink;
        cout<<"Enter video link: "<<endl;
        cin>>youtubeLink;
        cout<<"Loading video..."<<endl;
        
        string command = "yt-dlp -f \"bestvideo[height<=1080]\"+\"worstaudio\" -g "+youtubeLink;
        string output = exec(command.c_str());

        size_t separatorIndex = output.find('\n');

        string videoLink;

        
        if (separatorIndex != string::npos) {
            videoLink = output.substr(0, separatorIndex);
            audioLink = output.substr(separatorIndex + 1);
            audioLink.pop_back();
        } else {
            videoLink = output;
            videoLink.pop_back();
        }

        cap = VideoCapture(videoLink);
    }

    if(!cap.isOpened()){
        cout<<"Error getting the video signal"<<endl;
        return 1;
    }
    

    //Get fps
    int fps = cap.get(5);
    double msPerframe = 1/double(fps)*1000;


    //GET RESOLUTION
    int resX = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int resY = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    int termX = csbi.srWindow.Right - csbi.srWindow.Left+1;
    int termY = csbi.srWindow.Bottom - csbi.srWindow.Top;
    int prevTermX, prevTermY, prevResX, prevResY;
    prevTermX = prevTermY = prevResX = prevResY = 0;
    int newResX, newResY;
    tie (newResX, newResY) = resDimensions(resX, resY, termX, termY);


    Mat frame, resizedFrame, gray;
    

    //PLAY AUDIO
    if(!strcmp(argv[1], "video")){
        cout<<"Do you wish to play video with audio? (it may take a while if video is long) [Y/N]"<<endl;
        string playAudio;
        cin>>playAudio;
        if(playAudio != "N"){
            string command = "ffmpeg -y -i "+videoFilename+" -vn -loglevel quiet audio.wav";
            system(command.c_str());
            PlaySoundA("audio.wav", 0, SND_ASYNC);
            this_thread::sleep_for(milliseconds(10));
            remove("audio.wav");
        }
    }
    else if(!strcmp(argv[1], "link") && audioLink != ""){
        cout<<"Do you wish to play video with audio? (it may take a while if video is long) [Y/N]"<<endl;
        string playAudio;
        cin>>playAudio;
        if(playAudio != "N"){
            cout<<"Loading audio (this may take a while)..."<<endl;
            string command = "ffmpeg -y -i \""+audioLink+"\" -loglevel quiet audio.wav";
            system(command.c_str());
            PlaySoundA("audio.wav", 0, SND_ASYNC);
            this_thread::sleep_for(milliseconds(10));
            remove("audio.wav");
        }
    }

    bool hasFrames = true;
    auto start = high_resolution_clock::now();
    int nProcessedFrames = 0;
    while(hasFrames){

        //Get terminal dimensions
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
        termX = csbi.srWindow.Right - csbi.srWindow.Left+1;
        termY = csbi.srWindow.Bottom - csbi.srWindow.Top;
        if(termX != prevTermX || termY != prevTermY){
            prevTermX = termX;
            prevTermY = termY;
            tie (newResX, newResY) = resDimensions(resX, resY, termX, termY);
            if(newResX != prevResX || newResY != prevResY){
                system("cls");
                prevResX = newResX;
                prevResY = newResY;
            }
        }


        //RESIZE
        hasFrames = cap.read(frame);
        if(!hasFrames) break;
        resize(frame, resizedFrame, Size(newResX, newResY), INTER_LINEAR);


        //CONVERT TO GRAYSCALE
        cvtColor(resizedFrame, gray, COLOR_YUV2BGR);
        cvtColor(resizedFrame, gray, COLOR_BGR2GRAY);

        //CONTERT TO CHARS
        string charFrame = "\033[0;0H";

        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                charFrame += pixelToChar(static_cast<int>(gray.at<uchar>(i, j)));
                charFrame += " ";
            }
            charFrame += "\n";
        }

        
        //PRINT NEW FRAME
        cout<<charFrame;

        nProcessedFrames++;

        //WAIT FOR NEXT FRAME
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

        
        if(msPerframe*nProcessedFrames > duration.count())
            this_thread::sleep_for(milliseconds(int(msPerframe*nProcessedFrames-duration.count())));
    }

    system("cls");
    return 0;
}