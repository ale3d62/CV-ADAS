#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;
using namespace std;

vector<double> knn(const vector<double>& points, int k) {
    cout<<"uyt"<<endl;
    vector<double> distances;
    distances.reserve(points.size() * points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            distances.push_back(abs(points[i] - points[j]));
        }
    }

    vector<size_t> indices(distances.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&distances](size_t i, size_t j) {
        return distances[i] < distances[j];
    });

    vector<double> kDistances(k);
    for (int i = 1; i <= k; ++i) {
        kDistances[i - 1] = distances[indices[i]];
    }

    vector<double> distancesMean;
    distancesMean.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        double sum = 0;
        cout<<"size: "<<kDistances.size()<<endl;
        for (int j = 0; j < k; ++j) {
            cout<<i * k + j<<endl;
            sum += kDistances[i * k + j];
        }
        distancesMean.push_back(sum / k);
    }
    

    return distancesMean;
}

vector<bool> getPointsMask(const vector<double>& points, double threshold, int k) {
    cout<<"hgf"<<endl;
    vector<double> neighbours = knn(points, k);
    vector<bool> mask(neighbours.size());

    for (size_t i = 0; i < neighbours.size(); ++i) {
        mask[i] = neighbours[i] <= threshold;
    }

    return mask;
}

pair<int, int> getBestLine(const vector<vector<double>>& linePoints, double threshold, int k, bool rec) {
    
    int minPoints = k;
    cout<<"dsa"<<endl;
    cout<<linePoints[0].size()<<" - "<<linePoints[1].size()<<endl;
    if (linePoints[0].empty()) {
        return make_pair(NULL, NULL);
    }
    else if (linePoints[0].size() < minPoints) {
        return make_pair(NULL, NULL);
    }

    vector<double> linePointsBottom = linePoints[0];
    vector<double> linePointsTop = linePoints[1];
    cout<<"rew"<<endl;
    vector<bool> filteredMaskBottom = getPointsMask(linePointsBottom, threshold, k);
    vector<bool> filteredMaskTop = getPointsMask(linePointsTop, threshold, k);

    vector<bool> mask(linePointsBottom.size());
    for (size_t i = 0; i < linePointsBottom.size(); ++i) {
        mask[i] = filteredMaskBottom[i] && filteredMaskTop[i];
    }
    

    if (any_of(mask.begin(), mask.end(), [](bool val) { return val; })) {
        if (rec) {
            double sumBottom = 0, sumTop = 0;
            int count = 0;
            for (size_t i = 0; i < mask.size(); ++i) {
                if (mask[i]) {
                    sumBottom += linePointsBottom[i];
                    sumTop += linePointsTop[i];
                    ++count;
                }
            }
            double bestPointBottom = sumBottom / count;
            double bestPointTop = sumTop / count;
            return make_pair(static_cast<int>(bestPointBottom), static_cast<int>(bestPointTop));
        }
        else {
            vector<vector<double>> newLinePoints = { {}, {} };
            for (size_t i = 0; i < mask.size(); ++i) {
                if (mask[i]) {
                    newLinePoints[0].push_back(linePointsBottom[i]);
                    newLinePoints[1].push_back(linePointsTop[i]);
                }
            }
            return getBestLine(newLinePoints, threshold, max(5, static_cast<int>(mask.size()) - accumulate(mask.begin(), mask.end(), 0)), true);
        }
    }
    else {
        return make_pair(NULL, NULL);;
    }
}



vector<int> findLane(Mat img) {

    // LOWER RESOLUTION
    double resScaling = 1;
    resize(img, img, Size(), resScaling, resScaling);

    // CROP TO HALF THE HEIGHT
    int newImgHeight = img.rows / 2;
    Mat croppedImg = img(Rect(0, newImgHeight, img.cols, img.rows - newImgHeight));

    // MASK
    vector<Point> vertices = {Point(0, newImgHeight), Point(cvRound(img.cols * 0.3), 0), 
                               Point(cvRound(img.cols * 0.7), 0), Point(img.cols, newImgHeight)};
    Mat mask = Mat::zeros(croppedImg.size(), croppedImg.type());
    fillConvexPoly(mask, vertices.data(), vertices.size(), Scalar(255, 255, 255));
    Mat masked_image;
    bitwise_and(croppedImg, mask, masked_image);

    // LAB
    Mat lab;
    cvtColor(masked_image, lab, COLOR_BGR2Lab);
    
    // Color Thresholding
    Scalar lower_white = Scalar(200, 1, 1);
    Scalar upper_white = Scalar(255, 255, 255);
    Mat mask_lab;
    inRange(lab, lower_white, upper_white, mask_lab);

    Mat colorMask;
    bitwise_and(masked_image, masked_image, colorMask, mask_lab);

    // OPEN
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(cvRound(3 * resScaling), cvRound(3 * resScaling)));
    morphologyEx(colorMask, colorMask, MORPH_OPEN, kernel);

    // CANNY
    int t_lower = 50;
    int t_upper = 300;
    Mat edges;
    auto start = chrono::high_resolution_clock::now();
    Canny(colorMask, edges, t_lower, t_upper, 3, false);
    auto end = chrono::high_resolution_clock::now();
    int time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout<<"time: "<<time<<"ms"<<endl;
    
    // HOUGH
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, int(20 * resScaling), int(50 * resScaling), int(30 * resScaling));

    if (lines.empty())
        return {NULL, NULL, NULL, NULL};

    vector<vector<int>> linesLeft(2);
    vector<vector<int>> linesRight(2);
    for (size_t i = 0; i < lines.size(); ++i) {
        Vec4i line = lines[i];
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];

        if (x2 == x1)
            continue;

        float m = static_cast<float>(y2 - y1) / (x2 - x1);
        float b = y1 - m * x1;

        // Filter lines by angle
        float lineAngle = fabs(atan(m));

        // Not a line (60-120 deg or < 25 deg or > 155 deg)
        if ((lineAngle > 1 && lineAngle < 2.2) || lineAngle < 0.43 || lineAngle > 2.7)
            continue;

        int xCutBottom = static_cast<int>((newImgHeight - b) / m);
        int xCutTop = static_cast<int>(-b / m);

        // Filter mask line
        float maskAngle = asin(newImgHeight / sqrt(newImgHeight * newImgHeight + img.cols * img.cols * 0.09));
        if ((xCutBottom < img.cols * 0.01 && lineAngle > maskAngle - maskAngle * 0.1 && lineAngle < maskAngle + maskAngle * 0.1) ||
            (xCutBottom > img.cols * 0.99 && lineAngle > maskAngle - maskAngle * 0.1 && lineAngle < maskAngle + maskAngle * 0.1))
            continue;

        // Classify
        if (m > 0) {  // rightLine
            if (xCutBottom < img.cols * 0.65)
                continue;
            linesRight[0].push_back(xCutBottom);
            linesRight[1].push_back(xCutTop);
        } else {  // left line
            if (xCutBottom > img.cols * 0.35)
                continue;
            linesLeft[0].push_back(xCutBottom);
            linesLeft[1].push_back(xCutTop);
        }
    }
    /*
    pair<int, int> bestLinePointsLeft = getBestLine(linesLeft, static_cast<int>(30 * resScaling), max(5, static_cast<int>(linesRight[0].size())), false);
    pair<int, int> bestLinePointsRight = getBestLine(linesRight, static_cast<int>(30 * resScaling), max(5, static_cast<int>(linesRight[0].size())), false);
    
    if (bestLinePointsLeft != make_pair(NULL, NULL)){
        bestLinePointsLeft.first *= 1 / resScaling;
        bestLinePointsLeft.second *= 1 / resScaling;
    }
    if (bestLinePointsRight !=  make_pair(NULL, NULL)){
        bestLinePointsRight.first *= 1 / resScaling;
        bestLinePointsRight.second *= 1 / resScaling;
    }*/
    
    return {1,1,1,1};
    //return {bestLinePointsLeft.first, bestLinePointsLeft.second, bestLinePointsRight.first, bestLinePointsRight.second};
}


int main() {
    // Suppress warnings
    //cv::setUseOptimized(false);
    //cv::ocl::setUseOpenCL(false);
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Loop through video files from 7 to 22
    for (int i = 7; i <= 22; ++i) {
        // Construct video file name
        string filename = "test_videos/test" + to_string(i) + ".mp4";

        // Open video file
        VideoCapture vid(filename);

        // Check if video is opened successfully
        if (!vid.isOpened()) {
            cout << "Could not open video file: " << filename << endl<< endl<< endl<< endl;
            continue;
        }

        double totalTime = 0;
        int totalFrames = 0;
        pair<int, int> bestLinePointsLeft;
        pair<int, int> bestLinePointsRight;

        // Process frames in the video
        while (true) {
            Mat frame;
            // Capture frame-by-frame
            vid >> frame;

            // Check if frame is empty
            if (frame.empty())
                break;

            int imgHeight = frame.rows;
            int imgWidth = frame.cols;
            int newImgHeight = imgHeight / 2;

            auto start = chrono::high_resolution_clock::now();
            
            vector<int> newBestLinePoints = findLane(frame);
            
            auto end = chrono::high_resolution_clock::now();
            totalTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
            totalFrames++;

            if (newBestLinePoints[0] != NULL && newBestLinePoints[1] != NULL)
                bestLinePointsLeft = make_pair(newBestLinePoints[0], newBestLinePoints[1]);
            if (newBestLinePoints[2] != NULL && newBestLinePoints[3] != NULL)
                bestLinePointsRight = make_pair(newBestLinePoints[2], newBestLinePoints[3]);

            if (bestLinePointsLeft != make_pair(NULL, NULL))
                line(frame, Point(bestLinePointsLeft.second, newImgHeight), Point(bestLinePointsLeft.first, imgHeight), Scalar(0, 0, 255), 2);
            if (bestLinePointsRight != make_pair(NULL, NULL))
                line(frame, Point(bestLinePointsRight.second, newImgHeight), Point(bestLinePointsRight.first, imgHeight), Scalar(0, 0, 255), 2);

            imshow("Frame", frame);

            // Press Q on keyboard to exit
            if (waitKey(1) == 'q')
                break;
        }

        if (totalFrames > 0)
            cout << "Avg time: " << totalTime / totalFrames << " ms" << endl;
    }

    return 0;
}