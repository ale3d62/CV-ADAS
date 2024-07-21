#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;

const string IMG_PATH = "test_images/test2.jpg";

int main() {
    //cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    Mat img = imread(IMG_PATH, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error al cargar la imagen." << endl;
        return -1;
    }

    clock_t st = clock();

    // CROP
    int imgHeight = img.rows;
    int imgWidth = img.cols;
    int newImgHeight = round(imgHeight / 2);
    Mat croppedImg = img(Rect(0, newImgHeight, imgWidth, imgHeight - newImgHeight));

    // MASK
    vector<Point> vertices = {{0, newImgHeight}, {(int)(imgWidth * 0.3), 0}, {(int)(imgWidth * 0.7), 0}, {imgWidth, newImgHeight}};
    Mat mask = Mat::zeros(croppedImg.size(), CV_8UC1);
    vector<vector<Point>> contours = {vertices};
    fillPoly(mask, contours, Scalar(255, 255, 255));
    Mat masked_image;
    bitwise_and(croppedImg, croppedImg, masked_image, mask);

    // CANNY
    int t_lower = 50;
    int t_upper = 150;
    Mat edges;
    Canny(masked_image, edges, t_lower, t_upper, 3);

    cout << "Tiempo: " << (clock() - st) * 1000.0 / CLOCKS_PER_SEC << "ms" << endl;

    imshow("edges", edges);
    waitKey(0);
    return 0;
}