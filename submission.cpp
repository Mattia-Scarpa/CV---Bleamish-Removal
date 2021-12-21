#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include <iostream>
#include <cstdlib>
using namespace std;


string sourceWName = "Original Image";
Mat image;
int patchSize = 21;


Mat getBorder(Mat img, int& borderSize) {
  Mat src = img.clone();
  Mat frameMask = Mat::zeros(src.size(), CV_8U);
  frameMask(Range(1, patchSize-1), Range(1, patchSize-1)).setTo(255);
  bitwise_and(src, 0, src, frameMask);
  borderSize = countNonZero(255-frameMask);
  return src;
}


void swapCenterDFT(Mat src, Mat& swapped) {
  int centerX = src.cols/2;
  int centerY = src.rows/2;

  cv::Mat q1(src, cv::Rect(0, 0, centerX, centerY));
  cv::Mat q2(src, cv::Rect(centerX, 0, centerX, centerY));
  cv::Mat q3(src, cv::Rect(0, centerY, centerX, centerY));
  cv::Mat q4(src, cv::Rect(centerX, centerY, centerX, centerY));

  cv::Mat swapMap;

  q1.copyTo(swapMap);
  q4.copyTo(q1);
  swapMap.copyTo(q4);
  q2.copyTo(swapMap);
  q3.copyTo(q2);
  swapMap.copyTo(q3);

  swapped = src.clone();
}

void fourier(Mat sourceImg, Mat& dft_result) {

  Mat originalFloat;
  sourceImg.convertTo(originalFloat, CV_32FC1, 1.0/255.0);

  Mat originalComplex[2] = {originalFloat, cv::Mat::zeros(originalFloat.size(), CV_32F)};

  Mat dftReady;
  merge(originalComplex, 2, dftReady);

  dft(dftReady, dft_result, cv::DFT_COMPLEX_OUTPUT);
}

void getMagnitude(Mat dftImg, Mat& magnitude, bool centered = false) {

  Mat splitArray[2] = {cv::Mat::zeros(dftImg.size(), CV_32F), cv::Mat::zeros(dftImg.size(), CV_32F)};
  split(dftImg, splitArray);

  cv::magnitude(splitArray[0], splitArray[1], magnitude);

  magnitude += cv::Scalar::all(1);
  log(magnitude, magnitude);
  normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

  if (centered) {
    swapCenterDFT(magnitude, magnitude);
  }
}



void selector(int event, int x, int y, int flag, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
     Mat greyImg = *(Mat*) userdata;

     Mat patchGrey(greyImg, Rect(x-patchSize/2, y-patchSize/2, patchSize, patchSize));
     Mat patch(image, Rect(x-patchSize/2, y-patchSize/2, patchSize, patchSize));

     int ScoreTest = 10000;
     Mat tempPatch;

     for (int i(x-patchSize); i < x+patchSize; i++) {
       for (int j(y-patchSize); j < y+patchSize; j++) {
         int borderSize;

         if ((i < (x-patchSize/2))|| (i >= (x+patchSize/2)) && (j < (y-patchSize/2)) || (j >= (y+patchSize/2))) {
           Mat neighboorPatch(image, Rect(i-patchSize/2, j-patchSize/2, patchSize, patchSize));
           Mat neighboorPatchGrey(greyImg, Rect(i-patchSize/2, j-patchSize/2, patchSize, patchSize));
           Mat borderShift;
           divide(patch, neighboorPatch, borderShift, 1, CV_32F);
           borderShift = getBorder(borderShift, borderSize);
           float borderpatch = (float) mean(Scalar(1,1,1)-(sum(borderShift)/borderSize))[0];
           //cout << "border mean: " << abs(borderpatch) << endl;

           Mat neighboorDftImg;
           fourier(neighboorPatchGrey, neighboorDftImg);
           Mat neighboorMagnitude;
           getMagnitude(neighboorDftImg, neighboorMagnitude, true);
           float magpatch = (float) mean(neighboorMagnitude)[0];
           //cout << "border mag: " << magpatch << endl;

           if (magpatch + abs(borderpatch) < ScoreTest) {
             tempPatch = neighboorPatch;
             ScoreTest = magpatch + abs(borderpatch);
           }
         }
       }
     }
     seamlessClone(tempPatch, image, Mat::ones(tempPatch.rows, tempPatch.cols, CV_8U)*255, Point(x,y), image, NORMAL_CLONE);
     imshow(sourceWName, image);
   }
}



int main(int argc, char const *argv[]) {

  string path = "./blemish.png";

  image = imread(path, IMREAD_COLOR);
  Mat grey;
  cvtColor(image, grey, COLOR_BGR2GRAY);

  namedWindow(sourceWName);

  imshow(sourceWName, image);
  setMouseCallback(sourceWName, selector, (void*) &grey);
  waitKey();


  return 0;
}
