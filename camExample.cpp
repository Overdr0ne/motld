#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "motld/MultiObjectTLD.h"

#define LOADCLASSIFIERATSTART 0
#define CLASSIFIERFILENAME "test.moctld"

//uncomment if you have a high resolution camera and want to speed up tracking
#define FORCE_RESIZING
#define RESOLUTION_X 320
#define RESOLUTION_Y 240

#define MOUSE_MODE_MARKER 0
#define MOUSE_MODE_ADD_BOX 1
#define MOUSE_MODE_IDLE 2
#define MOUSE_MODE_ADD_GATE 4

// #define _GLIBCXX_USE_CXX11_ABI 0

cv::Mat curImage;
bool ivQuit = false;
int ivWidth, ivHeight;
ObjectBox mouseBox = {0,0,0,0,0,boost::circular_buffer<CvPoint>(CB_LEN)};
cv::Point gate[2];
int mouseMode = MOUSE_MODE_IDLE;
int drawMode = 255;
bool learningEnabled = true, save = false, load = false, reset = false, cascadeDetect = false, drawPath = true, drawGateEnabled = false;
std::string cascadePath = "../haarcascades/haarcascade_frontalface_alt.xml";
int Ndetections = 0;

typedef struct DebugInfo
{
  int NObjects;
  int side0Cnt;
  int side1Cnt;
} DebugInfo;

void Init(cv::VideoCapture& capture);
void* Run(cv::VideoCapture& capture);
void* Run(void);
void HandleInput(int interval = 1);
void MouseHandler(int event, int x, int y, int flags, void* param);
void drawMouseBox();
void writeDebug(DebugInfo dbgInfo);
bool isVideo = true;

int main(int argc, char *argv[])
{
  // cv::VideoCapture capture(0);

  // Init(capture);
  // if(isVideo)
  //   Run(capture);
  // else
    Run();
  cv::destroyAllWindows();
  return 0;
}

void Init(cv::VideoCapture& capture)
{
  if(!capture.isOpened()){
    isVideo = false;
  }
  else
  {
    //propose a resolution
    capture.set(CV_CAP_PROP_FRAME_WIDTH, RESOLUTION_X);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y);
    //get the actual (supported) resolution
    ivWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    ivHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "camera/video resolution: " << ivWidth << "x" << ivHeight << std::endl;

    cv::namedWindow("MOCTLD", 0); //CV_WINDOW_AUTOSIZE );
    // cv::resizeWindow("MOCTLD", ivWidth, ivHeight);
  }
  cv::namedWindow("MOCTLD", 0); //CV_WINDOW_AUTOSIZE );
  cv::setMouseCallback("MOCTLD", MouseHandler);
}

void* Run()
{
  int size = ivWidth*ivHeight;
  cv::CascadeClassifier cascade;
  std::vector<cv::Rect> detectedFaces;
  std::vector<ObjectBox> trackBoxes;
  cv::Rect detectBox;
  std::ofstream dbgFile;
  cv::Mat frame;
  cv::Mat resized;
  dbgFile.open ("dbg.dat",std::ios::ate);

  if(cascadePath != "")
    cascade.load( cascadePath );

  while(!ivQuit)
  {
    // Grab an image
    frame = cv::imread("./capture/img0.jpg");
    std::cout << "here" << std::endl;
    cv::resize(frame,resized,cv::Size(RESOLUTION_X,RESOLUTION_Y), 0, 0, cv::INTER_CUBIC);
    resized.copyTo(curImage);

    cascade.detectMultiScale( resized, detectedFaces,
      1.1, 2, 0
      //|CASCADE_FIND_BIGGEST_OBJECT
      //|CASCADE_DO_ROUGH_SEARCH
      |cv::CASCADE_SCALE_IMAGE,
      cv::Size(30, 30) );

    Ndetections = detectedFaces.size();
    dbgFile << Ndetections << std::endl;

    // for( std::vector<cv::Rect>::const_iterator r = detectedFaces.begin(); r != detectedFaces.end(); r++ )
    for( std::vector<cv::Rect>::const_iterator r = detectedFaces.begin(); r != detectedFaces.end(); r++ )
    {
      detectBox.x = r->x;
      detectBox.y = r->y;
      detectBox.width = r->width;
      detectBox.height = r->height;
      cv::rectangle(curImage,detectBox,cv::Scalar(0,0,255));
    }

    // Display result
    HandleInput();
    drawMouseBox();
    cv::imshow("MOCTLD", curImage);
    cv::imwrite( "./dbg.jpg", curImage );
  }
  dbgFile.close();
  return 0;
}

void* Run(cv::VideoCapture& capture)
{
  int size = ivWidth*ivHeight;
  cv::CascadeClassifier cascade;
  std::vector<cv::Rect> detectedFaces;
  std::vector<ObjectBox> trackBoxes;
  cv::Rect detectBox;
  std::ofstream dbgFile;
  cv::Mat frame;
  int i = 0;
  std::string myImg;
  std::stringstream ss;
  dbgFile.open ("dbg.dat",std::ios::ate);

  if(cascadePath != "")
    cascade.load( cascadePath );

  while(!ivQuit)
  {
    // Grab an image
    if(!capture.grab()){
      std::cout << "error grabbing frame" << std::endl;
      break;
    }
    capture.retrieve(frame);
    ss << "./capture/img" << i << ".jpg";
    myImg = ss.str();
    std::cout << myImg;
    cv::imwrite( myImg, frame );
    ss.str(std::string());
    i=i+1;
    std::cout << i << std::endl;
    frame.copyTo(curImage);

    cascade.detectMultiScale( frame, detectedFaces,
      1.1, 2, 0
      //|CASCADE_FIND_BIGGEST_OBJECT
      //|CASCADE_DO_ROUGH_SEARCH
      |cv::CASCADE_SCALE_IMAGE,
      cv::Size(30, 30) );

    Ndetections = detectedFaces.size();
    dbgFile << Ndetections << std::endl;

    // for( std::vector<cv::Rect>::const_iterator r = detectedFaces.begin(); r != detectedFaces.end(); r++ )
    for( std::vector<cv::Rect>::const_iterator r = detectedFaces.begin(); r != detectedFaces.end(); r++ )
    {
      detectBox.x = r->x;
      detectBox.y = r->y;
      detectBox.width = r->width;
      detectBox.height = r->height;
      cv::rectangle(curImage,detectBox,cv::Scalar(0,0,255));
    }

    // Display result
    HandleInput();
    drawMouseBox();
    cv::imshow("MOCTLD", curImage);
    cv::imwrite( "./dbg.jpg", curImage );
  }
  dbgFile.close();
  capture.release();
  return 0;
}

void HandleInput(int interval)
{
  int key = cvWaitKey(interval);
  if(key >= 0)
  {
    switch (key)
    {
      case 'd': drawMode ^= DEBUG_DRAW_DETECTIONS;  break;
      case 't': drawMode ^= DEBUG_DRAW_CROSSES;  break;
      case 'p': drawMode ^= DEBUG_DRAW_PATCHES;  break;
      case 'h': drawMode ^= DEBUG_DRAW_PATH;  break;
      case 'c': cascadeDetect = !cascadeDetect;  break;
      case 'g': drawGateEnabled = true; break;
      case 'l':
        learningEnabled = !learningEnabled;
        std::cout << "learning " << (learningEnabled? "en" : "dis") << "abled" << std::endl;
        break;
      case 'r': reset = true; break;
      case 's': save = true;  break;
      case 'o': load = true;  break;
      case 27:  ivQuit = true; break; //ESC
      default:
        //std::cout << "unhandled key-code: " << key << std::endl;
        break;
    }
  }
}

void MouseHandler(int event, int x, int y, int flags, void* param)
{
  if(drawGateEnabled)
  {
    switch(event){
      case CV_EVENT_LBUTTONDOWN:
        gate[0].x = x;
        gate[0].y = y;
        //gate[0][0] = x;
        //gate[0][1] = y;
        mouseMode = MOUSE_MODE_MARKER;
        break;
      case CV_EVENT_MOUSEMOVE:
        if(mouseMode == MOUSE_MODE_MARKER){
          gate[1].x = x;
          gate[1].y = y;
          //gate[1][0] = x;
          //gate[1][1] = y;
        }
        break;
      case CV_EVENT_LBUTTONUP:
        if(mouseMode != MOUSE_MODE_MARKER)
          break;
        mouseMode = MOUSE_MODE_ADD_GATE;
        break;
      case CV_EVENT_RBUTTONDOWN:
        mouseMode = MOUSE_MODE_IDLE;
        break;
    }
  }
  else
  {
    switch(event){
      case CV_EVENT_LBUTTONDOWN:
        mouseBox.x = x;
        mouseBox.y = y;
        mouseBox.width = mouseBox.height = 0;
        mouseMode = MOUSE_MODE_MARKER;
        break;
      case CV_EVENT_MOUSEMOVE:
        if(mouseMode == MOUSE_MODE_MARKER){
          mouseBox.width = x - mouseBox.x;
          mouseBox.height = y - mouseBox.y;
        }
        break;
      case CV_EVENT_LBUTTONUP:
        if(mouseMode != MOUSE_MODE_MARKER)
          break;
        if(mouseBox.width < 0){
          mouseBox.x += mouseBox.width;
          mouseBox.width *= -1;
        }
        if(mouseBox.height < 0){
          mouseBox.y += mouseBox.height;
          mouseBox.height *= -1;
        }
        if(mouseBox.width < 4 || mouseBox.height < 4){
          std::cout << "bounding box too small!" << std::endl;
          mouseMode = MOUSE_MODE_IDLE;
        }else
          mouseMode = MOUSE_MODE_ADD_BOX;
        break;
      case CV_EVENT_RBUTTONDOWN:
        mouseMode = MOUSE_MODE_IDLE;
        break;
    }
  }
}

void writeDebug(DebugInfo dbgInfo)
{
  std::ofstream dbgFile;
  char strSide0[25];
  char strSide1[25];
  char strNObj[25];
  cv::Point midPt;
  sprintf(strSide0, "Side0: %i", dbgInfo.side0Cnt);
  sprintf(strSide1, "Side1: %i", dbgInfo.side1Cnt);
  sprintf(strNObj, "#objects: %i", dbgInfo.NObjects);

  dbgFile.open ("dbg.dat",std::ios::ate);
  dbgFile << dbgInfo.NObjects << ' ' << Ndetections << std::endl;
  dbgFile.close();

  cv::putText(curImage, strSide0, cv::Point(0, ivHeight/10), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
  cv::putText(curImage, strSide1, cv::Point(ivWidth-ivHeight/3, ivHeight/10), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
  cv::putText(curImage, strNObj, cv::Point(ivWidth-3*ivHeight/4, ivHeight-ivHeight/4), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
}

void drawBox()
{
  if(mouseMode == MOUSE_MODE_MARKER)
    {
      cv::Point pt1; pt1.x = mouseBox.x; pt1.y = mouseBox.y;
      cv::Point pt2; pt2.x = mouseBox.x + mouseBox.width; pt2.y = mouseBox.y + mouseBox.height;
      cv::rectangle(curImage, pt1, pt2, CV_RGB(0,0,255));
    }
}

void drawMouseBox()
{
  if(mouseMode == MOUSE_MODE_MARKER)
  {
    cv::Point pt1; pt1.x = mouseBox.x; pt1.y = mouseBox.y;
    cv::Point pt2; pt2.x = mouseBox.x + mouseBox.width; pt2.y = mouseBox.y + mouseBox.height;
    cv::rectangle(curImage, pt1, pt2, CV_RGB(0,0,255));
  }
}
