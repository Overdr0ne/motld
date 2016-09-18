/* Copyright (C) 2012 Christian Lutz, Thorsten Engesser
 * 
 * This file is part of motld
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
This is a live demo invoking a webcam.
It makes use of OpenCV to capture the frames and highgui to display the output.

Use your mouse to draw bounding boxes for tracking.
There are some keys to customize which components are displayed:
 D - dis/enable drawing of detections (green boxes)
 P - dis/enable drawing of learned patches
 T - dis/enable drawing of tracked points
 L - dis/enable learning
 S - save current classifier to CLASSIFIERFILENAME
 O - load classifier from CLASSIFIERFILENAME (not implemented)
 R - reset classifier (not implemented)
 ESC - exit
*/

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <ctime>

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
//int gate[2][2];
cv::Point gate[2];
int mouseMode = MOUSE_MODE_IDLE;
int drawMode = 255;
bool learningEnabled = true, save = false, load = false, reset = false, cascadeDetect = false, drawPath = true, drawGateEnabled = false;
std::string cascadePath = "/home/sam/src/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml";
std::clock_t c_start;
std::clock_t c_end;
std::clock_t c_start1;
std::clock_t c_end1;
std::clock_t c_start2;
std::clock_t c_end2;
std::clock_t c_start3;
std::clock_t c_end3;
// auto t_start;
// auto t_end;

typedef struct DebugInfo
{
  int NObjects;
  int side0Cnt;
  int side1Cnt;
} DebugInfo;

void Init(cv::VideoCapture& capture);
void* Run(cv::VideoCapture& capture);
void HandleInput(int interval = 1);
void MouseHandler(int event, int x, int y, int flags, void* param);
void BGR2RGB(Matrix& maRed, Matrix& maGreen, Matrix& maBlue);
void drawMouseBox();
void writeDebug(DebugInfo dbgInfo);
void drawGate();

int main(int argc, char *argv[])
{
  cv::VideoCapture capture(0);

  Init(capture);
  Run(capture);
  cv::destroyAllWindows();
  return 0;
}


void Init(cv::VideoCapture& capture)
{
  if(!capture.isOpened()){
    std::cout << "error starting video capture" << std::endl;
    exit(0);
  }
  //propose a resolution
  capture.set(CV_CAP_PROP_FRAME_WIDTH, RESOLUTION_X);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y);
  //get the actual (supported) resolution
  ivWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  ivHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  std::cout << "camera/video resolution: " << ivWidth << "x" << ivHeight << std::endl;

  cv::namedWindow("MOCTLD", 0); //CV_WINDOW_AUTOSIZE );
  // cv::resizeWindow("MOCTLD", ivWidth, ivHeight);
  cv::setMouseCallback("MOCTLD", MouseHandler);
}

void* Run(cv::VideoCapture& capture)
{
  int size = ivWidth*ivHeight;
  int count = 1;
  DebugInfo dbgInfo;
  cv::CascadeClassifier cascade;
  std::vector<cv::Rect> detectedFaces;
  std::vector<ObjectBox> trackBoxes;
  ObjectBox detectBox;

  // Initialize MultiObjectTLD
  #if LOADCLASSIFIERATSTART
  MultiObjectTLD p = MultiObjectTLD::loadClassifier((char*)CLASSIFIERFILENAME);
  #else
  MOTLDSettings settings(COLOR_MODE_RGB);
  settings.useColor = false;
  MultiObjectTLD p(ivWidth, ivHeight, settings);
  #endif

  if(cascadePath != "")
    cascade.load( cascadePath );

  Matrix maRed;
  Matrix maGreen;
  Matrix maBlue;
  unsigned char img[size*3];
  while(!ivQuit)
  {
    /*
    if(reset){
      p = *(new MultiObjectTLD(ivWidth, ivHeight, COLOR_MODE_RGB));
      reset = false;
    }
    if(load){
      p = MultiObjectTLD::loadClassifier(CLASSIFIERFILENAME);
      load = false;
    }
    */

#if TIMING
    c_end = std::clock();
#endif
    std::cout << "Total: " << (c_end-c_start) << std::endl;
#if TIMING
    c_start = std::clock();
#endif

    // Grab an image
    if(!capture.grab()){
      std::cout << "error grabbing frame" << std::endl;
      break;
    }
    cv::Mat frame;
    capture.retrieve(frame);
    frame.copyTo(curImage);
    //BGR to RGB
    // for(int j = 0; j<size; j++){
    //   img[j] = frame.at<cv::Vec3b>(j).val[2];
    //   img[j+size] = frame.at<cv::Vec3b>(j).val[1];
    //   img[j+2*size] = frame.at<cv::Vec3b>(j).val[0];
    // }
#if TIMING
    c_start1 = std::clock();
#endif
    for(int i = 0; i < ivHeight; ++i){
      for(int j = 0; j < ivWidth; ++j){
        img[i*ivWidth+j] = curImage.at<cv::Vec3b>(i,j).val[2];
        img[i*ivWidth+j+size] = curImage.at<cv::Vec3b>(i,j).val[1];
        img[i*ivWidth+j+2*size] = curImage.at<cv::Vec3b>(i,j).val[0];
      }
    }
#if TIMING
    c_end1 = std::clock();
    std::cout << "time1: " << (c_end1-c_start1) << std::endl;
#endif

    // for(int i = 0; i < ivHeight; ++i){
    //   for(int j = 0; j < ivWidth; ++j){
    //     curImage.at<cv::Vec3b>(i,j).val[2] = 0;
    //     curImage.at<cv::Vec3b>(i,j).val[1] = 0;
    //     curImage.at<cv::Vec3b>(i,j).val[0] = 0;
    //   }
    // }
    // cv::imshow("MOCTLD", curImage);

#if TIMING
    c_start2 = std::clock();
#endif
    // Process it with motld
    p.processFrame(img);
#if TIMING
    c_end2 = std::clock();
    std::cout << "time2: " << (c_end2-c_start2) << std::endl;
#endif

    // Add new box
    if(mouseMode == MOUSE_MODE_ADD_BOX){
      p.addObject(mouseBox);
      mouseMode = MOUSE_MODE_IDLE;
    }

    if(mouseMode == MOUSE_MODE_ADD_GATE){
      p.addGate(gate);
      mouseMode = MOUSE_MODE_IDLE;
    }

    if(((count%20)==0) && cascadeDetect)
    {
      cascade.detectMultiScale( frame, detectedFaces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |cv::CASCADE_SCALE_IMAGE,
        cv::Size(30, 30) );

      for( std::vector<cv::Rect>::const_iterator r = detectedFaces.begin(); r != detectedFaces.end(); r++ )
      {
        detectBox.x = r->x;
        detectBox.y = r->y;
        detectBox.width = r->width;
        detectBox.height = r->height;
        if(p.isNewObject(detectBox))
          p.addObject(detectBox);
      }
      //printf("size detectedFaces: %i\n", detectedFaces.size());
    }
    count++;

    // Display result
    HandleInput();
    p.getDebugImage(img, maRed, maGreen, maBlue, drawMode);
#if TIMING
    c_start3 = std::clock();
#endif
    BGR2RGB(maRed, maGreen, maBlue);
#if TIMING
    c_end3 = std::clock();
    std::cout << "time3: " << (c_end3-c_start3) << std::endl;
#endif
    drawGate();
    drawMouseBox();
    dbgInfo.NObjects = p.getObjectTotal();
    dbgInfo.side0Cnt = p.getSide0Cnt();
    dbgInfo.side1Cnt = p.getSide1Cnt();
    writeDebug(dbgInfo);
    cv::imshow("MOCTLD", curImage);
    p.enableLearning(learningEnabled);
    if(save){
      p.saveClassifier((char*)CLASSIFIERFILENAME);
      save = false;
    }
  }
  //delete[] img;
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

void BGR2RGB(Matrix& maRed, Matrix& maGreen, Matrix& maBlue)
{
  for(int i = 0; i < ivHeight; ++i){
    for(int j = 0; j < ivWidth; ++j){
      curImage.at<cv::Vec3b>(i,j).val[2] = maRed.data()[i*ivWidth+j];
      curImage.at<cv::Vec3b>(i,j).val[1] = maGreen.data()[i*ivWidth+j];
      curImage.at<cv::Vec3b>(i,j).val[0] = maBlue.data()[i*ivWidth+j];
    }
  }

  //at this place you could save the images using
  //cvSaveImage(filename, curImage);
}

void drawGate()
{
  if(drawGateEnabled)
  {
    cv::line(curImage,gate[0],gate[1],cv::Scalar(0,0,255));
  }
}

void writeDebug(DebugInfo dbgInfo)
{
  char strSide0[25];
  char strSide1[25];
  char strNObj[25];
  cv::Point midPt;
  sprintf(strSide0, "Side0: %i", dbgInfo.side0Cnt);
  sprintf(strSide1, "Side1: %i", dbgInfo.side1Cnt);
  sprintf(strNObj, "#objects: %i", dbgInfo.NObjects);

  cv::putText(curImage, strSide0, cv::Point(0, ivHeight/10), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
  cv::putText(curImage, strSide1, cv::Point(ivWidth-ivHeight/3, ivHeight/10), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
  cv::putText(curImage, strNObj, cv::Point(ivWidth-3*ivHeight/4, ivHeight-ivHeight/4), CV_FONT_HERSHEY_SIMPLEX, 0.5, 0);
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
