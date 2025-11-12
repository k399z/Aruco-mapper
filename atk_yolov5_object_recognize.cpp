// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "atk_yolov5_object_recognize.h"
#include "marker.h"
#include "edge.h"
#include "robot.h"
#include "graph.h"
#include <sys/types.h>
#include <dirent.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sys/wait.h>
#include <rga/im2d.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <unordered_map>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>

#include <deque>
#include <chrono>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Global quit flag
static volatile bool quit = false;

// Camera calibration data structure
struct CameraCalibration {
  cv::Mat cameraMatrix;      // 3x3 intrinsic camera matrix
  cv::Mat distCoeffs;        // Distortion coefficients
  bool isCalibrated;
  
  CameraCalibration() : isCalibrated(false) {}
  
  // Load calibration from file
  bool loadFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Error: Cannot open calibration file: " << filename << std::endl;
      return false;
    }
    
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    
    if (cameraMatrix.empty() || distCoeffs.empty()) {
      std::cerr << "Error: Invalid calibration data in file" << std::endl;
      return false;
    }
    
    isCalibrated = true;
    std::cout << "Calibration loaded successfully from: " << filename << std::endl;
    return true;
  }
};

// Global camera calibration
static CameraCalibration g_camCalib;

double calculateYaw(const std::vector<cv::Point2f>& corners) {
  // corners[0] = top-left, [1] = top-right, [2] = bottom-right, [3] = bottom-left
  
  // s_a: average of horizontal edge lengths
  double top_edge = cv::norm(corners[1] - corners[0]);
  double bottom_edge = cv::norm(corners[2] - corners[3]);
  double s_a = (top_edge + bottom_edge) / 2.0;
  
  // s_i: vertical edge lengths
  double left_edge = cv::norm(corners[3] - corners[0]);
  double right_edge = cv::norm(corners[2] - corners[1]);
  double s_i = (left_edge + right_edge) / 2.0;
  
  // ψ_z = arccos(s_a / s_i)
  double yaw = std::acos(std::min(1.0, s_a / s_i));
  
  // Determine sign based on horizontal skew
  if (top_edge > bottom_edge) {
    yaw = -yaw;
  }
  
  return yaw;
}

// Calculate spatial edge between two markers using Law of Cosines
Edge calculateEdgeBetweenMarkers(const Marker& marker_a,
                                  const Marker& marker_b,
                                  double distance_ab) {
  cv::Point2f pos_a = marker_a.getPosition();
  cv::Point2f pos_b = marker_b.getPosition();
  
  // Phase angle: φ_ab = atan2(Δy, Δx)
  double delta_x = pos_b.x - pos_a.x;
  double delta_y = pos_b.y - pos_a.y;
  double phi_ab = std::atan2(delta_y, delta_x);
  
  // Angular differences from Law of Cosines
  // θ_ab = arccos((d_a² + d_ab² - d_b²) / (2·d_a·d_ab))
  double d_a = marker_a.getDistance();
  double d_b = marker_b.getDistance();
  
  double numerator_ab = d_a*d_a + distance_ab*distance_ab - d_b*d_b;
  double denominator_ab = 2.0 * d_a * distance_ab;
  double theta_ab = std::acos(numerator_ab / denominator_ab);
  
  double numerator_ba = d_b*d_b + distance_ab*distance_ab - d_a*d_a;
  double denominator_ba = 2.0 * d_b * distance_ab;
  double theta_ba = std::acos(numerator_ba / denominator_ba);
  
  return Edge(marker_a.getId(), marker_b.getId(), 
              distance_ab, theta_ab, theta_ba, phi_ab);
}

RK_U32 video_width = 640;
RK_U32 video_height = 480;
RK_U32 shrunken_width = 640;
RK_U32 shrunken_height = 360;
static int disp_width = 1920;
static int disp_height = 1080;

// Simple timing + fps stats (same spirit as main.cpp)
static inline double nowMs() {
  using clock = std::chrono::steady_clock;
  auto t = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(t).count();
}
struct FpsStats {
  double avgMs = 0.0;
  double fpsStart = nowMs();
  double avgFps = 0.0;
  double fps1sec = 0.0;
    double updateAvgMs(double frameMs) { avgMs = 0.98 * avgMs + 0.02 * frameMs; return avgMs; }
  double tickFps() {
    double n = nowMs();
    if (n - fpsStart > 1000.0) { fpsStart = n; avgFps = 0.7 * avgFps + 0.3 * fps1sec; fps1sec = 0.0; }
    fps1sec += 1.0;
    return avgFps;
  }
};

void *rkmedia_rknn_thread(void *args)
{
  FpsStats stats;
  Graph markerGraph;  // Graph to store all detected markers
  
  // Load camera calibration
  if (!g_camCalib.loadFromFile("camera_calibration.yml")) {
    std::cerr << "Warning: Running without camera calibration" << std::endl;
    std::cerr << "Run charuco_calibration program first to create calibration file" << std::endl;
  }
  
  // Use only DICT_6X6_50
  auto dict6x6_50 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);

  // Tune detection parameters
  cv::Ptr<cv::aruco::DetectorParameters> detParams = cv::aruco::DetectorParameters::create();
  detParams->adaptiveThreshWinSizeMin = 9;
  detParams->adaptiveThreshWinSizeMax = 23;
  detParams->adaptiveThreshWinSizeStep = 10;
  detParams->minMarkerPerimeterRate = 0.01f;
  detParams->maxMarkerPerimeterRate = 4.0f;
  detParams->polygonalApproxAccuracyRate = 0.05;

  // Remove CharUco board detection code - calibration is now pre-computed
  
  while (!quit)
  {
    using namespace cv;
    // 从 RGA 通道抓一帧；使用更长的超时时间以处理相机启动延迟
    MEDIA_BUFFER mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, 1000 /*ms*/);
    if (!mb) {
      // Camera not ready yet, wait a bit
      usleep(100000); // 100ms
      continue;
    }

    double start = nowMs();

    Mat orig_img = Mat(disp_height, disp_width, CV_8UC3, RK_MPI_MB_GetPtr(mb));

    // Build a shrunken image for faster ArUco detection
    Mat shrunken(shrunken_height, shrunken_width, CV_8UC3);
    cv::resize(orig_img, shrunken, Size(shrunken_width, shrunken_height), 0, 0, cv::INTER_AREA);

    rectangle(orig_img, Point(35, 35), Point(125, 125), Scalar(0, 255, 255), 5);
    int text_y = (20 - 16 > 0) ? (20 - 16) : (20 + 16);
    putText(orig_img, "test", Point(35, text_y),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 2);

    rga_buffer_t src, vo_dst; // removed unused venc_dst

    // 源图像缓冲区
    src = wrapbuffer_fd(RK_MPI_MB_GetFD(mb), disp_width, disp_height, RK_FORMAT_RGB_888);

    MB_IMAGE_INFO_S vo_ImageInfo = {(RK_U32)disp_width, (RK_U32)disp_height,
                                    (RK_U32)disp_width, (RK_U32)disp_height,
                                    IMAGE_TYPE_RGB888};
    MEDIA_BUFFER vo_mb = RK_MPI_MB_CreateImageBuffer(&vo_ImageInfo, RK_TRUE, 0);
    vo_dst = wrapbuffer_fd(RK_MPI_MB_GetFD(vo_mb), disp_width, disp_height, RK_FORMAT_RGB_888);


    // Buckets
    std::vector<int> correctIds; correctIds.reserve(64);
    std::vector<std::vector<cv::Point2f>> correctCorners; correctCorners.reserve(64);
    std::vector<std::string> correctLabels; correctLabels.reserve(64);
    std::vector<int> wrongIds; wrongIds.reserve(64);
    std::vector<std::vector<cv::Point2f>> wrongCorners; wrongCorners.reserve(64);
    std::vector<std::string> wrongLabels; wrongLabels.reserve(64);


    // Detect ArUco on the shrunken RGB frame
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejectedCorners;
    cv::aruco::detectMarkers(shrunken, dict6x6_50, corners, ids, detParams, rejectedCorners);

    // Scale factor to project shrunken detections back to original image size
    const float scaleX = static_cast<float>(disp_width)  / static_cast<float>(shrunken_width);
    const float scaleY = static_cast<float>(disp_height) / static_cast<float>(shrunken_height);

    if (!ids.empty()) {
      for (size_t k = 0; k < ids.size(); ++k) {
        // Scale corners back to full-res coordinates
        std::vector<cv::Point2f> scaledPts;
        scaledPts.reserve(corners[k].size());
        for (const auto& p : corners[k]) {
          scaledPts.emplace_back(p.x * scaleX, p.y * scaleY);
        }

        // Calculate center position for the marker
        cv::Point2f center(0, 0);
        for (const auto& pt : scaledPts) {
          center += pt;
        }
        center *= (1.0f / scaledPts.size());

        // Create and add marker to the graph

        markerGraph.addMarker(Marker(ids[k], center));
        

      }
    }

    // Draw correct (allowed) markers
    if (!correctIds.empty()) {
      cv::aruco::drawDetectedMarkers(orig_img, correctCorners, correctIds, cv::Scalar(153, 0, 255));
      for (size_t i = 0; i < correctCorners.size(); ++i) {
        const auto& pts = correctCorners[i];
        std::vector<cv::Point> poly; poly.reserve(pts.size());
        for (const auto& p : pts) poly.push_back(p);
        const cv::Point* ptsArr = poly.data();
        int npts = static_cast<int>(poly.size());
        cv::polylines(orig_img, &ptsArr, &npts, 1, true, cv::Scalar(153, 0, 255), 6, cv::LINE_AA);
        cv::Point2f c(0,0); for (const auto& p : pts) c += p; c *= 0.25f;
        cv::putText(orig_img, correctLabels[i], c + cv::Point2f(-20, -10),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(153, 0, 255), 1.5, cv::LINE_AA);
      }
    }


    double dur = nowMs() - start;
    std::string statsText = cv::format("avg %.2f ms  fps %.1f  det %d",
                                       stats.updateAvgMs(dur), stats.tickFps(), (int)correctLabels.size());
    cv::putText(orig_img, statsText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

    // Send to VO (copy after drawing)
    im_rect first_crop_rect = {0, 0, disp_width, disp_height};
    im_rect vo_rect = {0, 0, disp_width, disp_height};
    improcess(src, vo_dst, {}, first_crop_rect, vo_rect, {}, IM_SYNC);
    RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, vo_mb);

    RK_MPI_MB_ReleaseBuffer(vo_mb);
    RK_MPI_MB_ReleaseBuffer(mb);
    mb = NULL;
  }

  return nullptr;
}

int main(int argc, char *argv[])
{
  int ret;
  RK_CHAR *pcDevNode = "/dev/dri/card0";
  RK_S32 s32CamId_01 = 0;
  RK_S32 s32CamId_02 = 1;
  RK_U32 u32BufCnt = 3;
  RK_U32 fps = 20;

  // Silence potential unused warnings (in case of -Werror)
  (void)s32CamId_01;
  (void)s32CamId_02;
  (void)fps;

  RK_MPI_SYS_Init();

  VI_CHN_ATTR_S vi_chn_attr_01;
  memset(&vi_chn_attr_01, 0, sizeof(vi_chn_attr_01));
  vi_chn_attr_01.pcVideoNode = "/dev/video25";
  vi_chn_attr_01.u32BufCnt = u32BufCnt;
  vi_chn_attr_01.u32Width = video_width;
  vi_chn_attr_01.u32Height = video_height;
  vi_chn_attr_01.enPixFmt = IMAGE_TYPE_YUYV422;
  vi_chn_attr_01.enBufType = VI_CHN_BUF_TYPE_MMAP;
  vi_chn_attr_01.enWorkMode = VI_WORK_MODE_NORMAL;
  RK_MPI_VI_SetChnAttr(0, 0, &vi_chn_attr_01); // 设置第一个摄像头
  RK_MPI_VI_EnableChn(0, 0);                   // 启用第一个摄像头

  RGA_ATTR_S stRgaAttr_01;
  memset(&stRgaAttr_01, 0, sizeof(stRgaAttr_01));
  stRgaAttr_01.bEnBufPool = RK_TRUE;
  stRgaAttr_01.u16BufPoolCnt = 3;
  stRgaAttr_01.u16Rotaion = 0;
  stRgaAttr_01.stImgIn.u32X = 0;
  stRgaAttr_01.stImgIn.u32Y = 0;
  stRgaAttr_01.stImgIn.imgType = IMAGE_TYPE_YUYV422;
  stRgaAttr_01.stImgIn.u32Width = video_width;
  stRgaAttr_01.stImgIn.u32Height = video_height;
  stRgaAttr_01.stImgIn.u32HorStride = video_width;
  stRgaAttr_01.stImgIn.u32VirStride = video_height;
  stRgaAttr_01.stImgOut.u32X = 0;
  stRgaAttr_01.stImgOut.u32Y = 0;
  stRgaAttr_01.stImgOut.imgType = IMAGE_TYPE_RGB888;
  stRgaAttr_01.stImgOut.u32Width = disp_width;      //
  stRgaAttr_01.stImgOut.u32Height = disp_height;    // 输出高度为屏幕高度
  stRgaAttr_01.stImgOut.u32HorStride = disp_width;  // 输出水平跨度
  stRgaAttr_01.stImgOut.u32VirStride = disp_height; // 输出垂直跨度
  // stRgaAttr_01.enFlip = RGA_FLIP_H;
  ret = RK_MPI_RGA_CreateChn(0, &stRgaAttr_01);

  VO_CHN_ATTR_S stVoAttr_01 = {0};
  stVoAttr_01.pcDevNode = pcDevNode; // use the variable to avoid unused warning
  stVoAttr_01.emPlaneType = VO_PLANE_OVERLAY;
  stVoAttr_01.enImgType = IMAGE_TYPE_RGB888;
  stVoAttr_01.u16Zpos = 0;
  stVoAttr_01.stImgRect.s32X = 0;
  stVoAttr_01.stImgRect.s32Y = 0;
  stVoAttr_01.stImgRect.u32Width = disp_width;
  stVoAttr_01.stImgRect.u32Height = disp_height;
  stVoAttr_01.stDispRect.s32X = 0;
  stVoAttr_01.stDispRect.s32Y = 0;
  stVoAttr_01.stDispRect.u32Width = disp_width;
  stVoAttr_01.stDispRect.u32Height = disp_height;
  ret = RK_MPI_VO_CreateChn(0, &stVoAttr_01); // 注意VO通道号为0
  if (ret)
  {
    printf("ERROR: create VO[0] failed! ret=%d\n", ret);
    return -1;
  }

  MPP_CHN_S stSrcChn;
  MPP_CHN_S stDestChn;
  printf("Bind VI[0:0] to RGA[0:0]....\n");
  stSrcChn.enModId = RK_ID_VI;
  stSrcChn.s32DevId = 0;
  stSrcChn.s32ChnId = 0;
  stDestChn.enModId = RK_ID_RGA;
  stDestChn.s32DevId = 0; // fix: was stSrcChn.s32DevId = 0
  stDestChn.s32ChnId = 0;
  ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
  if (ret)
  {
    printf("ERROR: bind VI[0:0] to RGA[0:0] failed! ret=%d\n", ret);
    return -1;
  }

  // printf("Bind RGA[0:0] to  Vo[0:0]....\n");
  // stSrcChn.enModId = RK_ID_RGA;
  // stSrcChn.s32DevId = 0;
  // stSrcChn.s32ChnId = 0;
  // stDestChn.enModId = RK_ID_VO;
  // stSrcChn.s32DevId = 0;
  // stDestChn.s32ChnId = 0;
  // ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
  // if (ret)
  // {
  //   printf("ERROR: bind VI[0:0] to RGA[0:0] failed! ret=%d\n", ret);
  //   return -1;
  // }

  printf("%s initial finish\n", __func__);
  
  // Wait for camera to start streaming (give it 2 seconds)
  printf("Waiting for camera to start streaming...\n");
  sleep(2);
  
  // Verify camera is producing frames
  printf("Testing camera frame capture...\n");
  for (int i = 0; i < 5; i++) {
    MEDIA_BUFFER test_mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, 1000);
    if (test_mb) {
      printf("Camera test frame %d received successfully\n", i + 1);
      RK_MPI_MB_ReleaseBuffer(test_mb);
      break;
    } else {
      printf("Camera test frame %d timeout, retrying...\n", i + 1);
      sleep(1);
    }
  }
  
  pthread_t rkmedia_rknn_tidp;
  pthread_create(&rkmedia_rknn_tidp, NULL, rkmedia_rknn_thread, NULL);

  while (!quit)
  {
    usleep(500000);
  }

  printf("UnBind VI[0:0] to RGA[0:0]....\n");
  stSrcChn.enModId = RK_ID_VI;
  stSrcChn.s32DevId = 0;
  stSrcChn.s32ChnId = 0;
  stDestChn.enModId = RK_ID_RGA;
  stDestChn.s32DevId = 0; // fix: was stSrcChn.s32DevId = 0
  stDestChn.s32ChnId = 0;
  ret = RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
  if (ret)
  {
    printf("ERROR: unbind VI[0:0] to RGA[0:0] failed! ret=%d\n", ret);
    return -1;
  }

  RK_MPI_VO_DestroyChn(0);
  RK_MPI_RGA_DestroyChn(0);
  RK_MPI_VI_DisableChn(0, 0);

  return 0;
}
