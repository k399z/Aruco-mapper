// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "atk_yolov5_object_recognize.h"
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

// Forward declarations
class Marker;
class Edge;
class Graph;

class Edge
{
public:
  Edge(int marker_a_id, int marker_b_id, double dist,
       double theta_ab, double theta_ba, double phi);
  Edge(const Edge &) = default;
  Edge &operator=(const Edge &) = default;
  ~Edge();
  
  // Edge composition function M(E_ab, E_bc) -> E_ac
  static Edge compose(const Edge& e_ab, const Edge& e_bc);
  
  int getMarkerA() const { return a_id; }
  int getMarkerB() const { return b_id; }
  double getDistance() const { return distance; }
  double getThetaAB() const { return theta_ab; }
  double getThetaBA() const { return theta_ba; }
  double getPhi() const { return phi; }

private:
  int a_id, b_id;
  double distance;
  double theta_ab;
  double theta_ba;
  double phi;
};

Edge::Edge(int marker_a_id, int marker_b_id, double dist,
           double theta_ab_val, double theta_ba_val, double phi_val)
  : a_id(marker_a_id), b_id(marker_b_id), distance(dist),
    theta_ab(theta_ab_val), theta_ba(theta_ba_val), phi(phi_val)
{
}

Edge::~Edge()
{
}

// Implementation of M function for edge composition (multi-hop navigation)
// From paper: E_ac = M(E_ab, E_bc)
Edge Edge::compose(const Edge& e_ab, const Edge& e_bc)
{
  // φ_ac = φ_ab + φ_bc
  double phi_ac = e_ab.phi + e_bc.phi;
  
  // θ_ac = θ_ab + θ_bc - φ_bc
  double theta_ac = e_ab.theta_ab + e_bc.theta_ab - e_bc.phi;
  
  // θ_ca = -(θ_ab + θ_bc - φ_bc)
  double theta_ca = -theta_ac;
  
  // d_ac = √(d_ab² + d_bc² - 2·d_ab·d_bc·cos(φ_bc))
  double d_ac = std::sqrt(
    e_ab.distance * e_ab.distance + 
    e_bc.distance * e_bc.distance - 
    2.0 * e_ab.distance * e_bc.distance * std::cos(e_bc.phi)
  );
  
  return Edge(e_ab.a_id, e_bc.b_id, d_ac, theta_ac, theta_ca, phi_ac);
}

class Marker
{
public:
  Marker();
  Marker(int marker_id, const cv::Point2f& pos);
  void addEdge(const Edge& edge);
  const std::vector<Edge>& getEdges() const { return edges; }
  int getId() const { return id; }
  cv::Point2f getPosition() const { return position; }
  double getTheta() const { return theta; }
  void setPosition(const cv::Point2f& pos) { position = pos; }
  void setTheta(double t) { theta = t; }
  
  // Pose estimation data
  cv::Vec3d rvec;  // Rotation vector from solvePnP
  cv::Vec3d tvec;  // Translation vector from solvePnP
  
  Marker(const Marker &) = default;
  Marker &operator=(const Marker &) = default;
  ~Marker();

private:
  int id;
  double theta;
  cv::Point2f position;
  std::vector<Edge> edges;
};

Marker::Marker() : id(0), theta(0.0), position(0, 0)
{
}

Marker::Marker(int marker_id, const cv::Point2f& pos)
  : id(marker_id), theta(0.0), position(pos)
{
}

void Marker::addEdge(const Edge& edge)
{
  edges.push_back(edge);
}

Marker::~Marker()
{
}

class Graph
{
public:
  Graph();
  Graph(const Graph &) = default;
  Graph &operator=(const Graph &) = default;
  ~Graph();
  
  void addMarker(const Marker& marker);
  void addEdge(const Edge& edge);
  Marker* getMarker(int id);
  const std::unordered_map<int, Marker>& getMarkers() const { return markers; }

private:   
  std::unordered_map<int, Marker> markers;  // O(1) lookup by ArUco ID
  std::vector<Edge> edges;
};

Graph::Graph()
{
}

Graph::~Graph()
{
}

void Graph::addMarker(const Marker& marker)
{
  markers[marker.getId()] = marker;
}

void Graph::addEdge(const Edge& edge)
{
  edges.push_back(edge);
  
  // Add edge to both markers' adjacency lists
  auto it_a = markers.find(edge.getMarkerA());
  auto it_b = markers.find(edge.getMarkerB());
  
  if (it_a != markers.end()) {
    it_a->second.addEdge(edge);
  }
  if (it_b != markers.end()) {
    // Create reverse edge for marker B
    Edge reverse(edge.getMarkerB(), edge.getMarkerA(), 
                 edge.getDistance(), edge.getThetaBA(), 
                 edge.getThetaAB(), -edge.getPhi());
    it_b->second.addEdge(reverse);
  }
}

Marker* Graph::getMarker(int id)
{
  auto it = markers.find(id);
  return (it != markers.end()) ? &(it->second) : nullptr;
}

// Improved yaw calculation from paper equations
double findyaw(const std::vector<cv::Point2f>& corners)
{
  // sa = average of top and bottom edge lengths
  double top_edge = cv::norm(corners[0] - corners[1]);
  double bottom_edge = cv::norm(corners[3] - corners[2]);
  double sa = (top_edge + bottom_edge) / 2.0;
  
  // si1 and si2 = left and right edge lengths
  double si1 = cv::norm(corners[0] - corners[3]);
  double si2 = cv::norm(corners[1] - corners[2]);
  
  // yaw = arcsin(sa / si)
  double yaw = std::asin(sa / si1);
  
  // Determine sign based on which side is shorter
  return (si1 < si2) ? yaw : -yaw;
}

// Calculate spatial relationship between two markers
Edge calculateEdge(const Marker& marker_a, const Marker& marker_b)
{
  cv::Point2f pos_a = marker_a.getPosition();
  cv::Point2f pos_b = marker_b.getPosition();
  
  // Distance: Euclidean distance
  double d_ab = cv::norm(pos_b - pos_a);
  
  // Phase difference: φ = arctan(Δy/Δx)
  double delta_x = pos_b.x - pos_a.x;
  double delta_y = pos_b.y - pos_a.y;
  double phi_ab = std::atan2(delta_y, delta_x);
  
  // Angular differences (simplified - would use actual rotation data)
  double theta_ab = marker_a.getTheta();
  double theta_ba = marker_b.getTheta();
  
  return Edge(marker_a.getId(), marker_b.getId(), d_ab, theta_ab, theta_ba, phi_ab);
}

RK_U32 video_width = 1280;
RK_U32 video_height = 720;
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
    // 从 RGA 通道抓一帧；不阻塞太久，避免卡 pipeline
    MEDIA_BUFFER mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, 50 /*ms*/);
    if (!mb)
      continue;

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
        //add the marker to the graph
        markerGraph.addMarker(ids[k], corners[k]);
        // Scale corners back to full-res coordinates
        std::vector<cv::Point2f> scaledPts;
        scaledPts.reserve(corners[k].size());
        for (const auto& p : corners[k]) {
          scaledPts.emplace_back(p.x * scaleX, p.y * scaleY);
        }

        auto it = kSpecialNames.find(ids[k]);
        if (it == kSpecialNames.end()) {
          wrongCorners.push_back(std::move(scaledPts));
          wrongIds.push_back(ids[k]);
          wrongLabels.emplace_back(cv::format("Wrong_ID_%d", ids[k]));
        } else {
          correctCorners.push_back(std::move(scaledPts));
          correctIds.push_back(ids[k]);
          correctLabels.emplace_back(it->second);
        }
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
  vi_chn_attr_01.pcVideoNode = "/dev/video0";
  vi_chn_attr_01.u32BufCnt = u32BufCnt;
  vi_chn_attr_01.u32Width = video_width;
  vi_chn_attr_01.u32Height = video_height;
  vi_chn_attr_01.enPixFmt = IMAGE_TYPE_UYVY422;
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
  stRgaAttr_01.stImgIn.imgType = IMAGE_TYPE_UYVY422;
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

  pthread_t rkmedia_rknn_tidp;
  pthread_create(&rkmedia_rknn_tidp, NULL, rkmedia_rknn_thread, NULL);

  printf("%s initial finish\n", __func__);

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
