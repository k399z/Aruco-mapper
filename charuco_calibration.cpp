// CharUco board calibration program
// Detects CharUco board and saves camera calibration to file

#include "atk_yolov5_object_recognize.h"
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <rga/im2d.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <pthread.h>
#include <unistd.h>

using namespace cv;
using namespace std;

// CharUco board parameters
const int squaresX = 5;
const int squaresY = 7;
const float squareLength = 0.04f;  // 40mm
const float markerLength = 0.02f;  // 20mm

// Resolution settings (same as main program)
RK_U32 video_width = 1280;
RK_U32 video_height = 720;
static int disp_width = 1920;
static int disp_height = 1080;

// Calibration state
static volatile bool quit = false;
static volatile bool captureFrame = false;
static vector<vector<Point2f>> allCharucoCorners;
static vector<vector<int>> allCharucoIds;
static Size imageSize;
static pthread_mutex_t calibMutex = PTHREAD_MUTEX_INITIALIZER;

string getCurrentDateTime() {
    time_t now = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return string(buf);
}

void *calibration_thread(void *args)
{
    const char* outputFile = (const char*)args;
    
    // Initialize ArUco dictionary and CharUco board
    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::CharucoParameters> charucoParams = cv::aruco::CharucoParameters::create();
    
    detectorParams->adaptiveThreshWinSizeMin = 9;
    detectorParams->adaptiveThreshWinSizeMax = 23;
    detectorParams->adaptiveThreshWinSizeStep = 10;
    
    cv::aruco::CharucoBoard board(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
    cv::aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    int framesCaptured = 0;

    cout << "CharUco Board Calibration Program" << endl;
    cout << "=================================" << endl;
    cout << "Display will show on screen" << endl;
    cout << "Press Ctrl+C and type 'c' to capture frame" << endl;
    cout << "Press Ctrl+C and type 'q' to finish calibration" << endl;
    cout << "Collect at least 10 frames from different angles" << endl;

    while (!quit)
    {
        // Get frame from RGA channel
        MEDIA_BUFFER mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, 50);
        if (!mb)
            continue;

        Mat orig_img = Mat(disp_height, disp_width, CV_8UC3, RK_MPI_MB_GetPtr(mb));
        Mat displayFrame;
        orig_img.copyTo(displayFrame);
        
        if (imageSize.width == 0) {
            imageSize = orig_img.size();
        }

        // Detect CharUco board
        vector<int> charucoIds;
        vector<Point2f> charucoCorners;
        detector.detectBoard(orig_img, charucoCorners, charucoIds);

        // Draw detected corners
        if (!charucoIds.empty()) {
            cv::aruco::drawDetectedCornersCharuco(displayFrame, charucoCorners, charucoIds, Scalar(0, 255, 0));
            putText(displayFrame, "Board detected! Press 'c' to capture", 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        } else {
            putText(displayFrame, "No board detected", 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        putText(displayFrame, cv::format("Frames captured: %d (need 10+)", framesCaptured), 
                Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);

        // Handle capture command
        if (captureFrame) {
            captureFrame = false;
            if (!charucoIds.empty() && charucoIds.size() >= 4) {
                pthread_mutex_lock(&calibMutex);
                allCharucoCorners.push_back(charucoCorners);
                allCharucoIds.push_back(charucoIds);
                framesCaptured++;
                pthread_mutex_unlock(&calibMutex);
                
                cout << "Frame " << framesCaptured << " captured with " 
                     << charucoIds.size() << " corners" << endl;
                     
                // Visual feedback
                putText(displayFrame, "CAPTURED!", 
                        Point(disp_width/2 - 100, disp_height/2), 
                        FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 255), 3);
            } else {
                cout << "Not enough corners detected. Need at least 4." << endl;
            }
        }

        // Setup RGA buffers for display
        rga_buffer_t src, vo_dst;
        src = wrapbuffer_fd(RK_MPI_MB_GetFD(mb), disp_width, disp_height, RK_FORMAT_RGB_888);

        MB_IMAGE_INFO_S vo_ImageInfo = {(RK_U32)disp_width, (RK_U32)disp_height,
                                        (RK_U32)disp_width, (RK_U32)disp_height,
                                        IMAGE_TYPE_RGB888};
        MEDIA_BUFFER vo_mb = RK_MPI_MB_CreateImageBuffer(&vo_ImageInfo, RK_TRUE, 0);
        vo_dst = wrapbuffer_fd(RK_MPI_MB_GetFD(vo_mb), disp_width, disp_height, RK_FORMAT_RGB_888);

        // Copy drawn frame back
        displayFrame.copyTo(orig_img);

        // Send to display
        im_rect crop_rect = {0, 0, disp_width, disp_height};
        im_rect vo_rect = {0, 0, disp_width, disp_height};
        improcess(src, vo_dst, {}, crop_rect, vo_rect, {}, IM_SYNC);
        RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, vo_mb);

        RK_MPI_MB_ReleaseBuffer(vo_mb);
        RK_MPI_MB_ReleaseBuffer(mb);
    }

    // Compute calibration
    pthread_mutex_lock(&calibMutex);
    int finalFrameCount = framesCaptured;
    pthread_mutex_unlock(&calibMutex);

    if (finalFrameCount < 3) {
        cerr << "Error: Need at least 3 frames for calibration. Only got " 
             << finalFrameCount << endl;
        return nullptr;
    }

    cout << "\nComputing calibration from " << finalFrameCount << " frames..." << endl;

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    
    double repError = cv::aruco::calibrateCameraCharuco(
        allCharucoCorners, allCharucoIds, board, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs
    );

    cout << "Calibration complete!" << endl;
    cout << "Reprojection error: " << repError << endl;

    // Save calibration to file
    FileStorage fs(outputFile, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Error: Cannot open file for writing: " << outputFile << endl;
        return nullptr;
    }

    fs << "calibration_time" << getCurrentDateTime();
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << repError;
    fs << "frames_used" << finalFrameCount;
    fs.release();

    cout << "Calibration saved to: " << outputFile << endl;
    cout << "\nCamera Matrix:\n" << cameraMatrix << endl;
    cout << "\nDistortion Coefficients:\n" << distCoeffs << endl;

    return nullptr;
}

int main(int argc, char** argv)
{
    // Default output file
    const char* outputFile = "camera_calibration.yml";
    if (argc > 1) {
        outputFile = argv[1];
    }

    int ret;
    RK_CHAR *pcDevNode = "/dev/dri/card0";
    RK_U32 u32BufCnt = 3;

    RK_MPI_SYS_Init();

    // Setup VI (Video Input)
    VI_CHN_ATTR_S vi_chn_attr;
    memset(&vi_chn_attr, 0, sizeof(vi_chn_attr));
    vi_chn_attr.pcVideoNode = "/dev/video0";
    vi_chn_attr.u32BufCnt = u32BufCnt;
    vi_chn_attr.u32Width = video_width;
    vi_chn_attr.u32Height = video_height;
    vi_chn_attr.enPixFmt = IMAGE_TYPE_UYVY422;
    vi_chn_attr.enBufType = VI_CHN_BUF_TYPE_MMAP;
    vi_chn_attr.enWorkMode = VI_WORK_MODE_NORMAL;
    RK_MPI_VI_SetChnAttr(0, 0, &vi_chn_attr);
    RK_MPI_VI_EnableChn(0, 0);

    // Setup RGA (Raster Graphic Acceleration)
    RGA_ATTR_S stRgaAttr;
    memset(&stRgaAttr, 0, sizeof(stRgaAttr));
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 3;
    stRgaAttr.u16Rotaion = 0;
    stRgaAttr.stImgIn.u32X = 0;
    stRgaAttr.stImgIn.u32Y = 0;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_UYVY422;
    stRgaAttr.stImgIn.u32Width = video_width;
    stRgaAttr.stImgIn.u32Height = video_height;
    stRgaAttr.stImgIn.u32HorStride = video_width;
    stRgaAttr.stImgIn.u32VirStride = video_height;
    stRgaAttr.stImgOut.u32X = 0;
    stRgaAttr.stImgOut.u32Y = 0;
    stRgaAttr.stImgOut.imgType = IMAGE_TYPE_RGB888;
    stRgaAttr.stImgOut.u32Width = disp_width;
    stRgaAttr.stImgOut.u32Height = disp_height;
    stRgaAttr.stImgOut.u32HorStride = disp_width;
    stRgaAttr.stImgOut.u32VirStride = disp_height;
    ret = RK_MPI_RGA_CreateChn(0, &stRgaAttr);

    // Setup VO (Video Output)
    VO_CHN_ATTR_S stVoAttr = {0};
    stVoAttr.pcDevNode = pcDevNode;
    stVoAttr.emPlaneType = VO_PLANE_OVERLAY;
    stVoAttr.enImgType = IMAGE_TYPE_RGB888;
    stVoAttr.u16Zpos = 0;
    stVoAttr.stImgRect.s32X = 0;
    stVoAttr.stImgRect.s32Y = 0;
    stVoAttr.stImgRect.u32Width = disp_width;
    stVoAttr.stImgRect.u32Height = disp_height;
    stVoAttr.stDispRect.s32X = 0;
    stVoAttr.stDispRect.s32Y = 0;
    stVoAttr.stDispRect.u32Width = disp_width;
    stVoAttr.stDispRect.u32Height = disp_height;
    ret = RK_MPI_VO_CreateChn(0, &stVoAttr);
    if (ret) {
        printf("ERROR: create VO[0] failed! ret=%d\n", ret);
        return -1;
    }

    // Bind VI to RGA
    MPP_CHN_S stSrcChn, stDestChn;
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = 0;
    stSrcChn.s32ChnId = 0;
    stDestChn.enModId = RK_ID_RGA;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = 0;
    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret) {
        printf("ERROR: bind VI to RGA failed! ret=%d\n", ret);
        return -1;
    }

    // Start calibration thread
    pthread_t calib_thread_id;
    pthread_create(&calib_thread_id, NULL, calibration_thread, (void*)outputFile);

    printf("Calibration started. Commands:\n");
    printf("  c - capture frame\n");
    printf("  q - quit and compute calibration\n");

    // Simple command loop
    char cmd;
    while (!quit) {
        if (read(STDIN_FILENO, &cmd, 1) > 0) {
            if (cmd == 'c' || cmd == 'C') {
                captureFrame = true;
            } else if (cmd == 'q' || cmd == 'Q') {
                quit = true;
            }
        }
        usleep(100000);
    }

    pthread_join(calib_thread_id, NULL);

    // Cleanup
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = 0;
    stSrcChn.s32ChnId = 0;
    stDestChn.enModId = RK_ID_RGA;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = 0;
    RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);

    RK_MPI_VO_DestroyChn(0);
    RK_MPI_RGA_DestroyChn(0);
    RK_MPI_VI_DisableChn(0, 0);

    return 0;
}
