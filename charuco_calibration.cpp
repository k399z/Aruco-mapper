// CharUco board calibration program for RK3588
// Adapted from OpenCV sample for embedded hardware with RGA/VO display
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include <easymedia/rkmedia_api.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <pthread.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

using namespace cv;
using namespace std;

// CharUco board paramete don't change them
static int squaresX = 5;
static int squaresY = 4;
static float squareLength = 0.045f;
static float markerLength = 0.0345f;
static int dictionaryId = (int)aruco::DICT_6X6_50;

// Calibration flags
static bool refindStrategy = true;
static bool fixAspectRatio = false;
static float aspectRatio = 1.0f;
static bool assumeZeroTangentialDistortion = false;
static bool fixPrincipalPointAtCenter = false;

// Resolution settings (match main program to avoid VI config mismatch)
static RK_U32 video_width = 640;
static RK_U32 video_height = 480;
static int disp_width = 1920;
static int disp_height = 1080;

// Calibration state
static volatile bool quit = false;
static volatile bool captureFrame = false;
static bool enableAutoCapture = true;
static int autoIntervalMs = 1500;
static int minCornersForCapture = 12;
static int targetFrames = 10;
static bool autoStopWhenDone = true;
static vector<Mat> allCharucoCorners;
static vector<Mat> allCharucoIds;
static vector<vector<Point2f>> allImagePoints;
static vector<vector<Point3f>> allObjectPoints;
static vector<Mat> allImgs;
static Size imageSize;
static pthread_mutex_t calibMutex = PTHREAD_MUTEX_INITIALIZER;

string getCurrentDateTime() {
    time_t now = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return string(buf);
}

static bool saveCameraParams(const string &filename, Size imgSize, float aspectRatio, int flags,
                             const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);
    if(!fs.isOpened())
        return false;

    fs << "calibration_time" << getCurrentDateTime();
    fs << "image_width" << imgSize.width;
    fs << "image_height" << imgSize.height;

    if(flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if(flags != 0) {
        char buf[1024];
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        fs << "flagsDescription" << string(buf);
    }

    fs << "flags" << flags;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
}

void *calibration_thread(void *args)
{
    const char* outputFile = (const char*)args;
    
    // Setup calibration flags based on parameters
    int calibrationFlags = 0;
    if (fixAspectRatio) {
        calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
    }
    if (assumeZeroTangentialDistortion) {
        calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
    }
    if (fixPrincipalPointAtCenter) {
        calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;
    }
    
    // Initialize ArUco dictionary and CharUco board using new API
    cv::aruco::Dictionary dictionary = 
        cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    
    // Create CharUco board (non-Ptr version)
    cv::aruco::CharucoBoard board(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
    
    // Create detector parameters
    cv::aruco::DetectorParameters detectorParams;
    detectorParams.adaptiveThreshWinSizeMin = 3;
    detectorParams.adaptiveThreshWinSizeMax = 23;
    detectorParams.adaptiveThreshWinSizeStep = 4;
    detectorParams.minMarkerPerimeterRate = 0.03;
    detectorParams.maxMarkerPerimeterRate = 4.0;
    detectorParams.polygonalApproxAccuracyRate = 0.05;
    detectorParams.minCornerDistanceRate = 0.05;
    detectorParams.minDistanceToBorder = 3;
    
    // Create CharUco detector parameters
    cv::aruco::CharucoParameters charucoParams;
    
    // Create CharUco detector
    cv::aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    int framesCaptured = 0;
    auto lastCaptureTime = std::chrono::steady_clock::now() - std::chrono::milliseconds(autoIntervalMs);

    cout << "CharUco Board Calibration Program" << endl;
    cout << "=================================" << endl;
    cout << "Board: " << squaresX << "x" << squaresY << " squares" << endl;
    cout << "Square size: " << squareLength << "m, Marker size: " << markerLength << "m" << endl;
    cout << "Dictionary: " << dictionaryId << endl;
    cout << "Marker/Square length ratio: " << (markerLength / squareLength) << endl;
    cout << "\nDisplay will show on screen" << endl;
    if (enableAutoCapture) {
        cout << "Auto-capture: ON (interval=" << autoIntervalMs << " ms, minCorners="
             << minCornersForCapture << ", targetFrames=" << targetFrames << ")" << endl;
        cout << "Press Enter to capture now; 'q' to finish early" << endl;
    } else {
        cout << "Press Enter to capture frame" << endl;
        cout << "Press 'q' to finish calibration" << endl;
    }
    cout << "Collect at least 4 frames from different angles" << endl;

    int timeoutCount = 0;
    while (!quit)
    {
        // Get frame from RGA channel
        MEDIA_BUFFER mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, 1000);
        if (!mb) {
            if (++timeoutCount % 10 == 0) {
                std::cout << "[calibrator] RGA get buffer timeout (" << timeoutCount << ")" << std::endl;
            }
            usleep(5000);
            continue;
        }
        timeoutCount = 0;

        Mat image = Mat(disp_height, disp_width, CV_8UC3, RK_MPI_MB_GetPtr(mb));
        Mat imageCopy;
        image.copyTo(imageCopy);
        Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        
        if (imageSize.width == 0) {
            imageSize = image.size();
        }

        // Detect CharUco board using new API
        Mat currentCharucoCorners, currentCharucoIds;
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners;
        
        detector.detectBoard(gray, currentCharucoCorners, currentCharucoIds, markerCorners, markerIds);

        // Draw results
        if(markerIds.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
            // Overlay marker IDs for verification
            for (size_t mi = 0; mi < markerCorners.size() && mi < markerIds.size(); ++mi) {
                Point2f c = markerCorners[mi][0];
                cv::putText(imageCopy, cv::format("ID:%d", markerIds[mi]), Point((int)c.x+4, (int)c.y+4),
                            FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,255), 1);
            }
        }

        if(currentCharucoCorners.total() > 0) {
            cv::aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, 
                                                   currentCharucoIds, Scalar(0, 255, 0));
        }

        // Status display
        if(currentCharucoCorners.total() > 3) {
            if (enableAutoCapture) {
                putText(imageCopy, cv::format("Board detected! %d corners. Auto-capturing... (Enter=now)", (int)currentCharucoCorners.total()), 
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            } else {
                putText(imageCopy, cv::format("Board detected! %d corners. Press Enter", (int)currentCharucoCorners.total()), 
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }
        } else if(markerIds.size() > 0) {
            putText(imageCopy, cv::format("Found %d markers (need more)", (int)markerIds.size()), 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 165, 0), 2);
        } else {
            putText(imageCopy, "No markers detected", 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        putText(imageCopy, cv::format("Frames captured: %d (need 4+)", framesCaptured), 
                Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);

        static int dbgCounter = 0;
        if (++dbgCounter % 30 == 0) {
            cout << "[Debug] Detected " << markerIds.size() << " markers"
                 << ", charuco corners: " << currentCharucoCorners.total();
            if (!markerIds.empty()) {
                cout << " | IDs:";
                for (size_t i = 0; i < markerIds.size() && i < 10; ++i) cout << ' ' << markerIds[i];
                if (markerIds.size() > 10) cout << " ...";
            }
            cout << endl;
        }

        // Decide on auto-capture
        bool doAutoCapture = false;
        if (enableAutoCapture && markerIds.size() > 0 && currentCharucoCorners.total() >= (size_t)minCornersForCapture) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCaptureTime).count();
            if (elapsed >= autoIntervalMs) {
                doAutoCapture = true;
                lastCaptureTime = now;
            }
        }

        // Handle capture command or auto-capture
        if (captureFrame || doAutoCapture) {
            captureFrame = false;
            if (markerIds.size() > 0 && currentCharucoCorners.total() > 3) {
                // Match image points to object points
                vector<Point3f> currentObjectPoints;
                vector<Point2f> currentImagePoints;
                board.matchImagePoints(currentCharucoCorners, currentCharucoIds, 
                                      currentObjectPoints, currentImagePoints);
                
                if(currentImagePoints.empty() || currentObjectPoints.empty()) {
                    cout << "Point matching failed, try again." << endl;
                } else {
                    pthread_mutex_lock(&calibMutex);
                    allCharucoCorners.push_back(currentCharucoCorners);
                    allCharucoIds.push_back(currentCharucoIds);
                    allImagePoints.push_back(currentImagePoints);
                    allObjectPoints.push_back(currentObjectPoints);
                    allImgs.push_back(image.clone());
                    framesCaptured++;
                    pthread_mutex_unlock(&calibMutex);
                    
                    cout << "Frame " << framesCaptured << " captured with " 
                         << markerIds.size() << " markers and " 
                         << currentCharucoCorners.total() << " charuco corners" << endl;
                         
                    // Visual feedback
                    putText(imageCopy, "CAPTURED!", 
                            Point(disp_width/2 - 100, disp_height/2), 
                            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 255), 3);

                    if (enableAutoCapture && autoStopWhenDone && targetFrames > 0 && framesCaptured >= targetFrames) {
                        cout << "Target frames reached (" << targetFrames << "). Finishing..." << endl;
                        quit = true;
                    }
                }
            } else if (markerIds.size() > 0) {
                cout << "Insufficient charuco corners detected (" 
                     << currentCharucoCorners.total() << " corners). Need at least 4 corners." << endl;
            } else {
                cout << "No markers detected. Cannot capture." << endl;
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

        // Copy drawn frame back to original buffer
        imageCopy.copyTo(image);

        // Send to display
        im_rect crop_rect = {0, 0, disp_width, disp_height};
        im_rect vo_rect = {0, 0, disp_width, disp_height};
        improcess(src, vo_dst, {}, crop_rect, vo_rect, {}, IM_SYNC);
        RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, vo_mb);

        RK_MPI_MB_ReleaseBuffer(vo_mb);
        RK_MPI_MB_ReleaseBuffer(mb);
    }

    // Compute calibration using collected frames
    pthread_mutex_lock(&calibMutex);
    int finalFrameCount = framesCaptured;
    pthread_mutex_unlock(&calibMutex);

    if (finalFrameCount < 1) {
        cerr << "Error: No frames captured for calibration." << endl;
        return nullptr;
    }

    cout << "\n==================================" << endl;
    cout << "Computing calibration from " << finalFrameCount << " frames..." << endl;

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    
    // Initialize camera matrix if fixing aspect ratio
    if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = aspectRatio;
    }

    if(allImagePoints.size() < 4) {
        cerr << "Error: Not enough valid frames. Need at least 4, got " 
             << allImagePoints.size() << endl;
        return nullptr;
    }

    // Calibrate camera using standard calibrateCamera with matched points
    cout << "\nCalibrating camera with ChArUco corners..." << endl;
    double repError = cv::calibrateCamera(
        allObjectPoints, allImagePoints, imageSize, 
        cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

    cout << "\n==================================" << endl;
    cout << "Calibration complete!" << endl;
    cout << "Reprojection error: " << repError << " pixels" << endl;

    // Save calibration to file
    bool saveOk = saveCameraParams(outputFile, imageSize, aspectRatio, calibrationFlags,
                                   cameraMatrix, distCoeffs, repError);
    if (!saveOk) {
        cerr << "Error: Cannot save output file" << endl;
        return nullptr;
    }

    cout << "Calibration saved to: " << outputFile << endl;
    cout << "\nCamera Matrix:\n" << cameraMatrix << endl;
    cout << "\nDistortion Coefficients:\n" << distCoeffs << endl;

    return nullptr;
}

void *input_thread(void *args)
{
    if (enableAutoCapture) {
        // Still allow 'q' to quit early, but we won't advertise 'c'
    }
    // Set terminal to raw mode for immediate input
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    newt.c_cc[VMIN] = 1;   // Read at least 1 character
    newt.c_cc[VTIME] = 0;  // No timeout
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    
    // Flush any pending input
    tcflush(STDIN_FILENO, TCIFLUSH);

    char ch;
    while (!quit) {
        ssize_t n = read(STDIN_FILENO, &ch, 1);
        if (n > 0) {
            char cmd = tolower(ch);
            if (ch == '\n' || ch == '\r') {
                captureFrame = true;
                cout << "\n[Command] Capture (Enter)" << endl;
            } else if (cmd == 'c' && !enableAutoCapture) {
                captureFrame = true;
                cout << "\n[Command] Capture requested" << endl;
            } else if (cmd == 'q') {
                quit = true;
                cout << "\n[Command] Quit requested" << endl;
            }
        }
    }

    // Restore terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return nullptr;
}

int main(int argc, char** argv)
{
    // Parse command-line arguments
    const char* outputFile = "camera_calibration.yml";
    
    cout << "\nCharUco Board Camera Calibration for RK3588" << endl;
    cout << "============================================" << endl;
    
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-w" && i + 1 < argc) {
            squaresX = atoi(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            squaresY = atoi(argv[++i]);
        } else if (arg == "-sl" && i + 1 < argc) {
            squareLength = atof(argv[++i]);
        } else if (arg == "-ml" && i + 1 < argc) {
            markerLength = atof(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            dictionaryId = atoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "-rs" && i + 1 < argc) {
            refindStrategy = (string(argv[++i]) == "true" || string(argv[i]) == "1");
        } else if (arg == "-zt") {
            assumeZeroTangentialDistortion = true;
        } else if (arg == "-pc") {
            fixPrincipalPointAtCenter = true;
        } else if (arg == "-a" && i + 1 < argc) {
            fixAspectRatio = true;
            aspectRatio = atof(argv[++i]);
        } else if ((arg == "--auto" || arg == "-auto") && i + 1 < argc) {
            string v = argv[++i];
            enableAutoCapture = (v == "1" || v == "true" || v == "on");
        } else if (arg == "-ai" && i + 1 < argc) {
            autoIntervalMs = atoi(argv[++i]);
            if (autoIntervalMs < 0) autoIntervalMs = 0;
        } else if ((arg == "--min-corners" || arg == "-mc") && i + 1 < argc) {
            minCornersForCapture = atoi(argv[++i]);
            if (minCornersForCapture < 4) minCornersForCapture = 4;
        } else if ((arg == "--frames" || arg == "-n") && i + 1 < argc) {
            targetFrames = atoi(argv[++i]);
            if (targetFrames < 0) targetFrames = 0;
        } else if (arg == "--no-autostop") {
            autoStopWhenDone = false;
        } else if (arg == "--help" || arg == "-help") {
            cout << "\nUsage: " << argv[0] << " [options]" << endl;
            cout << "\nOptions:" << endl;
            cout << "  -w <int>      Number of squares in X direction (default: 5)" << endl;
            cout << "  -h <int>      Number of squares in Y direction (default: 7)" << endl;
            cout << "  -sl <float>   Square side length in meters (default: 0.0541)" << endl;
            cout << "  -ml <float>   Marker side length in meters (default: 0.0489)" << endl;
            cout << "  -d <int>      Dictionary ID (default: 0 = DICT_4X4_50)" << endl;
            cout << "  -o <file>     Output calibration file (default: camera_calibration.yml)" << endl;
            cout << "  -rs <bool>    Apply refind strategy (default: true)" << endl;
            cout << "  -zt           Assume zero tangential distortion" << endl;
            cout << "  -pc           Fix principal point at center" << endl;
            cout << "  -a <float>    Fix aspect ratio (fx/fy)" << endl;
            cout << "  --auto <0|1>  Enable/disable auto-capture (default: 1)" << endl;
            cout << "  -ai <ms>      Auto-capture minimum interval in ms (default: 1500)" << endl;
            cout << "  -mc <int>     Minimum ChArUco corners to capture (default: 12)" << endl;
            cout << "  -n <int>      Target number of frames; 0 = unlimited (default: 10)" << endl;
            cout << "  --no-autostop Do not auto-finish when target frames reached" << endl;
            cout << "  --help        Show this help message" << endl;
            cout << "\nDictionary IDs:" << endl;
            cout << "  0=DICT_4X4_50, 1=DICT_4X4_100, 2=DICT_4X4_250, 3=DICT_4X4_1000" << endl;
            cout << "  4=DICT_5X5_50, 5=DICT_5X5_100, 6=DICT_5X5_250, 7=DICT_5X5_1000" << endl;
            cout << "  8=DICT_6X6_50, 9=DICT_6X6_100, 10=DICT_6X6_250, 11=DICT_6X6_1000" << endl;
            cout << "  12=DICT_7X7_50, 13=DICT_7X7_100, 14=DICT_7X7_250, 15=DICT_7X7_1000" << endl;
            cout << "  16=DICT_ARUCO_ORIGINAL" << endl;
            return 0;
        } else {
            // Assume it's the output file if it's the last argument
            if (i == argc - 1 && arg[0] != '-') {
                outputFile = argv[i];
            }
        }
    }
    
    cout << "\nConfiguration:" << endl;
    cout << "  Board: " << squaresX << "x" << squaresY << " squares" << endl;
    cout << "  Square length: " << squareLength << " m" << endl;
    cout << "  Marker length: " << markerLength << " m" << endl;
    cout << "  Dictionary: " << dictionaryId << endl;
    cout << "  Output file: " << outputFile << endl;
    cout << "  Refind strategy: " << (refindStrategy ? "enabled" : "disabled") << endl;
    if (fixAspectRatio) cout << "  Aspect ratio: " << aspectRatio << " (fixed)" << endl;
    if (assumeZeroTangentialDistortion) cout << "  Zero tangential distortion: enabled" << endl;
    if (fixPrincipalPointAtCenter) cout << "  Principal point at center: enabled" << endl;
        cout << "  Auto-capture: " << (enableAutoCapture ? "enabled" : "disabled")
            << " (interval=" << autoIntervalMs << " ms, minCorners=" << minCornersForCapture
            << ", targetFrames=" << targetFrames << (autoStopWhenDone ? ", autostop" : ", no autostop") << ")" << endl;
    cout << endl;

    int ret;
    RK_CHAR *pcDevNode = "/dev/dri/card0";
    RK_U32 u32BufCnt = 6; // increase buffering to reduce starvation

    RK_MPI_SYS_Init();

    // Setup VI (Video Input)
    VI_CHN_ATTR_S vi_chn_attr;
    memset(&vi_chn_attr, 0, sizeof(vi_chn_attr));
    vi_chn_attr.pcVideoNode = "/dev/video25";
    vi_chn_attr.u32BufCnt = u32BufCnt;
    vi_chn_attr.u32Width = video_width;
    vi_chn_attr.u32Height = video_height;
    vi_chn_attr.enPixFmt = IMAGE_TYPE_YUYV422;
    vi_chn_attr.enBufType = VI_CHN_BUF_TYPE_MMAP;
    vi_chn_attr.enWorkMode = VI_WORK_MODE_NORMAL;
    RK_MPI_VI_SetChnAttr(0, 0, &vi_chn_attr);
    RK_MPI_VI_EnableChn(0, 0);

    // Setup RGA (Raster Graphic Acceleration)
    RGA_ATTR_S stRgaAttr;
    memset(&stRgaAttr, 0, sizeof(stRgaAttr));
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 6; // increase buffering to reduce starvation
    stRgaAttr.u16Rotaion = 0;
    stRgaAttr.stImgIn.u32X = 0;
    stRgaAttr.stImgIn.u32Y = 0;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_YUYV422;
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

    // Start input thread
    pthread_t input_thread_id;
    pthread_create(&input_thread_id, NULL, input_thread, NULL);

    printf("\nCalibration started. Commands:\n");
    printf("  Enter - capture current frame\n");
    printf("  q     - quit and compute calibration\n");
    printf("\nPress key directly.\n\n");

    // Wait for threads to complete
    pthread_join(input_thread_id, NULL);
    quit = true; // Ensure calibration thread exits
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
