// Copyright 2020 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "atk_yolov5_object_recognize.h"
#include "chuankou.cpp"
#include <sys/types.h>
#include <dirent.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sys/wait.h>

RK_U32 video_width = 1280;
RK_U32 video_height = 720;
static int disp_width = 1920;
static int disp_height = 1080;

std::queue<cv::Mat> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond;
bool stop_saving = false;
std::thread saver;

const int MAX_FRAMES = 1176454; // 最大允许的图片数量

std::vector<pid_t> audio_pids;
std::mutex audio_mutex;
std::condition_variable audio_cv;
bool audio_playing = false; // 标志位，表示是否正在播放音频

void play_audio(const char *command)
{
  audio_playing = true; // 设置标志位为 true
  pid_t pid = fork();
  if (pid == 0)
  {
    execlp("mpg123", "mpg123", "-a", "sysdefault:CARD=rockchiprk809co", command, (char *)NULL);
    _exit(1);
  }
  else if (pid > 0)
  {
    std::unique_lock<std::mutex> lock(audio_mutex);
    // audio_pids.push_back(pid);
    waitpid(pid, NULL, 0); // 等待子进程结束
    audio_playing = false; // 子进程结束后，设置标志位为 false
  }
  else
  {
    perror("fork");
  }
}

void terminate_audio()
{
  std::unique_lock<std::mutex> lock(audio_mutex);
  for (pid_t pid : audio_pids)
  {
    kill(pid, SIGTERM);
  }
  audio_pids.clear();
  audio_playing = true; // 确保标志位在终止音频后被重置
}

int get_initial_frame_count(const std::string &directory)
{
  int max_count = 0;
  DIR *dir = opendir(directory.c_str());
  if (dir)
  {
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
      std::string filename = entry->d_name;
      if (filename.rfind("frame_", 0) == 0 && filename.find(".jpg") != std::string::npos)
      {
        std::string number_str = filename.substr(6, filename.find(".jpg") - 6);
        try
        {
          int number = std::stoi(number_str);
          if (number > max_count)
          {
            max_count = number;
          }
        }
        catch (...)
        {
          // 忽略无法转换的文件名
        }
      }
    }
    closedir(dir);
  }
  return max_count + 1;
}

void save_frame_thread(const std::string &directory)
{
  int frame_count = get_initial_frame_count(directory);
  while (!stop_saving || !frame_queue.empty())
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cond.wait(lock, []
                    { return !frame_queue.empty() || stop_saving; });

    while (!frame_queue.empty())
    {
      cv::Mat frame = frame_queue.front();
      frame_queue.pop();
      lock.unlock();
      // 计算要保存的文件名
      int current_frame_index = frame_count % MAX_FRAMES;
      std::string filename = directory + "/frame_" + std::to_string(current_frame_index) + ".jpg";
      cv::imwrite(filename, frame);
      frame_count++;

      lock.lock();
    }
  }
}
int init_uart4(const char *device);
int send_message(int uart4_fd, const char *message);

static void sigterm_handler(int sig)
{
  fprintf(stderr, "signal %d\n", sig);
  quit = true;
}
int main(int argc, char *argv[])
{

  static const char *default_model_path = "/data/yolov5_exp2.rknn";
  const char *model_path = default_model_path;

  // 初始化帧计数
  const std::string save_directory = "/mnt/sdcard";

  // // 启动保存线程
  std::thread saver(save_frame_thread, save_directory);

  if (argc > 1)
  {
    model_path = argv[1]; // 使用提供的第一个参数作为 model_path
  }

  RK_CHAR *pDeviceName_01 = "/dev/video0";
  RK_CHAR *pDeviceName_02 = "/dev/video1"; // 第二个摄像头设备名称
  RK_CHAR *pDeviceName_03 = "/dev/video6";
  RK_CHAR *pDeviceName_04 = "/dev/video7";
  RK_CHAR *pcDevNode = "/dev/dri/card0";
  RK_S32 s32CamId_01 = 0;
  RK_S32 s32CamId_02 = 1;
  RK_U32 u32BufCnt = 3;
  RK_U32 fps = 30;
  int ret;

  // 初始化系统
  RK_MPI_SYS_Init();

  // 配置第一个摄像头
  VI_CHN_ATTR_S vi_chn_attr_01;
  memset(&vi_chn_attr_01, 0, sizeof(vi_chn_attr_01));
  vi_chn_attr_01.pcVideoNode = pDeviceName_01;
  vi_chn_attr_01.u32BufCnt = u32BufCnt;
  vi_chn_attr_01.u32Width = video_width;
  vi_chn_attr_01.u32Height = video_height;
  vi_chn_attr_01.enPixFmt = IMAGE_TYPE_UYVY422;
  vi_chn_attr_01.enBufType = VI_CHN_BUF_TYPE_MMAP;
  vi_chn_attr_01.enWorkMode = VI_WORK_MODE_NORMAL;
  ret = RK_MPI_VI_SetChnAttr(0, s32CamId_01, &vi_chn_attr_01);
  ret |= RK_MPI_VI_EnableChn(0, s32CamId_01);
  if (ret)
  {
    printf("ERROR: create VI[0:0] error! ret=%d\n", ret);
    return 0;
  }

  // 配置第二个摄像头
  VI_CHN_ATTR_S vi_chn_attr_02;
  memset(&vi_chn_attr_02, 0, sizeof(vi_chn_attr_02));
  vi_chn_attr_02.pcVideoNode = pDeviceName_02;
  vi_chn_attr_02.u32BufCnt = u32BufCnt;
  vi_chn_attr_02.u32Width = video_width;
  vi_chn_attr_02.u32Height = video_height;
  vi_chn_attr_02.enPixFmt = IMAGE_TYPE_UYVY422;
  vi_chn_attr_02.enBufType = VI_CHN_BUF_TYPE_MMAP;
  vi_chn_attr_02.enWorkMode = VI_WORK_MODE_NORMAL;
  ret = RK_MPI_VI_SetChnAttr(0, s32CamId_02, &vi_chn_attr_02);
  ret |= RK_MPI_VI_EnableChn(0, s32CamId_02);
  if (ret)
  {
    printf("ERROR: create VI[1:0] error! ret=%d\n", ret);
    return 0;
  }

  VI_CHN_ATTR_S vi_chn_attr_03;
  memset(&vi_chn_attr_03, 0, sizeof(vi_chn_attr_03));
  vi_chn_attr_03.pcVideoNode = pDeviceName_03;
  vi_chn_attr_03.u32BufCnt = u32BufCnt;
  vi_chn_attr_03.u32Width = video_width;
  vi_chn_attr_03.u32Height = video_height;
  vi_chn_attr_03.enPixFmt = IMAGE_TYPE_UYVY422;
  vi_chn_attr_03.enBufType = VI_CHN_BUF_TYPE_MMAP;
  vi_chn_attr_03.enWorkMode = VI_WORK_MODE_NORMAL;
  ret = RK_MPI_VI_SetChnAttr(0, 2, &vi_chn_attr_03);
  ret |= RK_MPI_VI_EnableChn(0, 2);
  if (ret)
  {
    printf("ERROR: create VI[1:0] error! ret=%d\n", ret);
    return 0;
  }

  VI_CHN_ATTR_S vi_chn_attr_04;
  memset(&vi_chn_attr_04, 0, sizeof(vi_chn_attr_04));
  vi_chn_attr_04.pcVideoNode = pDeviceName_04;
  vi_chn_attr_04.u32BufCnt = u32BufCnt;
  vi_chn_attr_04.u32Width = video_width;
  vi_chn_attr_04.u32Height = video_height;
  vi_chn_attr_04.enPixFmt = IMAGE_TYPE_UYVY422;
  vi_chn_attr_04.enBufType = VI_CHN_BUF_TYPE_MMAP;
  vi_chn_attr_04.enWorkMode = VI_WORK_MODE_NORMAL;
  ret = RK_MPI_VI_SetChnAttr(0, 3, &vi_chn_attr_04);
  ret |= RK_MPI_VI_EnableChn(0, 3);
  if (ret)
  {
    printf("ERROR: create VI[1:0] error! ret=%d\n", ret);
    return 0;
  }

  RK_U8 u8LayoutHor = 2; // 横向2个通道
  RK_U8 u8LayoutVer = 2; // 纵向2个通道
  // Chn layout: u8LayoutHor x u8LayoutVer
  RK_U32 u32ChnWidth = disp_width / u8LayoutHor;
  RK_U32 u32ChnHeight = disp_height / u8LayoutVer;
  RK_U16 u16ChnCnt = u8LayoutHor * u8LayoutVer;
  RK_U16 u16ChnIdx = 0;

  VMIX_DEV_INFO_S stDevInfo;
  stDevInfo.enImgType = IMAGE_TYPE_RGB888;
  stDevInfo.u16ChnCnt = u16ChnCnt;
  stDevInfo.u16Fps = fps;
  stDevInfo.u32ImgWidth = disp_width;
  stDevInfo.u32ImgHeight = disp_height;
  stDevInfo.bEnBufPool = RK_TRUE;
  stDevInfo.u16BufPoolCnt = 3;

  for (RK_U8 u8VerIdx = 0; u8VerIdx < u8LayoutVer; u8VerIdx++)
  {
    for (RK_U8 u8HorIdx = 0; u8HorIdx < u8LayoutHor; u8HorIdx++)
    {
      stDevInfo.stChnInfo[u16ChnIdx].stInRect.s32X = 0;
      stDevInfo.stChnInfo[u16ChnIdx].stInRect.s32Y = 0;
      stDevInfo.stChnInfo[u16ChnIdx].stInRect.u32Width = video_width;
      stDevInfo.stChnInfo[u16ChnIdx].stInRect.u32Height = video_height;
      stDevInfo.stChnInfo[u16ChnIdx].stOutRect.s32X = u8HorIdx * u32ChnWidth;
      stDevInfo.stChnInfo[u16ChnIdx].stOutRect.s32Y = u8VerIdx * u32ChnHeight;
      stDevInfo.stChnInfo[u16ChnIdx].stOutRect.u32Width = u32ChnWidth;
      stDevInfo.stChnInfo[u16ChnIdx].stOutRect.u32Height = u32ChnHeight;
      printf("#CHN[%d]:IN<0,0,%d,%d> --> Out<%d,%d,%d,%d>\n", u16ChnIdx,
             video_width, video_height, u8HorIdx * u32ChnWidth,
             u8VerIdx * u32ChnHeight, u32ChnWidth, u32ChnHeight);
      u16ChnIdx++;
    }
  }

  ret = RK_MPI_VMIX_CreateDev(0, &stDevInfo);
  if (ret)
  {
    printf("Init VMIX device failed! ret=%d\n", ret);
    return -1;
  }

  for (RK_U16 i = 0; i < stDevInfo.u16ChnCnt; i++)
  {
    ret = RK_MPI_VMIX_EnableChn(0, i);
    if (ret)
    {
      printf("Enable VM[0]:Chn[%d] failed! ret=%d\n", i, ret);
      return -1;
    }
  }

  RGA_ATTR_S stRgaAttr_01;
  memset(&stRgaAttr_01, 0, sizeof(stRgaAttr_01));
  stRgaAttr_01.bEnBufPool = RK_TRUE;
  stRgaAttr_01.u16BufPoolCnt = 3;
  stRgaAttr_01.u16Rotaion = 0;
  stRgaAttr_01.stImgIn.u32X = 0;
  stRgaAttr_01.stImgIn.u32Y = 0;
  stRgaAttr_01.stImgIn.imgType = IMAGE_TYPE_UYVY422;
  stRgaAttr_01.stImgIn.u32Width = disp_width;
  stRgaAttr_01.stImgIn.u32Height = disp_height;
  stRgaAttr_01.stImgIn.u32HorStride = disp_width;
  stRgaAttr_01.stImgIn.u32VirStride = disp_height;
  stRgaAttr_01.stImgOut.u32X = 0;
  stRgaAttr_01.stImgOut.u32Y = 0;
  stRgaAttr_01.stImgOut.imgType = IMAGE_TYPE_RGB888;
  stRgaAttr_01.stImgOut.u32Width = disp_width;      //
  stRgaAttr_01.stImgOut.u32Height = disp_height;    // 输出高度为屏幕高度
  stRgaAttr_01.stImgOut.u32HorStride = disp_width;  // 输出水平跨度
  stRgaAttr_01.stImgOut.u32VirStride = disp_height; // 输出垂直跨度

  ret = RK_MPI_RGA_CreateChn(s32CamId_01, &stRgaAttr_01);
  if (ret)
  {
    printf("ERROR: create RGA[0:0] falied! ret=%d\n", ret);
    return -1;
  }

  VO_CHN_ATTR_S stVoAttr_01 = {0};
  stVoAttr_01.pcDevNode = "/dev/dri/card0";
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

  // 绑定第一个摄像头的VI通道到RGA，再绑定RGA到VO
  MPP_CHN_S stSrcChn;
  MPP_CHN_S stDestChn;

  for (RK_U16 i = 0; i < u16ChnCnt; i++)
  {
    printf("#Bind VI[%u] to VM[0]:Chn[%u]....\n", i, i);
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = i;
    stSrcChn.s32ChnId = i;
    stDestChn.enModId = RK_ID_VMIX;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = i;
    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret)
    {
      printf("Bind vi[%u] to vmix[0]:Chn[%u] failed! ret=%d\n", i, i, ret);
      return -1;
    }
    //    RK_MPI_SYS_DumpChn(RK_ID_VMIX);
    //    getchar();
  }

  printf("#Bind VMX[0] to RGA....\n");
  stSrcChn.enModId = RK_ID_VMIX;
  stSrcChn.s32DevId = 0;
  stSrcChn.s32ChnId = 0; // invalid
  stDestChn.enModId = RK_ID_RGA;
  stDestChn.s32DevId = 0;
  stDestChn.s32ChnId = 0;
  ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
  if (ret)
  {
    printf("Bind VMX[0] to vo[0] failed! ret=%d\n", ret);
    return -1;
  }

  // printf("#Bind VMX[0] to RGA....\n");
  // stSrcChn.enModId = RK_ID_RGA;
  // stSrcChn.s32DevId = 0;
  // stSrcChn.s32ChnId = 0; // invalid
  // stDestChn.enModId = RK_ID_VO;
  // stDestChn.s32DevId = 0;
  // stDestChn.s32ChnId = 0;
  // ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
  // if (ret)
  // {
  //   printf("Bind VMX[0] to vo[0] failed! ret=%d\n", ret);
  //   return -1;
  // }

  pthread_t rkmedia_rknn_tidp;
  pthread_t recv_thread;
  pthread_create(&rkmedia_rknn_tidp, NULL, rkmedia_rknn_thread, (void *)model_path);

  printf("%s initial finish\n", __func__);
  signal(SIGINT, sigterm_handler);
  while (!quit)
  {
    usleep(500000);
  }

  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop_saving = true;
    queue_cond.notify_one();
  }
  saver.join(); // 等待保存线程完成
  // 在程序结束前终止所有音频播放进程
  terminate_audio();

  printf("%s exit!\n", __func__);
  printf("#UnBind VMX[0] to VO[0]....\n");
  stSrcChn.enModId = RK_ID_VMIX;
  stSrcChn.s32DevId = 0;
  stSrcChn.s32ChnId = 0; // invalid
  stDestChn.enModId = RK_ID_RGA;
  stDestChn.s32DevId = 0;
  stDestChn.s32ChnId = 0;
  ret = RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
  if (ret)
  {
    printf("UnBind VMX[0] to vo[0] failed! ret=%d\n", ret);
    return -1;
  }

  for (RK_U16 i = 0; i < u16ChnCnt; i++)
  {
    printf("#UnBind VI[%u] to VM[0]:Chn[%u]....\n", i, i);
    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = i;
    stSrcChn.s32ChnId = i;
    stDestChn.enModId = RK_ID_VMIX;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = i;
    ret = RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
    if (ret)
    {
      printf("UnBind vi[%u] to vmix[0]:Chn[%u] failed! ret=%d\n", i, i, ret);
      return -1;
    }
  }

  RK_MPI_VO_DestroyChn(0);
  RK_MPI_VO_DestroyChn(1);

  for (RK_U16 i = 0; i < u16ChnCnt; i++)
  {
    ret = RK_MPI_VMIX_DisableChn(0, i);
    if (ret)
      printf("Disable VIMX[0]:Chn[%u] failed! ret=%d\n", i, ret);
  }

  ret = RK_MPI_VMIX_DestroyDev(0);
  if (ret)
  {
    printf("DeInit VIMX[0] failed! ret=%d\n", ret);
  }

  RK_MPI_VI_DisableChn(0, 0);
  RK_MPI_VI_DisableChn(1, 1);

  return 0;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  unsigned int model_len = ftell(fp);
  unsigned char *model = (unsigned char *)malloc(model_len);
  fseek(fp, 0, SEEK_SET);

  if (model_len != fread(model, 1, model_len, fp))
  {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;

  if (fp)
  {
    fclose(fp);
  }
  return model;
}

static void printRKNNTensor(rknn_tensor_attr *attr)
{
  printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
         "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
         attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
         attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

int rgb24_resize(unsigned char *input_rgb, unsigned char *output_rgb,
                 int width, int height, int outwidth, int outheight)
{
  rga_buffer_t src = wrapbuffer_virtualaddr(input_rgb, width, height, RK_FORMAT_RGB_888);
  rga_buffer_t dst = wrapbuffer_virtualaddr(output_rgb, outwidth, outheight, RK_FORMAT_RGB_888);
  rga_buffer_t pat = {0};
  im_rect src_rect = {0, 0, width, height};
  im_rect dst_rect = {0, 0, outwidth, outheight};
  im_rect pat_rect = {0};
  IM_STATUS STATUS = improcess(src, dst, pat, src_rect, dst_rect, pat_rect, 0);
  if (STATUS != IM_STATUS_SUCCESS)
  {
    printf("imcrop failed: %s\n", imStrError(STATUS));
    return -1;
  }
  return 0;
}

void *rkmedia_rknn_thread(void *args)
{

  int frame_counter[4] = {0, 0, 0, 0}; // 为每个摄像头创建单独的 frame_counter

  pthread_detach(pthread_self());

  int ret;
  rknn_context ctx;
  int model_len = 0;
  unsigned char *model;
  const char *model_path = (const char *)args;

  // Load RKNN Model
  printf("Loading model ...\n");
  model = load_model(model_path, &model_len);
  ret = rknn_init(&ctx, model, model_len, 0);
  if (ret < 0)
  {
    printf("rknn_init fail! ret=%d\n", ret);
    return NULL;
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail! ret=%d\n", ret);
    return NULL;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  // print input tensor
  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (unsigned int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return NULL;
    }
    printRKNNTensor(&(input_attrs[i]));
  }

  // print output tensor
  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (unsigned int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return NULL;
    }
    printRKNNTensor(&(output_attrs[i]));
  }

  // get model's input image width and height
  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    width = input_attrs[0].dims[0];
    height = input_attrs[0].dims[1];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    width = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
  }

  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);
  static int uart4_fd = -1;
  if (uart4_fd == -1)
  {
    const char *uart_device = "/dev/ttyS4";
    uart4_fd = init_uart4(uart_device);
  }

  
  pthread_t recv_thread;
  int uart4_fd_copy = uart4_fd;
  if (pthread_create(&recv_thread, NULL, receive_thread, &uart4_fd_copy) != 0)
  {
    printf("Failed to create receive thread \n");
    return NULL;
  }

  int frame_counters = 0;
  while (!quit)
  {
    MEDIA_BUFFER src_mb = NULL;
    src_mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, -1);
    if (!src_mb)
    {
      printf("ERROR: RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
      break;
    }

    frame_counters++;
    if (frame_counters % 2 != 0)
    {
      RK_MPI_MB_ReleaseBuffer(src_mb);
      continue; // 跳过检测，每隔三帧检测一次
    }
    /*================================================================================
      =========================使用drm拷贝，可以使用如下代码===========================
      ================================================================================*/
    rga_context rga_ctx;
    drm_context drm_ctx;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

    // DRM alloc buffer
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    void *drm_buf = NULL;

    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, disp_width, disp_height, channel * 8, &buf_fd, &handle, &actual_size);
    memcpy(drm_buf, (uint8_t *)RK_MPI_MB_GetPtr(src_mb), disp_width * disp_height * channel);
    void *resize_buf = malloc(height * width * channel);
    // init rga context
    RGA_init(&rga_ctx);
    img_resize_slow(&rga_ctx, drm_buf, disp_width, disp_height, resize_buf, width, height);
    uint32_t input_model_image_size = width * height * channel;

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_model_image_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = resize_buf;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
      printf("ERROR: rknn_inputs_set fail! ret=%d\n", ret);
      return NULL;
    }
    free(resize_buf);
    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    drm_deinit(&drm_ctx, drm_fd);
    RGA_deinit(&rga_ctx);

    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
      printf("ERROR: rknn_run fail! ret=%d\n", ret);
      return NULL;
    }

    // Get Output
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
      outputs[i].want_float = 0;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
      printf("ERROR: rknn_outputs_get fail! ret=%d\n", ret);
      return NULL;
    }

    detect_result_group_t detect_result_group;
    memset(&detect_result_group, 0, sizeof(detect_result_group));
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
      out_scales.push_back(output_attrs[i].scale);
      out_zps.push_back(output_attrs[i].zp);
    }

    const float vis_threshold = 0.3;  // 用于可视化结果，通常用户自定义
    const float nms_threshold = 0.45; // 默认值 nms 抑制重叠检测框
    const float conf_threshold = 0.3; // 默认值为0.25，通常设置在0.3到0.5之间 置信度
    float scale_w = (float)width / disp_width;
    float scale_h = (float)height / disp_height;

    post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, height, width,
                 conf_threshold, nms_threshold, vis_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    int half_height = disp_height / 2;
    int half_width = disp_width / 2;
    int cam_width = disp_width / 2;
    int cam_height = disp_height / 2;

    std::ifstream file("/data/line_positions.txt");
    if (!file.is_open())
    {
      std::cerr << "Failed to open file." << std::endl;
    }

    std::vector<int> line1_y_positions;
    std::vector<int> line2_y_positions;
    int line1_y, line2_y;

    while (file >> line1_y >> line2_y)
    {
      line1_y_positions.push_back(line1_y);
      line2_y_positions.push_back(line2_y);
    }

    file.close();
    bool person_detected = false;
    // Draw Objects

    using namespace cv;

    Rect cam_areas[4] = {
        Rect(0, 0, cam_width, cam_height),
        Rect(cam_width, 0, cam_width, cam_height),
        Rect(0, cam_height, cam_width, cam_height),
        Rect(cam_width, cam_height, cam_width, cam_height)};

    bool person_detected_in_cam[4] = {false, false, false, false};
    Mat orig_img = Mat(disp_height, disp_width, CV_8UC3, RK_MPI_MB_GetPtr(src_mb));
    Mat img_to_save = orig_img.clone();

    for (int i = 0; i < detect_result_group.count; i++)
    {
      detect_result_t *det_result = &(detect_result_group.results[i]);

      if (strcmp(det_result->name, "person") == 0 && det_result->prop > vis_threshold)
      {
        int left = det_result->box.left;
        int top = det_result->box.top;
        int right = det_result->box.right;
        int bottom = det_result->box.bottom;
        int w = right - left;
        int h = bottom - top;

        if (left < 0)
          left = 0;
        if (top < 0)
          top = 0;
        if (left + w > disp_width)
          w = disp_width - left;
        if (top + h > disp_height)
          h = disp_height - top;

        rectangle(orig_img, Point(left, top), Point(right, bottom), Scalar(0, 255, 255), 5, 8, 0);
        std::ostringstream text_stream;
        text_stream << det_result->name << " " << std::fixed << std::setprecision(2) << det_result->prop;
        std::string text = text_stream.str();
        putText(orig_img, text, Point(left, top - 16), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 255), 2, 8, 0);


        for (int cam = 0; cam < 4; cam++)
        {
          if ((left >= cam_areas[cam].x && right <= cam_areas[cam].x + cam_areas[cam].width) &&
              (top >= cam_areas[cam].y && bottom <= cam_areas[cam].y + cam_areas[cam].height))
          {
            person_detected_in_cam[cam] = true;

            Mat cam_img = img_to_save(cam_areas[cam]);
            {
              std::unique_lock<std::mutex> lock(queue_mutex);
              frame_queue.push(cam_img);
              queue_cond.notify_one();
            }
            if (++frame_counter[cam] >= 60)
            {
              frame_counter[cam] = 0;
              {
                std::unique_lock<std::mutex> lock(queue_mutex);
                frame_queue.push(cam_img); // 保存对应摄像头的图像
                queue_cond.notify_one();
              }
            }
          }
        }

        for (int cam = 0; cam < 4; cam++)
        {
          int cam_left = (cam % 2) * cam_width;
          int cam_top = (cam / 2) * cam_height;
          int cam_right = cam_left + cam_width;
          int cam_bottom = cam_top + cam_height;

          // 添加调试信息
          if (left >= cam_left && right <= cam_right && top >= cam_top && bottom <= cam_bottom)
          {
            int line1_y = line1_y_positions[cam];
            int line2_y = line2_y_positions[cam];
            int line3_y = line1_y_positions[cam + 4];
            int line4_y = line2_y_positions[cam + 4];

            // 绘制蓝线
            line(orig_img, Point(cam_left, cam_top + line1_y), Point(cam_left + cam_width, cam_top + line1_y), Scalar(0, 255, 255), 2, 8, 0);
            line(orig_img, Point(cam_left, cam_top + line2_y), Point(cam_left + cam_width, cam_top + line2_y), Scalar(0, 0, 255), 2, 8, 0);
            line(orig_img, Point(cam_left + cam_width / 4.1, cam_top + line3_y), Point(cam_left + cam_width / 4, cam_top + line4_y), Scalar(255, 0, 0), 2, 8, 0);
            line(orig_img, Point(cam_left + cam_width / 1.3, cam_top + line4_y), Point(cam_left + cam_width / 1.29, cam_top + line3_y), Scalar(255, 0, 0), 2, 8, 0);

            double slope_left_line = static_cast<double>(line4_y - line3_y) / ((cam_width / 4) - (cam_width / 4.1));
            double slope_right_line = static_cast<double>(line3_y - line4_y) / ((cam_width / 1.29) - (cam_width / 1.3));

            if (left < cam_left)
              left = cam_left;
            if (right > cam_right)
              right = cam_right - 1;
            if (top < cam_top)
              top = cam_top;
            if (bottom > cam_bottom)
              bottom = cam_bottom - 1;

            if (left >= cam_left && right <= cam_right && top >= cam_top && bottom <= cam_bottom)
            {
              int right_top_y_on_left_line = slope_left_line * (right - (cam_left + cam_width / 4.1)) + (cam_top + line3_y);
              int left_top_y_on_right_line = slope_right_line * (left - (cam_left + cam_width / 1.3)) + (cam_top + line4_y);

              bool right_top_below_left_line = top > right_top_y_on_left_line;
              bool left_top_below_right_line = top > left_top_y_on_right_line;

              if (right_top_below_left_line || left_top_below_right_line)
              {
                continue;
              }

              if (bottom > cam_top + line1_y && bottom <= cam_top + line2_y)
              {
                char warning_message[100];
                if (cam >= 0 && cam < 1)
                {
                  snprintf(warning_message, sizeof(warning_message), "EF010000000001001B017B226C656674223A227761726E696E67227D000738", cam + 1);
                }
                else if (cam >= 1 && cam < 2)
                {
                  snprintf(warning_message, sizeof(warning_message), "EF010000000001001B017B227269676874223A227761726E696E67227D000738", cam + 1);
                }
                else
                {
                  snprintf(warning_message, sizeof(warning_message), "EF010000000001001B017B22646D6F642D72656172223A227761726E696E67227D004538", cam + 1);
                }
                send_hex_message(uart4_fd, warning_message);
              }

              if (bottom > cam_top + line2_y)
              {
                char message[100];
                if (cam < 2)
                {
                  snprintf(message, sizeof(message), "EF010000000001001B017B22646D6F642D66726F6E74223A2264616E676572227D000738", cam + 1);
                  // if (!audio_playing)
                  // {
                  //   audio_playing = true;
                  //   std::string audio_file = "waring.mp3";
                  //   std::thread audio_thread(play_audio, audio_file.c_str());
                  //   audio_thread.detach();
                  // }
                }
                else
                {
                  snprintf(message, sizeof(message), "EF010000000001001A017B22646D6F642D72656172223A2264616E676572227D00B566", cam + 1);
                  // if (!audio_playing)
                  // {
                  //   audio_playing = true;
                  //   std::string audio_file = "waring.mp3";
                  //   std::thread audio_thread(play_audio, audio_file.c_str());
                  //   audio_thread.detach();
                  // }
                }
                send_hex_message(uart4_fd, message);
              }
            }
          }
        }
      }
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    rga_buffer_t src, dst;
    MB_IMAGE_INFO_S dst_ImageInfo = {(RK_U32)disp_width, (RK_U32)disp_height, (RK_U32)disp_width,
                                     (RK_U32)disp_height, IMAGE_TYPE_RGB888};
    MEDIA_BUFFER dst_mb = RK_MPI_MB_CreateImageBuffer(&dst_ImageInfo, RK_TRUE, 0);
    dst = wrapbuffer_fd(RK_MPI_MB_GetFD(dst_mb), disp_width, disp_height, RK_FORMAT_RGB_888);
    src = wrapbuffer_fd(RK_MPI_MB_GetFD(src_mb), disp_width, disp_height, RK_FORMAT_RGB_888);

    im_rect src_rect, dst_rect;
    src_rect = {0, 0, disp_width, disp_height};
    dst_rect = {0};
    ret = imcheck(src, dst, src_rect, dst_rect, IM_CROP);
    if (IM_STATUS_NOERROR != ret)
    {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      break;
    }

    IM_STATUS CROP_STATUS = imcrop(src, dst, src_rect);
    if (CROP_STATUS != IM_STATUS_SUCCESS)
    {
      printf("ERROR: imcrop failed: %s\n", imStrError(CROP_STATUS));
    }

    RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, dst_mb);
    RK_MPI_MB_ReleaseBuffer(dst_mb);
    RK_MPI_MB_ReleaseBuffer(src_mb);

    src_mb = NULL;
    dst_mb = NULL;
  }

  if (uart4_fd != -1)
  {
    close(uart4_fd);
  }
  if (model)
  {
    delete model;
    model = NULL;
  }
  rknn_destroy(ctx);
  return NULL;
}
