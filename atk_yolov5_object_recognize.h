#ifndef _ATK_YOLOV5_OBJECT_RECOGNIZE_H
#define _ATK_YOLOV5_OBJECT_RECOGNIZE_H

#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>
#include <dlfcn.h>

#define _BASETSD_H

#include "im2d.h"
#include "rga.h"
#include "drm_func.h"
#include "rga_func.h"
#include <rga/im2d.h> 
#include "rknn_api.h"
#include "rkmedia_api.h"

#include "rkmedia_venc.h"
#include "sample_common.h"
#include "opencv2/opencv.hpp"
#include <opencv2/freetype.hpp>





void switch_camera_settings(int mode);

int rgb24_resize(unsigned char *input_rgb, unsigned char *output_rgb, int width,int height, int outwidth, int outheight);

static unsigned char *load_model(const char *filename, int *model_size);

static void printRKNNTensor(rknn_tensor_attr *attr);

void *rkmedia_rknn_thread(void *args);

void *rkmedia_rknn_thread2(void *args);

#endif
