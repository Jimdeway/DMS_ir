

//ncnn 的相依檔 
#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "layer.h"
#include "net.h"
#include "gpu.h"
//
 
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <float.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif

//呼喚agx喇叭的驅動相依檔 
#include <alsa/asoundlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//平行化pthread相依檔
#include <pthread.h>


#include "mobilefacenet.h"
#include "UltraFace.hpp"


using namespace std;
using namespace cv;


#define PCM_DEVICE "default"
#define iconsize 60




char name_text[3][256];
char name_path[256];
bool eye_flag = true;
bool eye_flag2 = true;
bool eye_flag2_n = false;
bool flag_sleep = false;
bool head_f = true;
bool smoke_f = true;
bool phone_f = true;
bool sleep_f = true;
bool sleep2_f = true;
bool sound_f = true;
bool light_f = true;
bool temperature_f = true;
bool flagClose1 = true;
bool flagClose2 = true;
bool flagdraw = false;
static bool soundswitch = true;

char sound_select=4;


double start_t;
double start_t2;
double start_t2_n;
double start_t_recong;
cv::Mat icon_head0;
cv::Mat icon_head1;
cv::Mat icon_smoke0;
cv::Mat icon_smoke1;
cv::Mat icon_phone0;
cv::Mat icon_phone1;
cv::Mat icon_sleep0;
cv::Mat icon_sleep1;
cv::Mat head;
cv::Mat sleep1;
cv::Mat sleep2;
cv::Mat phone;
cv::Mat smoke;
cv::Mat icon_roi;
cv::Mat result_recong;
cv::Mat result;
char smoke_c = 0;
char smoke_nc = 0;
char phone_c = 0;
char phone_nc = 0;
char smoke_c2 = 0;
char smoke_nc2 = 0;

///////////////////////////


double start_t3;

cv::Mat icon_smoke0_c;
cv::Mat icon_smoke1_c;


cv::Mat icon_roi_c;

char smoke_c_c = 0;
char smoke_nc_c = 0;



///////////////////////////////

bool flag55=true ,flag5 = false;

double totalFrame =0;
double totalClose =0;
double perclos =0;


std::vector<cv::Point3d> model_points;


//儲存音檔資訊 
struct args{
	char path[128];
	int sec;

};


//agx 喇叭驅動 
void *play_thread(void* arg)
{
	//關門用旗標
	soundswitch = false;
	
	
  struct args* arggg = (struct args*)arg;
	unsigned int pcm, tmp, dir;
	unsigned int rate, channels, seconds;
	snd_pcm_t *pcm_handle;
	snd_pcm_hw_params_t *params;
	snd_pcm_uframes_t frames;
	char *buff;
	int buff_size, loops;
	//while(1){
  
  
	int file = open(arggg->path,O_RDONLY);
	
	//參數設定
	//依照音檔的格式 
	rate 	 = 16000;
	channels = 1;
	seconds  = 1;

	/* Open the PCM device in playback mode */
	if (pcm = snd_pcm_open(&pcm_handle, PCM_DEVICE,
					SND_PCM_STREAM_PLAYBACK, 0) < 0) 
		printf("ERROR: Can't open \"%s\" PCM device. %s\n",
					PCM_DEVICE, snd_strerror(pcm));

	/* Allocate parameters object and fill it with default values*/
	snd_pcm_hw_params_alloca(&params);

	snd_pcm_hw_params_any(pcm_handle, params);

	/* Set parameters */
	if (pcm = snd_pcm_hw_params_set_access(pcm_handle, params,
					SND_PCM_ACCESS_RW_INTERLEAVED) < 0) 
		printf("ERROR: Can't set interleaved mode. %s\n", snd_strerror(pcm));

	if (pcm = snd_pcm_hw_params_set_format(pcm_handle, params,
						SND_PCM_FORMAT_S16_LE) < 0) 
		printf("ERROR: Can't set format. %s\n", snd_strerror(pcm));

	if (pcm = snd_pcm_hw_params_set_channels(pcm_handle, params, channels) < 0) 
		printf("ERROR: Can't set channels number. %s\n", snd_strerror(pcm));

	if (pcm = snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, 0) < 0) 
		printf("ERROR: Can't set rate. %s\n", snd_strerror(pcm));

	/* Write parameters */
	if (pcm = snd_pcm_hw_params(pcm_handle, params) < 0)
		printf("ERROR: Can't set harware parameters. %s\n", snd_strerror(pcm));

	/* Resume information */
	printf("PCM name: '%s'\n", snd_pcm_name(pcm_handle));

	printf("PCM state: %s\n", snd_pcm_state_name(snd_pcm_state(pcm_handle)));

	snd_pcm_hw_params_get_channels(params, &tmp);
	printf("channels: %i ", tmp);

	if (tmp == 1)
		printf("(mono)\n");
	else if (tmp == 2)
		printf("(stereo)\n");

	snd_pcm_hw_params_get_rate(params, &tmp, 0);
	printf("rate: %d bps\n", tmp);

	printf("seconds: %d\n", seconds);	

	/* Allocate buffer to hold single period */
	snd_pcm_hw_params_get_period_size(params, &frames, 0);

	buff_size = frames * channels * 2 /* 2 -> sample size */;
	buff = (char *) malloc(buff_size);

	snd_pcm_hw_params_get_period_time(params, &tmp, NULL);
	
	for (loops = (seconds * arggg->sec) / tmp; loops > 0; loops--) {

		if (pcm = read(file, buff, buff_size) == 0) {
			printf("Early end of file.\n");
            sound_select=4;
            soundswitch = true;
			return 0;
		}
		
		if (pcm = snd_pcm_writei(pcm_handle, buff, frames) == -EPIPE) {
			printf("XRUN.\n");
			snd_pcm_prepare(pcm_handle);
		} else if (pcm < 0) {
			printf("ERROR. Can't write to PCM device. %s\n", snd_strerror(pcm));
		}
	}
    
	snd_pcm_drain(pcm_handle);
	snd_pcm_close(pcm_handle);
	free(buff);
	
	//聲音選擇初始化&開門用旗標
    sound_select=4;
    soundswitch = true;
	//}
    return 0;
}


//參考https://blog.csdn.net/c20081052/article/details/89479970
//參考https://github.com/yuenshome/yuenshome.github.io/issues/9
//參考https://blog.csdn.net/u014090429/article/details/100762308
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
 
    return  norm(I, shouldBeIdentity) < 1e-6;
 
}
Vec3f rotationMatrixToEulerAngles(Mat &R)
{
 
    assert(isRotationMatrix(R));
 
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
 
}

//視覺化測試歐拉角效果
void plot_pose_cube(cv::Mat& img, float yaw, float pitch, float roll, float tdx, float tdy, float size){
    float p = pitch * CV_PI / 180;
    float y = -(yaw * CV_PI / 180);
    float r = roll * CV_PI / 180;
    int face_x = tdx - 0.50 * size;
    int face_y = tdy - 0.50 * size;

    int x1 = size * (cos(y) * cos(r)) + face_x;
    int y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y;
    int x2 = size * (-cos(y) * sin(r)) + face_x;
    int y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y;
    int x3 = size * (sin(y)) + face_x;
    int y3 = size * (-cos(y) * sin(p)) + face_y;

    //Draw base in red
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x1), int(y1)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x2), int(y2)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)), cv::Scalar(0, 0, 255), 3);
    cv::line(img, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x2 - face_x), int(y1 + y2 - face_y)), cv::Scalar(0, 0, 255), 3);
    //Draw pillars in blue
    cv::line(img, cv::Point(int(face_x), int(face_y)), cv::Point(int(x3), int(y3)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x1), int(y1)), cv::Point(int(x1 + x3 - face_x), int(y1 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x2), int(y2)), cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)), cv::Scalar(255, 0, 0), 2);
    cv::line(img, cv::Point(int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(255, 0, 0), 2);
    //Draw top in green
    cv::line(img, cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
                 cv::Point(int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x1 - face_x), int(y3 + y1 - face_y)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, cv::Point(int(x3), int(y3)), cv::Point(int(x3 + x2 - face_x), int(y3 + y2 - face_y)), cv::Scalar(0, 255, 0), 2);

}






//ncnn的官方範例程式 
//主要用YOLOV5_V62
//#define YOLOV5_V60 1 //YOLOv5 v6.0
#define YOLOV5_V62 1 //YOLOv5 v6.2 export  onnx model method https://github.com/shaoshengsong/yolov5_62_export_ncnn

#if YOLOV5_V60 || YOLOV5_V62
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif //YOLOV5_V60    YOLOV5_V62

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov5;

    //yolov5.opt.use_vulkan_compute = true;
    //yolov5.opt.use_bf16_storage = true;
    

    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#if YOLOV5_V62
    
   		//主要模型位置設定 
        if (yolov5.load_param("./model/y70000-opt.param"))
            exit(-1);
        if (yolov5.load_model("./model/y70000-opt.bin"))
            exit(-1);


    
    
   
#elif YOLOV5_V60
    if (yolov5.load_param("yolov5s_6.0.param"))
        exit(-1);
    if (yolov5.load_model("yolov5s_6.0.bin"))
        exit(-1);
#else
    yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    if (yolov5.load_param("yolov5s.param"))
        exit(-1);
    if (yolov5.load_model("yolov5s.bin"))
        exit(-1);
#endif

    const int target_size = 320;
    float prob_threshold;
    

        prob_threshold = 0.60f;
    

    
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();
   	// ex.set_light_mode(true);
   
   	//設定參與運算的cpu數量
    ex.set_num_threads(4);
    //
    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

#if YOLOV5_V62
    
        ex.extract("642", out);
 
        
#elif YOLOV5_V60
        ex.extract("376", out);
#else
        ex.extract("781", out);
#endif

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
#if YOLOV5_V62
    
    
        ex.extract("656", out);
    
        
#elif YOLOV5_V60
        ex.extract("401", out);
#else
        ex.extract("801", out);
#endif
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}





//畫phone&smoke的bbox 
static void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects)
{
    //rect:bbox資訊(x y(左上頂點) width height) 
    //prob 信心值 
	float temp = 0;
	float smoke_prob=0;
	float phone_prob=0;
	cv::Rect smoke_rect;
	cv::Rect phone_rect;
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        int x1 = obj.rect.x;
        int x2 = obj.rect.x + obj.rect.width;
        int centerX = (x1 + x2) / 2;
//        if(obj.prob> 0.5f && centerX > bgr.cols/2){
		
		
		//判斷bbox在畫面右半邊
        if(centerX > bgr.cols/2){
           // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            //    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            //cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

            switch (obj.label)
            {
                case 0:
                    //cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 255),2);
                break;
                case 1:
                    smoke_prob = obj.prob;
					smoke_rect = obj.rect;
                break;
                case 2:
                    phone_prob = obj.prob;
				    phone_rect = obj.rect;    
                break;
            }


        }
        
    }

	//smoke畫bbox條件判斷 
    if(smoke_prob>0.70)
  	{
		
		smoke_c++;
		if(smoke_c > 5)
  		{
			cv::rectangle(bgr, smoke_rect, cv::Scalar(255, 0, 255),2);		
			smoke_f=false;
            sound_select=3;
			//image(cv::Rect(0, 0, image.cols/2, image.rows)) = image2;
			smoke_nc = 0;
		}
	}
	else
	{
		smoke_nc++;
		if(smoke_nc > 10) smoke_c = 0;
	}

 	
 	//phone畫bbox條件判斷 
  	if(phone_prob>0.78)
  	{
		phone_c++;
		
		if(phone_c > 3)
  		{
			cv::rectangle(bgr, phone_rect, cv::Scalar(255, 255, 0),2);		
			phone_f = false;
            sound_select=2;
			//image(cv::Rect(image.cols/2, 0, image.cols/2, image.rows)) =image2; 
			phone_nc = 0;
		}	
	}
    else
	{
		phone_nc++;
		if(phone_nc > 10) phone_c = 0;
	}

    //cv::imshow("image", image);
    //cv::waitKey(1);
}


//計算閉眼用
float eye_ear(int eye_top1 , int eye_top2, int eye_bottom1, int eye_bottom2, int eye_left, int eye_right){
	
	float A = abs(eye_top1 - eye_bottom1);
	float B = abs(eye_top2 - eye_bottom2);
	float C = 2.0 * abs(eye_left - eye_right);
	float ear = (A+B)/C;
	return ear;

}
float ear(float *out,int landmark_size_width, int landmark_size_height, int x1, int y1) {
	
	float eye_left = eye_ear(out[123]*landmark_size_height+y1, out[127]*landmark_size_height+y1,
							 out[135]*landmark_size_height+y1, out[131]*landmark_size_height+y1,
							 out[120]*landmark_size_width+x1, out[128]*landmark_size_width+x1);

    float eye_right = eye_ear(out[139]*landmark_size_height+y1, out[143]*landmark_size_height+y1,
							  out[151]*landmark_size_height+y1, out[147]*landmark_size_height+y1,
							  out[136]*landmark_size_width+x1, out[144]*landmark_size_width+x1);

	float ear = (eye_left+eye_right)/2.f;

	return ear;
}





//乘客睡眠後處理 
void sleepimg(){

	
	sleep2_f = false;
    sound_f = false;
    light_f = false;
    temperature_f = false;

	if(flag55 ){

			flag55 =false;
			flag5 = true;

		}
	
	


}



//臉部關鍵點(PFLD)模型載入與資訊讀取 
//參考https://github.com/Brightchu/pfld-ncnn
static int landmark_detector(ncnn::Net &pfld, cv::Mat &bgr, float * landmarks,  int img_size = 112)
{
    ncnn::Mat out;
 
    

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, img_size, img_size);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);
    time_t begin_t  = clock();
    
    ncnn::Extractor ex = pfld.create_extractor();
    ex.set_num_threads(4);
    ex.input("input", in);
    ex.extract("output", out);
    
    time_t end_t  = clock();
    double time = (double)(end_t - begin_t) / CLOCKS_PER_SEC / 1;
    std::cout << "timeff = " << time * 1000  << "ms" << std::endl;
    //std::cout << "FPS: " << 1 / time  << std::endl;
    for (int j = 0; j < out.w; j++)
    {
        landmarks[j] = out[j];
//        std::cout << j <<" " << landmarks[j] << std::endl;
    }
    return 0;
}



//駕駛分心與疲勞判斷 
void driveFaceBehavior(float ear ,char turn)
{

	//printf("totalFrame = %f\n",totalFrame);
        //printf("totalClose = %f\n",totalClose);
    //flagdraw = true;

	//疲勞判斷
    if(eye_flag) 
	{
		start_t = ncnn::get_current_time();
		eye_flag=false;
	}
    totalFrame++;
    if(ear < 0.095f  )
    {
        totalClose ++;
    }

    if((double)(ncnn::get_current_time() - start_t)/1000.f> 2.0f)
    {
        eye_flag=true;
        perclos =totalClose/totalFrame;

        printf("totalFrame = %f\n",totalFrame);
        printf("totalClose = %f\n",totalClose);
        printf("perclos = %f\n",totalClose/totalFrame);

        totalClose = 0;
        totalFrame = 0;

	}

    if((perclos) > 0.7f)
    {
        sleep_f = false;
        sound_select=1;
		//flagdraw = false;

    }
    ////////////////////////////////////////
    
    
    //分心判斷
    if(turn == 1){
		head_f = false;
        sound_select=0;
	//	eye_flag=true;
       // flagdraw = false;
	//	sleep_f = true;
   //     totalClose = 0;
    //    totalFrame = 0;
	}

	/*

	if(turn == 1){
		head_f = false;
		eye_flag=true;
	}
	else{

		if(ear < 0.095f  ){
			
			if(eye_flag) 
			{
				start_t = ncnn::get_current_time();
				eye_flag=false;
			}
		 //   printf("ear_time=%f\n",(double)(ncnn::get_current_time() - start_t)/1000.f);
        
			if((double)(ncnn::get_current_time() - start_t)/1000.f> 2.0f){
			
            	sleep_f = false;
			}
        }
		else
		{
			eye_flag=true;
		}
	}
			
	*/
}


//乘客睡眠判斷 
void passengerFaceBehavior(float ear)
{
	
	if(ear < 0.098f){
		
		if(eye_flag2) 
		{
			start_t2 = ncnn::get_current_time();
			eye_flag2=false;
			
		}
	//	printf("ear_time=%f\n",(double)(ncnn::get_current_time() - start_t2)/1000.f);
			
		if((double)(ncnn::get_current_time() - start_t2)/1000.f > 2.0f){
			sleepimg();
			eye_flag2_n=true;
        
		}
		if(flag5){
			sleepimg();
		}
    }
	else
	{
		eye_flag2=true;
		if(eye_flag2_n) 
		{
		start_t2_n = ncnn::get_current_time();
		eye_flag2_n=false;
		
		}
		//printf("%d\n",(ncnn::get_current_time() - start_t2_n)/1000.f);
		if((double)(ncnn::get_current_time() - start_t2_n)/1000.f > 2.0f){
			if(flag5 ){     //reset

				flag5 =false;
				flag55 = true;
				flag_sleep = true;
				
			}
			
		}
		if(flag5){
			sleepimg();
		}
			
	}		
}


//相似度計算
void Euclidean_distance_averag(Recognize &recognize, const std::string &Driver_Information, std::vector<float>&samplefea0,
    std::vector<float>&samplefea1){
    std::string sample0_files = Driver_Information + "/sample0.jpg";
    std::string sample1_files = Driver_Information + "/sample1.jpg";

    //const char *model_path = "model";
    //Recognize recognize(model_path);
    cv::Mat sampleimg0 = cv::imread(sample0_files);
    cv::Mat sampleimg1 = cv::imread(sample0_files);

    //std::vector<float>samplefea0, sampleimg1;

    recognize.start(sampleimg0, samplefea0);
    recognize.start(sampleimg1, samplefea1);

    printf("Euclidean_distance_averag\n");
}

//相似度平均
double calculSimilar_avg(std::vector<float>&croppedfea, std::vector<float>&samplefea0, std::vector<float>&samplefea1){

    double similar0 = calculSimilar(samplefea0, croppedfea);
    double similar1 = calculSimilar(samplefea1, croppedfea);
    
    return (similar0+similar1)/2;
}

//人臉識別
//參考https://github.com/liguiyuan/mobilefacenet-ncnn 
int face_recongition(cv::VideoCapture &mVideoCapture, cv::Mat &frame, Recognize &recognize, int &name_count,UltraFace &ultraface){
    
    char verification_text[256]="Driver is being authenticated";
    std::vector<float> samplefea0, samplefea1;
    char name_path[256];
	bool flag_face = true;
    double end;
    double time;  

    cv::Mat m_roi ;

	
    while(1){
		//判斷是否連續10秒錯誤 
        if(time > 10500.0f){
            cv::imwrite("./image/error.jpg",frame);
            cv::waitKey(500);
            cv::putText(frame,"error",cv::Point(15,40),cv::FONT_HERSHEY_COMPLEX,1.5,cv::Scalar(0,0,255),2);
            imshow("output",frame);
            cv::waitKey(1000);
            return 3;
        }

        mVideoCapture >> frame;
        cv::resize(frame,frame, cv::Size(640,360),0,0,cv::INTER_AREA);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);
		
        
		//計算每個臉 
        for(size_t i = 0; i < face_info.size(); i++)
        {
            auto face = face_info[i];
            //printf("count %d  %f\n",obj.label ,obj.prob);
            //cv::rectangle(frame, Point(face.x1,face.y1),Point(face.x2,face.y2), cv::Scalar(0, 255, 0), 2, 8, 0);
                int x = face.x1;
                int y = face.y1;
                int w = face.x2-face.x1;
                int h = face.y2-face.y1;
                int centerX = (face.x1 + face.x2) / 2;
                
                
                //有效範圍 
                cv::Rect face_roi = cv::Rect(frame.cols/2, 0, frame.cols/2, frame.rows);
                
                if(centerX > frame.cols/2 ){
                    //計時開始 
					if(flag_face){
                            start_t_recong = ncnn::get_current_time();
                            flag_face = false;
                        }
                    
                    //擷取臉部照片 
                    m_roi = frame(cv::Rect(x,y,w,h));
                    //cv::imwrite("a.jpg",m_roi);
                    cv::Mat croppedImage;
                    std::vector<float> croppedfea;
                    m_roi.copyTo(croppedImage);
                    
                    //計算臉特徵值 
                    recognize.start(croppedImage, croppedfea);
                    //double similar = calculSimilar(samplefea1, croppedfea);
                    cv::rectangle(frame, face_roi, cv::Scalar(0, 255, 0), 2, 8, 0);
                    
                    for (int i = 0; i < name_count ;i++){
                        sprintf(name_path, "User-information/%s", name_text[i]);
                        
                        //比對相似度 
                        Euclidean_distance_averag(recognize, name_path, samplefea0, samplefea1);
                        double similar = calculSimilar_avg(croppedfea, samplefea0, samplefea1);
                        printf("similar = %f \n", similar);

                        
                        if(similar>0.82)
                        {
                            return i;
                            break;
                        }
                    }
                    end = ncnn::get_current_time();

                    time = end - start_t_recong;
                    int chose = time/1000;
                    if(chose<10)
                    cv::putText(frame,std::to_string(chose),cv::Point(15,40),cv::FONT_HERSHEY_COMPLEX,1.5,cv::Scalar(0,255,0),2);


                }
                else{
                    cv::rectangle(frame, face_roi, cv::Scalar(0, 0, 255), 2, 8, 0);
                    //計時重新開啟 
                    flag_face = true;
                }

        }
        //cv::putText(frame, verification_text, cv::Point(10, 240), cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 255),2);
		//result_recong = cv::imread("./image/fall.jpg");
		//result_recong = cv::imread("./image/success.jpg");
		//全螢幕設定 
		namedWindow("output",CV_WINDOW_NORMAL);
		setWindowProperty("output", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//
		//frame.copyTo(result_recong(cv::Rect(0,0,frame.cols,frame.rows)));
		imshow("output",frame);
        if(cv::waitKey(1)=='q') break;
    }
	return -1;
}

//////////////////////////////////////////////////////////////////////////////



//排序大小比對 
bool cmpArea(Object lsh, Object rsh) {
    if (lsh.rect.width > rsh.rect.width)
        return false;
    else
        return true;
}




//分心與睡眠判斷
int datection_f(cv::Mat& frame_f, ncnn::Net &pfld, std::vector<Object>& objects_f){

    char objCount = 0;
    
    const int num_landmarks = 106 * 2;
/******************************************
    std::vector<Object> objects_f;
    detect_yolov5(frame_f, objects_f);
    if(flagdraw)
        draw_objects(frame_f, objects_f);
*********************************************/
	std::vector<Object> objects_face;
	
	//臉部大小排序 
	for(size_t i = 0; i < objects_f.size(); i++)
    {
		Object obj = objects_f[i];		
		if(obj.prob> 0.6f && !(obj.label))
        {
			objects_face.emplace_back(obj);
			//cout << obj.prob << endl << obj.label<<endl;
		}
	}
	if(objects_face.size()>1)
		sort(objects_face.begin(), objects_face.end(), cmpArea);
	
	//cout << objects_face.size() << endl;
	
	
    for(size_t i = 0; 1 < objects_face.size() ? (i < 2) :( i < objects_face.size()); i++) 
    {
        Object& obj = objects_face[i];
       // printf("count %d  %f\n",obj.label ,obj.prob);
	//	cout << obj.prob << endl << obj.label<<endl;
        if(obj.prob> 0.5f && !(obj.label))
        {
            int x = obj.rect.x;
            int y = obj.rect.y;
            int w = obj.rect.width;
            int h = obj.rect.height;
            int shift = w * 0.01;
            int centerx = x + w/2;
            
            //臉部BBOX整體位置設定 
            x = (x - shift) < 0 ? 0: x - shift;
            y = (y - shift) < 0 ? 0: y - shift;
            w = w + shift * 2;
            h = h + shift * 2;
            w = (w > frame_f.cols) ? frame_f.cols : w;
            h = (h > frame_f.rows) ? frame_f.rows : h;
            
            //printf("count\n");
            cv::Mat ROI(frame_f, obj.rect);
           
            cv::resize(ROI, ROI, cv::Size(112, 112));
            float landmarks[num_landmarks];
            
            //臉部關鍵點定位 
            landmark_detector(pfld, ROI, landmarks, 112);
            
            
            //2D臉部位置 
            std::vector<cv::Point2d> image_points;
            image_points.push_back( cv::Point2d(landmarks[33 * 2] * w + x, landmarks[33 * 2 + 1] * h + y) );    // LEFT_EYEBROW_LEFT
            image_points.push_back( cv::Point2d(landmarks[38 * 2] * w + x, landmarks[38 * 2 + 1] * h + y) );    // LEFT_EYEBROW_RIGHT
            image_points.push_back( cv::Point2d(landmarks[50 * 2] * w + x, landmarks[50 * 2 + 1] * h + y) );    // RIGHT_EYEBROW_LEFT
            image_points.push_back( cv::Point2d(landmarks[46 * 2] * w + x, landmarks[46 * 2 + 1] * h + y) );    // RIGHT_EYEBROW_RIGHT
            image_points.push_back( cv::Point2d(landmarks[60 * 2] * w + x, landmarks[60 * 2 + 1] * h + y) );    // LEFT_EYE_LEFT
            image_points.push_back( cv::Point2d(landmarks[64 * 2] * w + x, landmarks[64 * 2 + 1] * h + y) );    // LEFT_EYE_RIGHT
            image_points.push_back( cv::Point2d(landmarks[68 * 2] * w + x, landmarks[68 * 2 + 1] * h + y) );    // RIGHT_EYE_LEFT
            image_points.push_back( cv::Point2d(landmarks[72 * 2] * w + x, landmarks[72 * 2 + 1] * h + y) );    // RIGHT_EYE_RIGHT
            image_points.push_back( cv::Point2d(landmarks[55 * 2] * w + x, landmarks[55 * 2 + 1] * h + y) );    // NOSE_LEFT
            image_points.push_back( cv::Point2d(landmarks[59 * 2] * w + x, landmarks[59 * 2 + 1] * h + y) );    // NOSE_RIGHT
            image_points.push_back( cv::Point2d(landmarks[76 * 2] * w + x, landmarks[76 * 2 + 1] * h + y) );    // MOUTH_LEFT
            image_points.push_back( cv::Point2d(landmarks[82 * 2] * w + x, landmarks[82 * 2 + 1] * h + y) );    // MOUTH_RIGHT
            image_points.push_back( cv::Point2d(landmarks[85 * 2] * w + x, landmarks[85 * 2 + 1] * h + y) );    // LOWER_LIP
            image_points.push_back( cv::Point2d(landmarks[16 * 2] * w + x, landmarks[16 * 2 + 1] * h + y) );    // CHIN


			//相機矩陣設定 
            double focal_length = frame_f.cols; // Approximate focal length.
            Point2d center = cv::Point2d(frame_f.cols/2,frame_f.cols/2);
            cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
            cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
            //TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
            
            cv::Mat rotation_vector; // Rotation in axis-angle form
            cv::Mat translation_vector;
            cv::Mat rotation_matrix(3, 3, cv::DataType<double>::type);

            cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);


            cv::Rodrigues(rotation_vector, rotation_matrix);
            cv::Vec3f vec;
            vec=rotationMatrixToEulerAngles(rotation_matrix);


			
            vec[0]*=180.0/3.141592653589793;
            vec[1]*=180.0/3.141592653589793;
            vec[2]*=180.0/3.141592653589793;
            //plot_pose_cube(frame_f,vec[1],-vec[0]+7,vec[2],landmarks[53 * 2] * w + x,landmarks[53 * 2 + 1] * h + y,w);
            std::cout << "Vector: " << vec << std::endl;


			//畫關鍵點 
            for(int i=0; i < num_landmarks / 2;i++){
                cv::circle(frame_f, cv::Point(landmarks[i * 2] * w + x, landmarks[i * 2 + 1] * h + y), 1.7,cv::Scalar(0, 0, 255), -1);
            }
            float ear_value = ear(landmarks, frame_f.cols, frame_f.rows, x, y);
            printf("ear =  %f\n",ear_value);
            
            //判斷臉是在乘客或駕駛 
            if(x < frame_f.cols/2)
            {
                cv::putText(frame_f,"Pitch",cv::Point(15,frame_f.rows-30),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(-vec[0]+7),cv::Point(43,frame_f.rows-30),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            
            
            	cv::putText(frame_f,"Yaw",cv::Point(15,frame_f.rows-20),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(vec[1]),cv::Point(43,frame_f.rows-20),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            
            
            	cv::putText(frame_f,"Roll",cv::Point(15,frame_f.rows-10),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(vec[2]),cv::Point(43,frame_f.rows-10),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
                
                
                passengerFaceBehavior(ear_value);
                if(17<vec[0]){
                //sleep_f = true;
                //eye_flag2=true;	
                }
            }
            else
            {
                
                cv::putText(frame_f,"Pitch",cv::Point(frame_f.cols-100,frame_f.rows-30),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(-vec[0]+7),cv::Point(frame_f.cols-70,frame_f.rows-30),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            
            
            	cv::putText(frame_f,"Yaw",cv::Point(frame_f.cols-100,frame_f.rows-20),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(vec[1]),cv::Point(frame_f.cols-70,frame_f.rows-20),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            
            
            	cv::putText(frame_f,"Roll",cv::Point(frame_f.cols-100,frame_f.rows-10),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
            	cv::putText(frame_f,std::to_string(vec[2]),cv::Point(frame_f.cols-70,frame_f.rows-10),cv::FONT_HERSHEY_COMPLEX,0.35,cv::Scalar(0,255,255),1);
                
                int turn = 0;
                
                //分心判斷
                if(17<vec[0]  || vec[0]<-9 || vec[1]>45 || vec[1]<-26){
                	turn =1;
                	
                
                }
                //printf("%d\n\n",turn);
                driveFaceBehavior(ear_value, turn);
               // printf("ear =  %f\n",ear_value);
            }
            if(17<vec[0]){
                //sleep_f = true;
                //eye_flag=true;	
             }
            //runlandmark2(ROI, frame_f,frame_b, landmark, obj.rect.x, obj.rect.y,false);
            objCount++;
        }
		//printf("%d\n",num_box);    
        //cv::putText(image_1, name_text[name_id], cv::Point(x1, y1-40), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
        //printf("dis = %f\n",dis);
    }

	if(objects_f.size()== FP_ZERO && !objCount){
		cv::putText(frame_f, "No face detected", cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
		
	}
    return 0;
}


int main()
{
    int ix,jx;
    bool flagswit = true;
    int swit;
                
    int name_count = 0;
    int name_id = 0;
    char szTest[256] = {0};
    
    //PFLD模型位置 
    const char * param_path = "./model/ncnncols.param";
    const char * bin_path = "./model/ncnncols.bin";
    
    //臉部檢測模型位置 
    const char * param_path2 = "./model/RFB-320.param";
    const char * bin_path2 = "./model/RFB-320.bin";
    
    //聲音警告用 
    struct args d1[4];
    pthread_t player;
    

    FILE *fp = fopen("User-information/name.txt", "r");
    while(!feof(fp))
    {
        fscanf(fp, "%s\n", szTest);
            //printf("%s\n",szTest);
        strcpy(name_text[name_count],szTest);
        name_count++;
    }
    printf("%s\n",name_text[0]);
    printf("%s\n",name_text[1]);
    printf("%s\n",name_text[2]);
    sprintf(name_path, "User-information/%s", name_text[0]);
    fclose(fp);

    //double start1 = ncnn::get_current_time();
	
	
	
	//聲音警告參數設定 
    snprintf(d1[0].path,sizeof(d1[0].path),"./sound/look.wav");
	  //d1.path ="/home/es912-23/sound/look.wav";
	  d1[0].sec = 1300000;
    
    snprintf(d1[1].path,sizeof(d1[1].path),"./sound/sleep.wav");
	  //d1.path ="/home/es912-23/sound/look.wav";
	  d1[1].sec = 1300000;

    snprintf(d1[2].path,sizeof(d1[2].path),"./sound/phone.wav");
	  //d1.path ="/home/es912-23/sound/look.wav";
	  d1[2].sec = 1600000;

    snprintf(d1[3].path,sizeof(d1[3].path),"./sound/smoke.wav");
	  //d1.path ="/home/es912-23/sound/look.wav";
	  d1[3].sec = 1300000;




	//警告圖像讀入 
    icon_head0 = cv::imread("icon/head0.jpg");
    icon_head1 = cv::imread("icon/head1.jpg");
    icon_smoke0 = cv::imread("icon/smoke0.jpg");
    icon_smoke1 = cv::imread("icon/smoke1.jpg");
    icon_phone0 = cv::imread("icon/phone0.jpg");
    icon_phone1 = cv::imread("icon/phone1.jpg");
    icon_sleep0 = cv::imread("icon/sleep0.jpg");
    icon_sleep1 = cv::imread("icon/sleep1.jpg");
                
                
    //臉部3D位置 
    model_points.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));               // LEFT_EYEBROW_LEFT
    model_points.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));          // LEFT_EYEBROW_RIGHT
    model_points.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));       // RIGHT_EYEBROW_LEFTr
    model_points.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));        // RIGHT_EYEBROW_RIGHT
    model_points.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));      // LEFT_EYE_LEFT
    model_points.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));       // LEFT_EYE_RIGHT
    model_points.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));               // RIGHT_EYE_LEFT
    model_points.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));          // RIGHT_EYE_RIGHT
    model_points.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));       // NOSE_LEFT
    model_points.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));        // NOSE_RIGHT
    model_points.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));      // MOUTH_LEFT
    model_points.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));       // MOUTH_RIGHT
    model_points.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));      // LOWER_LIP
    model_points.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));       // CHIN
                
                

    //警告圖像大小設定 
    cv::resize(icon_head0, icon_head0, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_head1, icon_head1, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_smoke0, icon_smoke0, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_smoke1, icon_smoke1, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_phone0, icon_phone0, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_phone1, icon_phone1, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_sleep0 ,icon_sleep0, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);
    cv::resize(icon_sleep1, icon_sleep1, cv::Size(iconsize, iconsize), 0, 0, cv::INTER_AREA);

    
    //PFLD模型讀入 
    ncnn::Net pfld;
    pfld.load_param(param_path);
    pfld.load_model(bin_path);
    
    //臉部檢測模型讀入 
    UltraFace ultraface(bin_path2, param_path2, 320, 240, 1, 0.7,0.6);

                

                
    //臉部識別模型讀入      
    Recognize recognize("model/");


    cv::Mat frame_f;
    //cv::Mat frame_b;
    cv::Mat frame_recong;
    
    //影像讀入 
    cv::VideoCapture cap(0,CAP_V4L2);
   
	//影像讀入設定 
    cap.set(cv::CAP_PROP_FRAME_WIDTH , 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT , 1080);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    
    //臉部識別 
    //name_id = face_recongition(cap,frame_recong, recognize, name_count,ultraface);

    int c=0,avg=0;
    while (true)
    {
        double start = ncnn::get_current_time();
        bool ret = cap.read(frame_f);
        if(!ret){
            break;
        }


        cv::resize(frame_f,frame_f, cv::Size(640,360),0,0,cv::INTER_AREA);
        //cv::resize(frame_b,frame_b, cv::Size(640,360),0,0,cv::INTER_AREA);


        //警告判斷重設 
        head_f = true;
        smoke_f = true;
        phone_f = true;
        sleep_f = true;
        sleep2_f = true;
        sound_f = true;
        light_f = true;
        temperature_f = true;

		//物件偵測用結構體宣告 
        std::vector<Object> objects_f;
        
        //std::vector<Object> objects_b;
        //printf("%c\n",swit);
        /*
        switch (swit){
            case 'f':
                flagswit = true;
        //detect_yolov5(frame_b, objects_b);
                break;
            case 'b':
                flagswit = false;
                break;

            default:
                break;
        }
        */
        //if(flagswit){
        
        //YOLOV5物件檢測 
        detect_yolov5(frame_f, objects_f);
                        //if(flagdraw)
                        
        //畫phone&smoke的BBOX 
        draw_objects(frame_f, objects_f);
        
        //分心與睡眠判斷 
        datection_f(frame_f, pfld,objects_f);
        //}
        //else{
                        //double start5 = ncnn::get_current_time();
        //datection_b(frame_f, pfld,ultraface);
                 //   double end5 = ncnn::get_current_time();
                 //   double start6 = ncnn::get_current_time();
                         
                  //  double end6 = ncnn::get_current_time();

        //}

        

                    


        //最終圖示警告呈現判斷 
        if(head_f){
            head = icon_head0;
        }
        else{
            head = icon_head1;
        }
        if(sleep_f){
            sleep1 = icon_sleep0;
        }
        else{
            sleep1 = icon_sleep1;
        }
        if(smoke_f){
            smoke = icon_smoke0;
        }
        else{
            smoke = icon_smoke1;
        }
        if(phone_f){
            phone = icon_phone0;
        }
        else{
            phone = icon_phone1;
        }
        if(sleep2_f){
            sleep2=icon_sleep0;
        }
        else{
            sleep2=icon_sleep1;
        }
        /*
        if(flagswit){

        if(sound_f){
            // putText(frame_f, "Music:ON", cv::Point(30, (frame_f.rows)*17/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0),1.5);
        }
        else{
            //putText(frame_f, "Music:OFF", cv::Point(30, (frame_f.rows)*17/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255),1.5);
        }
        if(light_f){
        // putText(frame_f, "Light:ON", cv::Point(30, (frame_f.rows)*18/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0),1.5);
        }
        else{
            //putText(frame_f, "Light:OFF", cv::Point(30, (frame_f.rows)*18/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255),1.5);
        }
        if(temperature_f){
            //putText(frame_f, "Temperature:Normal", cv::Point(30, (frame_f.rows)*19/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0),1.5);
        }
        else{
            //putText(frame_f, "Temperature:Sleep Mode", cv::Point(30, (frame_f.rows)*19/20), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255),1.5);
        }


        }
        */
        //最終聲音警告呈現判斷 
        if(soundswitch){
            switch(sound_select){
                case 0:
                  pthread_create(&player, NULL, play_thread, (void*)&d1[sound_select]);
                break;
                case 1:
                  pthread_create(&player, NULL, play_thread, (void*)&d1[sound_select]);
                break;
                case 2:
                  pthread_create(&player, NULL, play_thread, (void*)&d1[sound_select]);
                break;
                case 3:
                  soundswitch = false;
                  pthread_create(&player, NULL, play_thread, (void*)&d1[sound_select]);
                break;
                default:
                  
                break;
            }
        }


        //printf("%u %u %u %u\n",head_f,sleep_f,smoke_f,phone_f,sleep2_f);
        
        //圖示警告貼到影像上 
		if(objects_f.size()!= NULL){
            //if(flagswit){
                for(ix = 0;ix<60;ix++){
                    for(jx = 0;jx<60;jx++){
                        cv::Vec3b &rgb_head = head.at<cv::Vec3b>(ix, jx);
                        cv::Vec3b &rgb_sleep = sleep1.at<cv::Vec3b>(ix, jx);
                        cv::Vec3b &rgb_smoke = smoke.at<cv::Vec3b>(ix, jx);
                        cv::Vec3b &rgb_phone = phone.at<cv::Vec3b>(ix, jx);
                        cv::Vec3b &rgb_sleep2 = sleep2.at<cv::Vec3b>(ix, jx);
                            
                        if (rgb_head[1]+rgb_head[2] >90 )
                            frame_f.at<cv::Vec3b>(ix+10, jx+frame_f.cols-70) = head.at<cv::Vec3b>(ix, jx);
                        if (rgb_sleep[1]+rgb_sleep[2] >90 )
                            frame_f.at<cv::Vec3b>(ix+90, jx+frame_f.cols-70) = sleep1.at<cv::Vec3b>(ix, jx);
                        if (rgb_smoke[1]+rgb_smoke[2] >90 )
                            frame_f.at<cv::Vec3b>(ix+170, jx+frame_f.cols-70) = smoke.at<cv::Vec3b>(ix, jx);
                        if (rgb_phone[1]+rgb_phone[2]>90 )
                            frame_f.at<cv::Vec3b>(ix+250, jx+frame_f.cols-70) = phone.at<cv::Vec3b>(ix, jx);
                        if (rgb_sleep2[1]+rgb_sleep2[2] >90 )
                            frame_f.at<cv::Vec3b>(ix+10, jx+10) = sleep2.at<cv::Vec3b>(ix, jx);
                    }      
                }
            //}
            /*
            else{
                for(ix = 0;ix<60;ix++){
                    for(jx = 0;jx<60;jx++){
                        cv::Vec3b &rgb_sleep2 = sleep2.at<cv::Vec3b>(ix, jx);
                        if (rgb_sleep2[1]+rgb_sleep2[2] >90 )
                            frame_f.at<cv::Vec3b>(ix+10, jx+10) = sleep2.at<cv::Vec3b>(ix, jx);
                    }
                }       
            }
            */
        }
        else{
        	cv::putText(frame_f, "No face detected", cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
        }

        double end = ncnn::get_current_time();
        double time = end - start;
                    //double time1 = end1 - start1;
                    //double time2 = end2 - start2;
                    //double time3 = end3 - start3;
                    //double time4 = end4 - start4;
                   // double time5 = end5 - start5;
                    //double time6 = end6 - start6;
                   // printf("Time:%7.2f \n",time1);
                   // printf("Time:%7.2f \n",time2);
                  //  printf("Time:%7.2f \n",time3);
                  //  printf("Time:%7.2f \n",time4);
                  //  printf("Time:%7.2f \n",time5);
                  //  printf("Time:%7.2f \n",time6);
            //c++;
            //avg += time;
        printf("Time:%7.2f \n",time);
        cv::putText(frame_f, "FPS:" + std::to_string(int(1/(time*0.001))), cv::Point(frame_f.cols-180, frame_f.rows-20), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
                   // cv::putText(frame_f, name_text[name_id], cv::Point(frame_f.cols*3/5, 30), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255),2);
                    
                    //result=cv::imread("./image/forward_back.jpg");
                    //frame_f.copyTo(result(cv::Rect(result.cols/2, 0, frame_f.cols, frame_f.rows)));
                    //frame_b.copyTo(result(cv::Rect(0, 0, frame_b.cols, frame_b.rows)));
        //全螢幕設定 
		namedWindow("output",CV_WINDOW_NORMAL);
		setWindowProperty("output", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		
					//cv::resize(frame_f,frame_f, cv::Size(640,360),0,0,cv::INTER_AREA);
        cv::imshow("output", frame_f);
        if(cv::waitKey(1)=='q')break; 

    }
                
        printf("Avg Time:%7.2f \n",(double)avg/c);
        //cap0.release();
        cap.release();
        //cap2.release();
    return 0;
}



/*

int main(int argc, char** argv)
{
    cv::Mat frame;
    cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH , 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT , 480);
     while (true)
    {
        cap >> frame;
        std::vector<Object> objects;


        double start = ncnn::get_current_time();
      	
		detect_yolov5(frame, objects);

        draw_objects(frame, objects);

        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Time:%7.2f \n",time);

       
       // cv::imshow("image", frame);
        //if(cv::waitKey(30)=='q')break; 
    }
    

    return 0;
}
*/
