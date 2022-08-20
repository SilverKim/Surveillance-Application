#include <opencv2/opencv.hpp>

#include "proc.cpp"
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <utility>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <ctime>
#include <list>
#include <set>
#include <stdio.h>
#include <vector>

#include "ctpl_stl.h" //for threadPoolExecutor

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace cv;
using namespace cv::xfeatures2d;

// Global variables
bool filedump = false;
bool show_debug = false;
bool no_rtp = false;
int screen_max_width = 1800;
int fps_window = 1;

std::string camera_cap =
    "video/x-raw,width=1280,height=720,framerate=15/1,format=NV12";
std::string vaapi_enc = "quality-level=7 qos=true";
std::string rtp_pay = "pt=5000 config-interval=5 mtu=60000";
//std::string host_ip = "192.168.1.102";
std::string host_ip = "10.227.13.194";

// Internal variables
json conf;
struct self;

class DataFeeder {
public:
  ctpl::thread_pool pool;
  std::vector<self> devices;
  int crop_top = -1;
  int crop_bottom = -1;
  Mat canvas;
  int canvas_width, canvas_height;
} DataFeeder;

struct self {
  json conf;
  bool is_first = false;
  bool is_last = false;
  int dev_id = DataFeeder.devices.size();

  Mat img;
  int img_pos = 0;
  int img_offset = 0;
  int img_width = 0;
  int alpha_pos = 0;

  std::string camera_cap, rtp_pay, vaapi_enc;

  Mat __dummy_img;

  bool update;
  bool __update_cam_first;
  bool update_alpha;

  int alpha_width;
  int real_height;
  int real_width;

  VideoCapture cap;

  Mat coord_x, coord_y;
  int coord_roi_offset, coord_roi_width;

  int offset_x, offset_y;

  int alpha_right_x, alpha_right_w, alpha_left_x, alpha_left_w;
  int img_x, img_w;

  int canvas_pos;

  Mat roi_img;
  Mat roi_canvas_img, roi_canvas_alpha;
  Mat roi_alpha_left, roi_alpha_right;
  Mat mask, imask;
};

Mat __read(self *sf) {
  Mat frame;
  sf->cap >> frame;
  return frame;
}

Mat __read_dummy(self *sf) { return sf->__dummy_img.clone(); }

void __calculate_cylindrical_coords(self *sf, int focal_length) {
  auto tup = calculate_cylindrical_coords_with_np(
      sf->real_width, sf->real_height, focal_length);
  sf->coord_x = std::get<0>(tup);
  sf->coord_y = std::get<1>(tup);

  // Find ROI
  Mat dummy_img(sf->real_height, sf->real_width, CV_8UC3,
                Scalar(0, 0, 255));

  // Mat result;
  remap(dummy_img, dummy_img, sf->coord_x, sf->coord_y, INTER_LINEAR,
        BORDER_CONSTANT);

  cv::Rect points = extract_image(dummy_img);

  cv::Point x;
  x = points.tl();

  sf->coord_roi_offset = x.x;
  sf->coord_roi_width = points.width;
}

bool __update_cam_first(self *sf) {
  Mat rm;
  remap(__read(sf), rm, sf->coord_x, sf->coord_y, INTER_LINEAR,
        BORDER_REPLICATE);

  for (int i = 0; i < rm.rows; i++) {
    for (int j = 0; j < sf->real_width; j++) {
      DataFeeder.canvas.at<Vec3b>(i, j) = rm.at<Vec3b>(i, j);
    }
  }
  return true;
}

bool update(self *sf) {
  remap(__read(sf), sf->img, sf->coord_x, sf->coord_y, INTER_LINEAR,
        BORDER_REPLICATE);

  Mat img_tmp(DataFeeder.devices[0].real_height -
                  (DataFeeder.crop_bottom - sf->offset_y) -
                  (DataFeeder.crop_top + sf->offset_y),
              sf->img_w, CV_8UC3);
  int id = 0;
  for (int i = DataFeeder.crop_top + sf->offset_y;
       i < DataFeeder.devices[0].real_height -
               (DataFeeder.crop_bottom - sf->offset_y);
       i++) {
    for (int j = 0; j < sf->img_w; j++) {
      img_tmp.at<Vec3b>(id, j) = sf->img.at<Vec3b>(i, j + sf->img_x);
    }
    id = id + 1;
  }
  for (int i = 0;
       i < sf->real_height - DataFeeder.crop_bottom - DataFeeder.crop_top;
       i++) {
    for (int j = 0; j < sf->img_w; j++) {
      DataFeeder.canvas.at<Vec3b>(i + DataFeeder.crop_top,
                                  j + sf->canvas_pos + sf->alpha_left_w) =
          img_tmp.at<Vec3b>(i, j);
    }
  }
  return true;
}

bool update_alpha(self *sf) {
  std::vector<self>::iterator pd = DataFeeder.devices.begin();
  std::advance(pd, sf->dev_id - 1);

  // 1)sf->img
  Mat first(DataFeeder.devices[0].real_height -
                (DataFeeder.crop_bottom - sf->offset_y) -
                (DataFeeder.crop_top + sf->offset_y),
            sf->alpha_left_w, CV_8UC3);
  int id1 = 0;
  for (int i = DataFeeder.crop_top + sf->offset_y;
       i < DataFeeder.devices[0].real_height -
               (DataFeeder.crop_bottom - sf->offset_y);
       i++) {
    for (int j = 0; j < sf->alpha_left_w; j++) {
      first.at<Vec3b>(id1, j) = sf->img.at<Vec3b>(i, j + sf->alpha_left_x);
    }
    id1 = id1 + 1;
  }

  Mat first_c;
  cvtColor(first, first_c, COLOR_BGR2GRAY);
  first_c.convertTo(first_c, CV_32FC1);
  multiply(sf->imask, first_c, first_c);

  // 2)pd->img
  Mat second(DataFeeder.devices[0].real_height -
                 (DataFeeder.crop_bottom - pd->offset_y) -
                 (DataFeeder.crop_top + pd->offset_y),
             pd->alpha_right_w, CV_8UC3);
  int id2 = 0;
  for (int i = DataFeeder.crop_top + pd->offset_y;
       i < DataFeeder.devices[0].real_height -
               (DataFeeder.crop_bottom - pd->offset_y);
       i++) {
    for (int j = 0; j < pd->alpha_right_w; j++) {
      second.at<Vec3b>(id2, j) = pd->img.at<Vec3b>(i, j + pd->alpha_right_x);
    }
    id2 = id2 + 1;
  }

  Mat second_c;
  cvtColor(second, second_c, COLOR_BGR2GRAY);
  second_c.convertTo(second_c, CV_32FC1);
  multiply(sf->mask, second_c, second_c);

  Mat src = Mat::zeros(first_c.size(), CV_8UC3);
  add(second_c, first_c, src);

  Mat result;
  convertScaleAbs(src, result);

  cvtColor(result, result, COLOR_GRAY2BGR);

  for (int i = 0;
       i < sf->real_height - DataFeeder.crop_bottom - DataFeeder.crop_top;
       i++) {
    for (int j = 0; j < sf->alpha_left_w; j++) {
      DataFeeder.canvas.at<Vec3b>(i + DataFeeder.crop_top, j + sf->canvas_pos) =
          result.at<Vec3b>(i, j);
    }
  }
  return true;
}

void __sanitize_cropping_area() {
  for (int i = 0; i < DataFeeder.devices.size(); i++) {
    if (DataFeeder.devices[i].offset_y < 0) {
      DataFeeder.crop_top =
          std::max(DataFeeder.crop_top, -DataFeeder.devices[i].offset_y);
    } else {
      DataFeeder.crop_bottom =
          std::max(DataFeeder.crop_bottom, DataFeeder.devices[i].offset_y);
    }
  }
}

void __populate_canvas() {
  // Calculate image offsets in canvas
  int canvas_pos_last = 0;
  for (int i = 0; i < DataFeeder.devices.size(); i++) {
    if (!DataFeeder.devices[i].is_first) {
      DataFeeder.devices[i].img.create(DataFeeder.devices[i].real_height,
                                       DataFeeder.devices[i].real_width,
                                       CV_8UC3);
    }
    if (!DataFeeder.devices[i].is_last) {
      DataFeeder.devices[i].alpha_right_w =
          DataFeeder.devices[i + 1].alpha_width;
    } else {
      DataFeeder.devices[i].alpha_right_w = 0;
    }

    DataFeeder.devices[i].img_x = DataFeeder.devices[i].coord_roi_offset +
                                  DataFeeder.devices[i].coord_roi_width +
                                  DataFeeder.devices[i].offset_x;
    DataFeeder.devices[i].img_w =
        -DataFeeder.devices[i].offset_x - DataFeeder.devices[i].alpha_right_w;
    DataFeeder.devices[i].alpha_left_x =
        DataFeeder.devices[i].img_x - DataFeeder.devices[i].alpha_width;
    DataFeeder.devices[i].alpha_left_w = DataFeeder.devices[i].alpha_width;

    DataFeeder.devices[i].alpha_right_x =
        DataFeeder.devices[i].img_x + DataFeeder.devices[i].img_w;

    // Update canvas pos
    if (DataFeeder.devices[i].dev_id == 1) {
      DataFeeder.devices[i].canvas_pos =
          (DataFeeder.devices[0].coord_roi_offset +
           DataFeeder.devices[0].coord_roi_width -
           DataFeeder.devices[0].alpha_right_w);
      canvas_pos_last = DataFeeder.devices[i].canvas_pos;
    } else {
      DataFeeder.devices[i].canvas_pos = canvas_pos_last;
    }
    canvas_pos_last = canvas_pos_last + DataFeeder.devices[i].alpha_left_w +
                      DataFeeder.devices[i].img_w;
  }
  DataFeeder.canvas.create(DataFeeder.devices[0].real_height, canvas_pos_last,
                           CV_8UC3);

  // Debug point
  DataFeeder.canvas_width =
      canvas_pos_last - DataFeeder.devices[0].coord_roi_offset;
  DataFeeder.canvas_height = DataFeeder.devices[0].real_height -
                             (DataFeeder.crop_top + DataFeeder.crop_bottom);

  DataFeeder.devices[0].img = DataFeeder.canvas;
}

void __precompute_roi() {
  int rh = DataFeeder.devices[0].real_height;
  int ct = DataFeeder.crop_top;
  int cb = DataFeeder.crop_bottom;
  int ih = rh - (ct + cb);

  for (int i = 0; i < DataFeeder.devices.size(); i++) {
    int iy = DataFeeder.devices[i].offset_y;
    int cax = DataFeeder.devices[i].canvas_pos;
    int cix =
        DataFeeder.devices[i].canvas_pos + DataFeeder.devices[i].alpha_left_w;
    int iw = DataFeeder.devices[i].img_x;
    int alx = DataFeeder.devices[i].alpha_left_x;
    int alw = DataFeeder.devices[i].alpha_left_w;
    int arx = DataFeeder.devices[i].alpha_right_x;
    int arw = DataFeeder.devices[i].alpha_right_w;

    Mat mask = generate_gradation_mask(alw, ih);

    if (mask.rows == 0) {
      continue;
    }
    mask.convertTo(mask, CV_32FC1, 1.0 / 255);
    DataFeeder.devices[i].mask = mask;
    DataFeeder.devices[i].imask = (Scalar::all(1.0) - mask);
  }
}

void populate(json conf) {
  int threads = 0;
  // int threads = 4;

  // Load variables
  DataFeeder.crop_top = int(conf["crop_top"]);
  DataFeeder.crop_bottom = int(conf["crop_bottom"]);
  __sanitize_cropping_area();

  // Mark last device
  DataFeeder.devices.back().is_last = true;

  // Create threadpool
  if (threads == 0) {
    DataFeeder.pool;
  } else {
    DataFeeder.pool.resize(threads);
  }

  // Prepare canvas
  __populate_canvas();

  // Prepare ROI
  __precompute_roi();
}

Mat read() {
  __update_cam_first(&DataFeeder.devices[0]);

  if (DataFeeder.pool.size() > 1) {
    // Update and remap individual buffer
    for (int i = 1; i < DataFeeder.devices.size(); i++) {
      DataFeeder.pool.push([](int i) { DataFeeder.devices[i].update; });
    }
    for (int j = 0; j < DataFeeder.pool.size(); j++) {
      if (!DataFeeder.pool.pop()) {
        std::cout << "[-] Something wrong.." << std::endl;
      }
    }

    // Update alpha
    for (int i = 1; i < DataFeeder.devices.size(); i++) {
      DataFeeder.pool.push([](int i) { DataFeeder.devices[i].update_alpha; });
    }
    for (int j = 0; j < DataFeeder.pool.size(); j++) {
      if (!DataFeeder.pool.pop()) {
        std::cout << "[-] Something wrong (Alpha).." << std::endl;
      }
    }

  } else {
    for (int i = 1; i < DataFeeder.devices.size(); i++) {
      update(&DataFeeder.devices[i]);
      update_alpha(&DataFeeder.devices[i]);
    }
  }

  // Debug point
  Mat temp(
      DataFeeder.canvas.rows - DataFeeder.crop_bottom - DataFeeder.crop_top,
      DataFeeder.canvas.cols - DataFeeder.devices[0].coord_roi_offset, CV_8UC3);
  for (int i = 0; i < DataFeeder.canvas.rows - DataFeeder.crop_bottom -
                          DataFeeder.crop_top;
       i++) {
    for (int j = 0;
         j < DataFeeder.canvas.cols - DataFeeder.devices[0].coord_roi_offset;
         j++) {
      temp.at<Vec3b>(i, j) = DataFeeder.canvas.at<Vec3b>(
          i + DataFeeder.crop_top, j + DataFeeder.devices[0].coord_roi_offset);
    }
  }
  return temp;
}

int main(int argc, char **argv) {
  cv::setUseOptimized(true);

  if (argc < 2) {
    std::cout << "No Input Images" << std::endl;
    return -1;
  }

  // Load configuration
  std::ifstream configuration("configuration.json", std::ifstream::binary);
  configuration >> conf;

  std::cout << "Start stitching" << std::endl;

  self *sf = new self[sizeof(argc)];

  // Load images and precompute matrixs
  for (int i = 1; i <= sizeof(argc); i++) {
    std::cout << "Load image:" << argv[i] << std::endl;

    sf[i - 1].conf = conf;
    sf[i - 1].is_first = false;
    sf[i - 1].is_last = false;
    sf[i - 1].dev_id = DataFeeder.devices.size();

    sf[i - 1].img = 0;
    sf[i - 1].img_pos = 0;
    sf[i - 1].img_offset = 0;
    sf[i - 1].img_width = 0;
    sf[i - 1].alpha_pos = 0;

    std::string s = argv[i];
    if (s.rfind("/dev/video", 0) == 0) {
      if (no_rtp == true) {
		std::string str = "v4l2src io-mode=4 device=" + s + " ! " + camera_cap +"! videoconvert dither=0 chroma-resampler=0 ""alpha-value=0 chroma-mode=3 ! appsink";
        sf[i - 1].cap.open(str, CAP_GSTREAMER);
      } else if (sf[i - 1].dev_id == 1) {
		std::string str = "v4l2src io-mode=4 device=" + s + " ! " + camera_cap +"! tee name=t" + std::to_string(sf[i-1].dev_id)+" t"+std::to_string(sf[i-1].dev_id) + ". ! vaapih264enc " +vaapi_enc + " ! rtph264pay " + rtp_pay +" ! udpsink host=" + host_ip + " port="+ std::to_string(5000+sf[i - 1].dev_id) + " t" +std::to_string(sf[i - 1].dev_id) +". ! videoconvert dither=0 chroma-resampler=0 alpha-value=0 chroma-mode=3 ! appsink alsasrc device=hw:1,0 ! audio/x-raw,format=S16LE,rate=48000,channels=2 ! audioconvert ! vorbisenc ! rtpvorbispay " + rtp_pay + " ! udpsink host=" + host_ip +" port=5005";
        sf[i - 1].cap.open(str, CAP_GSTREAMER);
      } else {
		std::string str = "v4l2src io-mode=4 device=" + s + " ! " + camera_cap + " ! tee name=t" + std::to_string(sf[i-1].dev_id) + " t" + std::to_string(sf[i-1].dev_id) +". ! vaapih264enc quality-level=7 ! rtph264pay " + rtp_pay +" ! udpsink host=" + host_ip + " port=" + std::to_string(5000+sf[i-1].dev_id)+ " t" + std::to_string(sf[i - 1].dev_id) +". ! videoconvert dither=0 chroma-resampler=0 alpha-value=0 chroma-mode=3 ! appsink";
        sf[i - 1].cap.open(str, CAP_GSTREAMER);
      }
      if (!sf[i - 1].cap.isOpened()) {
        std::cerr << "Unable to start camera" + s;
        return -1;
      }
    } else {
      sf[i - 1].__dummy_img = imread(argv[i]);
      auto x = __read(&sf[i - 1]);
      auto y = __read_dummy(&sf[i - 1]);
      x = y;
    }

    // First camera handler
    if (DataFeeder.devices.size() == 0) {
      sf[i - 1].is_first = true;
      sf[i - 1].update = sf[i - 1].__update_cam_first;
    }

    // Load simple variables
    std::string idx = "camera" + std::to_string(i - 1);
    sf[i - 1].alpha_width = conf[idx]["alpha_width"];
    if (sf[i - 1].alpha_width < 0) {
      sf[i - 1].alpha_width = 0;
    }

    sf[i - 1].real_height = __read(&sf[i - 1]).rows;
    sf[i - 1].real_width = __read(&sf[i - 1]).cols;

    // Generate precomputed resources
    __calculate_cylindrical_coords(&sf[i - 1], conf[idx]["focal_length"]);

    sf[i - 1].offset_x = conf[idx]["offset_x"];
    sf[i - 1].offset_y = conf[idx]["offset_y"];
    DataFeeder.devices.push_back(sf[i - 1]);
  }

  // Prepare stitching infrastructure
  populate(conf["global"]);
  std::cout << "Canvas: " << DataFeeder.canvas_width << " "
            << DataFeeder.canvas_height << std::endl;

  // FPS
  time_t fps_prev_time;
  fps_prev_time = time(NULL); // current time?
  int fps_counter = 0;

  // Debug
  bool enable_resizer;
  if (screen_max_width < DataFeeder.canvas.cols) {
    enable_resizer = true;
  } else {
    enable_resizer = false;
  }

  //int offset = 0;
  int offset = DataFeeder.devices[0].coord_roi_offset - 1;

  // Create UDP sink for stiching image
  VideoWriter out;
  std::string str = "appsrc is-live=true block=true ! video/x-raw,format=BGR ! queue ! videoconvert n-threads=4 dither=0 chroma-resampler=0 alpha-value=0 chroma-mode=3 ! video/x-raw,stream-format=(string)byte-stream ! videocrop right=100 ! vaapih264enc "+ vaapi_enc+" ! rtph264pay "+ rtp_pay+" ! udpsink host="+host_ip+" port=5004";
  out.open(str, CAP_GSTREAMER, 20, Size(DataFeeder.canvas_width + offset, DataFeeder.canvas_height));

  /*
  VideoWriter out(
      "appsrc is-live=true block=true ! video/x-raw,format=BGR ! queue ! "
      "videoconvert ! xvimagesink",
      CAP_GSTREAMER, 24,
      Size(DataFeeder.canvas_width + offset, DataFeeder.canvas_height));
 */
  if (filedump == true) {
    VideoWriter fileout(
        ""
        "appsrc ! multifilesink location=out_%0d.raw"
        "",
        CAP_GSTREAMER, 24,
        Size(DataFeeder.canvas_width, DataFeeder.canvas_height));
  }

  while (1) {
    Mat canvas = read();

    for (int i = 0; i < DataFeeder.canvas.rows; i++) {
      for (int j = 0; j < DataFeeder.canvas_width; j++) {
        DataFeeder.canvas.at<Vec3b>(i, j) = canvas.at<Vec3b>(i, j);
      }
    }

    out.write(DataFeeder.canvas);
    if (filedump == true) {
      // fileout.write(DataFeeder.canvas);
    }

    // Debug
    if (show_debug == true) {
      if (enable_resizer == true) {
        imshow("result", resize_by_width(canvas, screen_max_width));
      } else {
        imshow("result", canvas);
      }
    }

    // Measure FPS
    fps_counter = fps_counter + 1;
    if (time(NULL) - fps_prev_time > fps_window) {
      auto fps = fps_counter / (time(NULL) - fps_prev_time);
      fps_prev_time = time(NULL);

      std::cout << "FPS for " << fps_window << " secs: " << fps * 0.1f
                << std::endl;
      std::cout << "Offset: " << offset
                << " Canvas: " << DataFeeder.canvas.size() << ", "
                << DataFeeder.devices[0].coord_roi_offset << std::endl;

      fps_counter = 0;
    }

    // Exit program
    if (show_debug == true) {
      if (waitKey(1) == 'q') {
        break;
      }
    }
  }
  std::cout << "Stop stitching" << std::endl;

  destroyAllWindows();
  return 0;
}
