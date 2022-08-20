#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;

Mat resize_by_width(Mat img, int max_width) {
  int height = img.rows;
  int width = img.cols;
  float scale = (float)max_width / (float)width;
  Mat result;

  Size size(width * scale, height * scale);
  resize(img, result, size);

  return result;
}

std::vector<uchar> linspace(int a, int b, int N) {
  int h = (b - a) / static_cast<uchar>(N - 1);
  std::vector<uchar> xs(N);
  typename std::vector<uchar>::iterator x;
  int val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

Mat generate_gradation_mask(int width, int height) {
  int end = -1;
  if (width < 1 | height < 1) {
    ;
    Mat none;
    return none;
  }

  if (end < 0) {
    end = width;
  }
  std::vector<uchar> pattern = linspace(255, 0, end);

  int padding = width - end;

  /* if */

  Mat temp = Mat(pattern).reshape(0, 30);
  temp.convertTo(temp, CV_8UC1);

  Mat tile(height, pattern.size(), CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < temp.rows; j++) {
      tile.at<uchar>(i, j) = temp.at<uchar>(0, j);
    }
  }

  Mat result;
  cv::normalize(tile, result, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);

  return result;
}

Rect extract_image(Mat img) {
  Mat result;
  cvtColor(img, result, COLOR_BGR2GRAY);
  Mat thresh;
  threshold(result, thresh, 1, 255, THRESH_BINARY);

  std::vector<std::vector<Point>> contours;
  findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  return boundingRect(contours[0]);
}

std::tuple<Mat, Mat> calculate_cylindrical_coords(float width, float height,
                                                  float F) {
  Mat map_x(height, width, CV_32FC1, Scalar(-1));
  Mat map_y(height, width, CV_32FC1, Scalar(-1));

  int num1 = (height / 2);
  int num2 = (width / 2);

  for (int y = -num1; y < num1; y++) {
    for (int x = -num2; x < num2; x++) {
      float dst_x = round((F * atanf(x / F)) + width / 2);
      float dst_y = round((F * y / sqrt(pow(x, 2) + pow(F, 2))) + height / 2);

      if (dst_x < 0 or dst_x >= width or dst_y < 0 or dst_y >= height) {
        continue;
      }
      map_x.at<float>(dst_y, dst_x) = x + int(width / 2);
      map_y.at<float>(dst_y, dst_x) = y + int(height / 2);
    }
  }
  return {map_x, map_y};
}

std::tuple<Mat, Mat>
calculate_cylindrical_coords_with_np(float width, float height, float F) {
  Mat K = (Mat_<float>(3, 3) << F, 0, width / 2, 0, F, height / 2, 0, 0, 1);
  Mat Kinv = K.inv();

  // pixel coordinates
  // 1) indices
  Mat x(height, width, CV_32FC1);
  Mat y(height, width, CV_32FC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      y.at<float>(i, j) = i;
      x.at<float>(i, j) = j;
    }
  }

  // 2) stack
  Mat z(x.rows, x.cols, CV_32FC1, Scalar(1)); // np.ones_like
  Mat X(height, width, CV_32FC3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      Vec3f &pt = X.at<Vec3f>(i, j);
      pt[0] = x.at<float>(i, j);
      pt[1] = y.at<float>(i, j);
      pt[2] = z.at<float>(i, j);
    }
  }
  Mat result_X = X.reshape(1, height * width); //# of ch, # of rows

  // 3) X = Kinv.dot(X.T).T
  Mat T = result_X.t();
  Mat tmp = Kinv * T;
  result_X = tmp.t();

  // calculate cylindrical coords (sin\theta, h, cos\theta)
  Mat A(result_X.rows, 3, CV_32FC1);
  for (int i = 0; i < result_X.rows; i++) {
    A.at<float>(i, 0) = sin(result_X.at<float>(i, 0));
    A.at<float>(i, 1) = result_X.at<float>(i, 1);
    A.at<float>(i, 2) = cos(result_X.at<float>(i, 0));
  }

  // project back to image-pixels plane
  Mat T2 = A.t();
  Mat tmp2 = K * T2;
  Mat B = tmp2.t();

  // back from homog coords
  Mat up(B.rows, B.cols - 1, CV_32FC1);
  Mat down(B.rows, B.cols - 1, CV_32FC1);

  for (int i = 0; i < B.rows; i++) {
    for (int j = 0; j < B.cols - 1; j++) {
      up.at<float>(i, j) = B.at<float>(i, j);
      down.at<float>(i, j) = B.at<float>(i, B.cols - 1);
    }
  }
  Mat B2 = up / down;

  // make sure warp coords only within image bounds
  for (int i = 0; i < B2.rows; i++) {
    if ((B2.at<float>(i, 0) < 0) | (B2.at<float>(i, 0) >= width) |
        (B2.at<float>(i, 1) < 0) | (B2.at<float>(i, 1) >= height)) {
      for (int j = 0; j < B2.cols; j++) {
        B2.at<float>(i, j) = -1;
      }
    }
  }
  Mat B3 = B2.reshape(2, height);

  Mat coords1(B3.rows, B3.cols, CV_32FC1);
  Mat coords2(B3.rows, B3.cols, CV_32FC1);

  for (int i = 0; i < B3.rows; i++) {
    for (int j = 0; j < B3.cols; j++) {
      Vec2f &pt = B3.at<Vec2f>(i, j);
      coords1.at<float>(i, j) = pt[0];
      coords2.at<float>(i, j) = pt[1];
    }
  }

  Mat cylin1, cylin2;
  convertMaps(coords1, coords2, cylin1, cylin2, CV_16SC2);

  return {cylin1, cylin2};
}
