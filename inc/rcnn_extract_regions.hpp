#ifndef _RCNN_EXTRACT_REGIONS_HPP_
#define _RCNN_EXTRACT_REGIONS_HPP_

#include "includes.hpp"
#include "crop.hpp"

extern Mat cvMatToMat(const CvMat* m, bool copyData);

float max_coord(float x1, float x2);
float min_coord(float x1, float x2);
Mat iplImageToMat(const IplImage* img, bool copyData);
CvMat Rcnn_extract_regions(IplImage* image, CvMat* box, int batch_size, Crop crop_param);

#endif
