#ifndef _SELECTIVE_SEARCH_BOXES_HPP_
#define _SELECTIVE_SEARCH_BOXES_HPP_

#include "includes.hpp"
extern Mat cvMatToMat(const CvMat* m, bool copyData);
CvMat *selective_search_boxes(IplImage* img, bool fast_mode, double im_width);
#endif
