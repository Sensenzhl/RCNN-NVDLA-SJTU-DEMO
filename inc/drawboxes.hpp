#ifndef _DRAW_BOXES_HPP_
#define _DRAW_BOXES_HPP_

#include "includes.hpp"

string DoubleToString(double Input);
Mat drawboxes(CvMat* im, CvMat* box, int num_class);
Mat cvMatToMat(const CvMat* m, bool copyData);

#endif
