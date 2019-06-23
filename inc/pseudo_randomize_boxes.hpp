#ifndef _PSEUDO_RANDOMIZE_BOXES_HPP_
#define _PSEUDO_RANDOMIZE_BOXES_HPP_

#include "includes.hpp"

int *randarray(int maxnum);
CvMat Pseudo_randomize_boxes(CvMat *boxes, CvMat *priority);

#endif
