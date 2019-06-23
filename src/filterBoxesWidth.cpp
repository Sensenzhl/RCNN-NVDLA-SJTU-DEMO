#include "defines.hpp"
#include "filterBoxesWidth.hpp"

using namespace cv;
using namespace std;

CvMat *FilterBoxesWidth(CvMat *boxes, int minBoxSize)
{
	int   length_index_reserved = 0;
	int   * index_reserved = new int[boxes->rows];
	int   * top_index      = new int[boxes->rows];	
	float * ptr;
	float * ptr_out;

	printf("minBoxSize = %d\n", minBoxSize);
	//select boxes row & column > minBoxSize into array index_reserved[]
	for (int i = 0; i < boxes->rows; i++)
	{		  
		ptr = (float*)(boxes->data.ptr + i * boxes->step);
		//printf("width = %f height = %f\n",(*(ptr + 2) - *(ptr)),(*(ptr + 3) - *(ptr + 1)));
		if (((*(ptr + 2) - *(ptr)) >= minBoxSize) & ((*(ptr + 3) - (*(ptr + 1))) >= minBoxSize))
		{
			index_reserved[length_index_reserved] = i;
			length_index_reserved = length_index_reserved + 1;
		}			
	}

	CvMat* boxes_out = cvCreateMat(length_index_reserved, boxes->cols, CV_32FC1);

	//select boxes row & column > minBoxSize
	for (int i = 0; i < length_index_reserved; i++)
	{  	   
		ptr     = (float*)(boxes->data.ptr + index_reserved[i] * boxes->step);
		ptr_out = (float*)(boxes_out->data.ptr + i * boxes_out->step);
		for (int j = 0; j < boxes->cols; j++)
		{
			*(ptr_out + j) = *(ptr + j);
		}
	}
	
	delete [] top_index;

	return boxes_out;
}
