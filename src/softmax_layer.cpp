#include "defines.hpp"
#include "softmax_layer.hpp"

CvMat *softmax_layer(CvMat *boxes)
{
	float	*ptr;
	float	sum;

	for (int i = 0; i < boxes->rows; i++)
	{
		sum = 0;
		ptr = (float *)(boxes->data.ptr + i * boxes->step);
		for (int j = 4; j < boxes->cols; j++)
		{
			sum += exp(*(ptr + j));
		}

		for (int k = 4; k < boxes->cols; k++)
		{
			*(ptr + k) = exp(*(ptr + k)) / sum;
		}
	}
	
	return boxes;
}


