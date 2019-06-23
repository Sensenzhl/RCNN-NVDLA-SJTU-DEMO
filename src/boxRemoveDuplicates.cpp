#include "defines.hpp"
#include "boxRemoveDuplicates.hpp"

//using namespace cv;
//using namespace std;

CvMat *BoxRemoveDuplicates(CvMat *boxes)
{
	int   length_index_reserved = boxes->rows;
	int   * index_reserved = new int[length_index_reserved];
	int   offset = 0;
	float * ptr;
	float * ptr_out;
	int   flag_exist = 0;
	index_reserved[0] = 0;
	length_index_reserved = 1;
	
	for (int i = 0; i < boxes->rows; i++)
	{
		const float* ptr = (const float*)(boxes->data.ptr + i * boxes->step);//stepÊÇ×Ö½ÚÊý£¬ËùÒÔÊ×µØÖ·mat.dataÒªÓÃuchar*ÀàÐÍ£¨¼´mat.data.ptr£©£¬  
		//calculate row address and converting into real data type(float*) for further calculation
	}	

	//select unique rows into index_reserved array
	for (int i = 0; i < boxes->rows; i++)
	{		
		ptr = (float*)(boxes->data.ptr + i * boxes->step);
		flag_exist = 0;
		for (int j = 1; j < (length_index_reserved + 1); j++)
		{			
			ptr_out = (float*)(boxes->data.ptr + index_reserved[j - 1] * boxes->step);

			if ((*(ptr_out) == *(ptr)) && (*(ptr_out + 1) == *(ptr + 1))
				&& (*(ptr_out + 2) == *(ptr + 2)) && (*(ptr_out + 3) == *(ptr + 3)))
			{
				flag_exist = 1;
				break;
			}
		}
		if (!flag_exist)
		{
			index_reserved[length_index_reserved] = i;
			length_index_reserved = length_index_reserved + 1;
			flag_exist = 1;
		}
	}

	CvMat* boxes_out = cvCreateMat(length_index_reserved, boxes->cols, CV_32FC1);

	//assign unique rows to boxes_out
	for (int i = 0; i < length_index_reserved; i++)
	{
		ptr = (float*)(boxes->data.ptr + i * boxes->step);
		ptr_out = (float*)(boxes_out->data.ptr + index_reserved[i] * boxes_out->step);
		for (int j = 0; j < boxes->cols; j++)
		{
			*(ptr_out + j) = *(ptr + j);
		}
	}

	delete [] index_reserved;

	return boxes_out;
}
