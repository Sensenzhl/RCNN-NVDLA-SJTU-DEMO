#include "defines.hpp"
#include "nms.hpp"

using namespace cv;
using namespace std;

CvMat *nms(CvMat* boxes, double overlap)
{
	CvMat		  *box_remained = cvCreateMat(boxes->rows, boxes->cols, CV_32FC1);
	double		  top_score = -1;
	float		  max_x1;
	float		  max_x2;
	float		  max_y1;
	float		  max_y2;
	float		  max_score_area;
	int		  index;
	const float	  *ptr;
	float	   	  *ptr_remained = NULL;
	float	   	  *ptr_temp = NULL;

	float		  height;
	float		  width;
	float		  area;

	float		  inter_x1;
	float		  inter_x2;
	float	      	  inter_y1;
	float		  inter_y2;
	float		  inter_area;
	float	      	  ratio;
	int		  length_index_reserved = boxes->rows;
	int	          *index_reserved = new int[length_index_reserved];
	int	          *index_reserved_temp = new int[length_index_reserved];
	int	          *top_index = new int[length_index_reserved];
	int		  top_num = 0;	
	int		  i = 0;
	int		  length_index_reserved_temp;
	int		  offset;

	//initial reserved box row = boxes.rows
	while (i < boxes->rows)
	{
		*(index_reserved + i) = i;
		i++;
	}

	while (length_index_reserved != 0)
	{
		//get reserved box matrix
		for (int i = 0; i < length_index_reserved; i++)
		{
			ptr = (const float*)(boxes->data.ptr + index_reserved[i] * boxes->step);//stepÊÇ×Ö½ÚÊý£¬ËùÒÔÊ×µØÖ·mat.dataÒªÓÃuchar*ÀàÐÍ£¨¼´mat.data.ptr£©£¬  
			ptr_remained = (float*)(box_remained->data.ptr + index_reserved[i] * box_remained->step);   
			for (int j = 0; j < boxes->cols; j++)
			{
				*(ptr_remained + j) = *(ptr + j);
			}
		}
		
		ptr = (const float*)(box_remained->data.ptr + index_reserved[0] * box_remained->step);
		top_score = *(ptr + 4);
		top_index[top_num] = index_reserved[0];
		max_x1 = *(ptr);
		max_x2 = *(ptr + 2);
		max_y1 = *(ptr + 1);
		max_y2 = *(ptr + 3);
		max_score_area = (max_x2 - max_x1 + 1) * (max_y2 - max_y1 + 1);

		for (int i = 0; i < length_index_reserved; i++)
		{
			int j = 4; //Get the biggest score(column = 5 thus j = 4) and index, then calculate its area		
			ptr = (const float*)(box_remained->data.ptr + index_reserved[i] * box_remained->step);

			if (top_score < (*(ptr + j)))
			{
				top_score = *(ptr + j);
				//*(top_index + top_num * sizeof(int)) = i;
				top_index[top_num] = index_reserved[i];
				max_x1      = *(ptr);
				max_x2      = *(ptr + 2);
				max_y1      = *(ptr + 1);
				max_y2      = *(ptr + 3);

				max_score_area = (max_x2 - max_x1 + 1) * (max_y2 - max_y1 + 1);
			}
		}		

		//index = 0 ~ max_num - 1, differ from malab
		length_index_reserved_temp = length_index_reserved;
		length_index_reserved = 0;
		offset = 0;
		top_num = top_num + 1;

		//clear index_reserved in a new term
		index_reserved_temp = index_reserved;
		memset(index_reserved, 0, 0);
		
		//Calculate every box's area & intersect area & NMS ratio
		for (int i = 0; i < length_index_reserved_temp; i++)
		{
			ptr = (const float*)(box_remained->data.ptr + index_reserved_temp[i] * box_remained->step);

			//Calculate every box's area
			height = (*(ptr + 2)) - (*(ptr))+ 1;
			width  = (*(ptr + 3)) - (*(ptr + 1)) + 1;
			area   = height * width;

			//Calculate intersect area
			inter_x1   =  max(max_x1, (*(ptr)));
			inter_x2   =  min(max_x2, (*(ptr + 2)));
			inter_y1   =  max(max_y1, (*(ptr + 1)));
			inter_y2   =  min(max_y2, (*(ptr + 3)));
			inter_area =  (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1);

			//Calculate NMS ratio = intersect area / (max_score_area +  This box area - intersect area)
			ratio      =  inter_area / (max_score_area + area - inter_area);

			//Reserve boxes which NMS ratio < overlap
			if (ratio < overlap){
				index_reserved[offset] = index_reserved_temp[i];
				length_index_reserved = length_index_reserved + 1; //row number of reserved matrix
				offset = offset + 1;
			}			 
		}
	}

	CvMat *box_out = cvCreateMat(top_num, 5, CV_32FC1);
	
	//get final box matrix after nms
	for (int i = 0; i < top_num; i++)
	{
		ptr      = (const float*)(boxes->data.ptr + top_index[i] * boxes->step);
		ptr_temp = (float*)(box_out->data.ptr + i * box_out->step);
		   
		for (int j = 0; j < box_out->cols; j++)
		{
			*(ptr_temp + j) = *(ptr + j);
		}
	}

	delete [] index_reserved;
	//delete [] index_reserved_temp;

	return box_out;
}
