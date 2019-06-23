#include "defines.hpp"
#include "pseudo_randomize_boxes.hpp"

int *randarray(int maxnum)
{
	//randomize priority
	int x;

	int length = int(maxnum);
	int *y = new int[length];
	int *temp = new int[length];
	
	int flag_exist = 0;
	int j = 0;
	int index;
	int flag_find = 0;
	int u;

	for (j = 0; j < maxnum; j++)
	{
		temp[j] = j;
	}

	j = 0;

	for (int j = 0; j < maxnum; j++)
	{
		index = rand() % length;
		y[j] = temp[index];
		length = length - 1;
		flag_find = 0;

		for (int k = 0; k < length; k++)
		{
			if (k != index)
			{
				if (!flag_find)
					temp[k] = temp[k];
				else
					temp[k] = temp[k + 1];
			}
			else
			{
				temp[k] = temp[k + 1];
				flag_find = 1;
			}
		}
	}
	delete [] temp;
	
	return y;
}

CvMat Pseudo_randomize_boxes(CvMat *boxes, CvMat *priority)
{
	int   array_length       =  priority->rows;
	int   * rand_array       =  new int[array_length];
	
	rand_array               =  randarray(priority->rows);

	CvMat * rand_array_box   =  cvCreateMat(priority->rows, 1, CV_32FC1);
	CvMat * priority_product =  cvCreateMat(priority->rows, 1, CV_32FC1);
	
	float * ptr;
	float * ptr_boxes;
	float * ptr_out;

	//select boxes row & column > minBoxSize into array index_reserved[]
	for (int i = 0; i < rand_array_box->rows; i++)
	{
		ptr = (float*)(rand_array_box->data.ptr + i * rand_array_box->step);
		*(ptr) = rand_array[i];
	}

	//print priority and random array
	for (int i = 0; i < rand_array_box->rows; i++)
	{
		ptr = (float*)(rand_array_box->data.ptr + i * rand_array_box->step);
		//calculate row address and converting into real data type(float*) for further calculation
	}

	for (int i = 0; i < priority->rows; i++)
	{
		ptr = (float*)(priority->data.ptr + i * priority->step);//stepÊÇ×Ö½ÚÊý£¬ËùÒÔÊ×µØÖ·mat.dataÒªÓÃuchar*ÀàÐÍ£¨¼´mat.data.ptr£©£¬  
		//calculate row address and converting into real data type(float*) for further calculation
	}	
	
	cvMul(priority, rand_array_box, priority_product);
	
	//print priority_product
	for (int i = 0; i < priority_product->rows; i++)
	{
		ptr = (float*)(priority_product->data.ptr + i * priority_product->step);//stepÊÇ×Ö½ÚÊý£¬ËùÒÔÊ×µØÖ·mat.dataÒªÓÃuchar*ÀàÐÍ£¨¼´mat.data.ptr£©£¬  
		//calculate row address and converting into real data type(float*) for further calculation
	}

	//InsertionSort Boxes(ascending)
	for (int i = 1; i < priority_product->rows; i++)         // ÀàËÆ×¥ÆË¿ËÅÆÅÅÐò
	{
		ptr = (float*)(priority_product->data.ptr + i * priority_product->step);
		ptr_boxes = (float*)(boxes->data.ptr + i * boxes->step);
		int get_length = boxes->cols + 1;
		float *get = new float[get_length];

		// ÓÒÊÖ×¥µ½Ò»ÕÅÆË¿ËÅÆ
		get[0] = *(ptr);  
		for (int k = 0; k < boxes->cols; k++)
		{
			get[k + 1] = *(ptr_boxes + k);
		}

		int j   = i - 1;                   // ÄÃÔÚ×óÊÖÉÏµÄÅÆ×ÜÊÇÅÅÐòºÃµÄ		

		while (j >= 0 && ((*((float*)(priority_product->data.ptr + j * priority_product->step))) > get[0]))    // ½«×¥µ½µÄÅÆÓëÊÖÅÆ´ÓÓÒÏò×ó½øÐÐ±È½Ï
		{
			(*((float*)(priority_product->data.ptr + (j + 1) * priority_product->step))) = (*((float*)(priority_product->data.ptr + j * priority_product->step))); // Èç¹û¸ÃÊÖÅÆ±È×¥µ½µÄÅÆ´ó£¬¾Í½«ÆäÓÒÒÆ
			//ptr_out = (float*)(out_boxes->data.ptr + (j + 1) * out_boxes->step);
			for (int k = 0; k < boxes->cols; k++)
			{
				(*((float*)(boxes->data.ptr + (j + 1) * boxes->step + k))) = (*((float*)(boxes->data.ptr + j * boxes->step + k)));
			}
			j--;
		}

		(*((float*)(priority_product->data.ptr + (j + 1) * priority_product->step))) = get[0]; // Ö±µ½¸ÃÊÖÅÆ±È×¥µ½µÄÅÆÐ¡(»ò¶þÕßÏàµÈ)£¬½«×¥µ½µÄÅÆ²åÈëµ½¸ÃÊÖÅÆÓÒ±ß(ÏàµÈÔªËØµÄÏà¶Ô´ÎÐòÎ´±ä£¬ËùÒÔ²åÈëÅÅÐòÊÇÎÈ¶¨µÄ)
		//ptr_out = (float*)(out_boxes->data.ptr + (j + 1) * out_boxes->step);
		ptr_boxes = (float*)(boxes->data.ptr + (j + 1) * boxes->step);
		for (int k = 0; k < boxes->cols; k++)
		{
			*(ptr_boxes + k) = get[k + 1];
		}
	}

	delete [] rand_array;

	return  *boxes;
}
