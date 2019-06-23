#include "defines.hpp"
#include "main.hpp"

int main(int argc, char *argv[])
{
	double		score;
	IplImage	*img;
	CvMat		temp;
	Mat		image_draw;
	Mat		image_temp;
	float 		*ptr;
	float 		*ptr_temp;
	float 		*ptr_thresholded;
	float 		*ptr_scored;
	float 		*ptr_removed_duplicate;
	int         flag_box_exist=0;

	//getFiles("./test_image/",files);
	//char str[30];
	//int  size = files.size();
	//for (int i = 0; i < size; i++)
	//{
	//    cout << "file size: "<< size <<files[i]<< endl;
	//}

	std::cout << "OpenCV Version: " << CV_VERSION << endl; 
	img = cvLoadImage("./test_image/person-bike.jpg");

	CvMat *mat = cvGetMat(img, &temp);

	cvNamedWindow("Input_Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Input_Image", img);
	cvWaitKey(0);

	double 		overlap = NMS_OVERLAP;

	//*********************************************  Selective Search Boxes  **********************************
	bool      	fast_mode = FAST_MODE;
	double    	im_width  = IM_WIDTH;
	CvMat     	*crop_boxes = selective_search_boxes(img, fast_mode, im_width);
	CvMat     	*boxes 	= cvCloneMat(crop_boxes);

	double    	minBoxSize = MIN_BOX_SIZE;
	printf("boxes number: %d \n", boxes->rows);

	CvMat     	*boxes_filtered = FilterBoxesWidth(boxes, minBoxSize);
	printf("remained boxes number after filtered: %d \n", boxes_filtered->rows);

	CvMat 	  	*boxes_remove_duplicates = BoxRemoveDuplicates(boxes_filtered);
	printf("remained boxes number removed duplicates: %d \n", boxes_remove_duplicates->rows);

	float 	  	*ptr_tmp;
	int   	  	Box_num = boxes_remove_duplicates->rows;

	double    	scale = img->width / im_width;

	//scale all boxes 
	for (int i = 0; i < boxes_remove_duplicates->rows; i++)
	{
		ptr_tmp = (float*)(boxes_remove_duplicates->data.ptr + i * boxes_remove_duplicates->step);

		for (int j = 0; j < boxes_remove_duplicates->cols; j++)
		{
			*(ptr_tmp + j) = (*(ptr_tmp + j) - 1) * scale + 1;
		}
	}

	//**********************************************  RCNN Extract Regions  **********************************
	//Image Crop Initialization
	int   	  	batch_size = 256;
	int   	  	result;

	Crop      	crop_param((char *)"warp", 227, 16);
	char      	cmdbuf0[80];
	char      	cmdbuf1[80];
	char      	cmdbuf2[80];
    char 		cmdbuf3[80];
	char      	crop_image_dir[80];

	printf("Box_num = %d\n", Box_num);

	//convert cropped mat from RGB to BGR
	Mat 	  	mat_bgr;
	Mat 	  	mat_rgb  = iplImageToMat(img, true);
	cvtColor(mat_rgb,mat_bgr,CV_RGB2BGR);
	IplImage  	img_bgr(mat_bgr);

	//cvNamedWindow("BGR-IPL", CV_WINDOW_AUTOSIZE);
	//cvShowImage("BGR-IPL", &img_bgr);
	//cvWaitKey(0);

	for (int j = 0; j < Box_num; j++)
	//for (int j = 0; j < 1000; j++)
	{
		CvMat *input_box = cvCreateMat(1, 4, CV_32FC1);

		ptr = (float*)(input_box->data.ptr + 0 * input_box->step);
		ptr_temp = (float*)(boxes_remove_duplicates->data.ptr + j * boxes_remove_duplicates->step);

		for (int k = 0; k < boxes_remove_duplicates->cols; k++)
		{
			*(ptr + k) = *(ptr_temp + k);
		}		

		CvMat 	image_batches = Rcnn_extract_regions(&img_bgr, input_box, batch_size, crop_param);

		Mat   	cropped_Mat = cvMatToMat(&image_batches, true);

		sprintf(crop_image_dir, "./images/cropped_image_%03d.jpg",j);
		//printf("%s \n", crop_image_dir);
		imwrite(crop_image_dir, cropped_Mat);
	}

	//************************************************    DLA Processing   ***********************************************
	sprintf(cmdbuf0, "rm ./*.dat");
	printf("%s \n", cmdbuf0);
	system(cmdbuf0);	
    	//**********************************************   Put Input into  DLA   *********************************************
	sprintf(cmdbuf1, "./run.py --image_dir  ./images  --dst_dir    ./output   --mean_file  ./mean/image_mean.mat");
	printf("%s \n", cmdbuf1);
	system(cmdbuf1);	

	//*****************************************    Converting .dat into .txt    *************************************************
	sprintf(cmdbuf2, "./convert_dat_into_txt.py");
	printf("%s \n", cmdbuf2);
	system(cmdbuf2);

    //******************************************    Get output data from .txt    *************************************************

    int		box_col_num = TOTAL_CLASS + 4;

    CvMat 		*boxes_scored = cvCreateMat(boxes_remove_duplicates->rows,box_col_num,CV_32FC1);

    printf("get txt data \n");

    char 		buf[2000];
    char 		spliter[] = " ,!";
    char 		*pch;
    int  		j;
    float 		f_num = 0;

#ifdef TEST_WITH_TXT
	for (int ind = 0 ; ind < 1500; ind ++)
#else
    	for (int ind = 0 ; ind < Box_num; ind ++)
#endif
    	{
    		string message;
    		ifstream infile;
    		if(ind < 100)
    			sprintf(cmdbuf3, "./txt/data%03d.txt",ind);
    		else
    			sprintf(cmdbuf3, "./txt/data%d.txt",ind);
    		infile.open(cmdbuf3);

    		ptr = (float*)(boxes_scored->data.ptr + ind * boxes_scored->step);
    		ptr_removed_duplicate = (float*)(boxes_remove_duplicates->data.ptr + ind * boxes_remove_duplicates->step);

    		*(ptr) = *(ptr_removed_duplicate);
    		*(ptr + 1) = *(ptr_removed_duplicate + 1);
    		*(ptr + 2) = *(ptr_removed_duplicate + 2);
    		*(ptr + 3) = *(ptr_removed_duplicate + 3);

    		if(infile.is_open())
    		{
    			while(infile.good() && !infile.eof())
    			{
    				memset(buf,0,2000);
    				infile.getline(buf,2000);
    			}

    			pch = strtok(buf,spliter);

    			j = 4;

    			while(pch != NULL)
    			{
    				f_num=atof(pch);
    	    			*(ptr + j) = f_num;
    				pch = strtok(NULL,spliter);
    				j++;
    			}

    			infile.close();
    		}
	        else
	        {
	        	for(int cnt = 4; cnt < (TOTAL_CLASS + 4); cnt ++)
	        	{
	            		*(ptr + cnt) = 0;		
	        	}		
	        }
    	}
	//************************************************   Post Processing   **********************************   

	//*****************************************   NMS  & DRAW Boxes Processing  **********************************
	printf("Final Stage \n"); 
	double 	threshold = THRESHOLD;
	int    	length_index_reserved = 0;	
	int    	*index_reserved = new int[boxes_scored->rows];

	printf("boxes_scored size is [%d %d] \n",boxes_scored->rows,boxes_scored->cols);

	for (int num_class = 0; num_class < TOTAL_CLASS; num_class++)
	{ 
		length_index_reserved = 0;
		
		for (int i = 0; i < boxes_scored->rows; i++)
		{
			ptr = (float*)(boxes_scored->data.ptr + i * boxes_scored->step);

			if ((*(ptr + 4 + num_class)) > THRESHOLD)
			{
				index_reserved[length_index_reserved] = i;
				length_index_reserved = length_index_reserved + 1;
			}
		}

		//printf("index_reserved done \n");

		if(length_index_reserved > 0)
		{
			CvMat * boxes_thresholded = cvCreateMat(length_index_reserved, 5, CV_32FC1);

			for (int i = 0; i < length_index_reserved; i++)
			{
				ptr      = (float*)(boxes_scored->data.ptr + (index_reserved[i]) * boxes_scored->step);
				ptr_thresholded = (float*)(boxes_thresholded->data.ptr + i * boxes_thresholded->step);

				for (int j = 0; j < (boxes_thresholded->cols - 1); j++)
				{
					*(ptr_thresholded + j) = *(ptr + j);
				}
				*(ptr_thresholded + 4) = *(ptr + 4 + num_class);
			}

			//printf("Class %d boxes_scored after thresholded is %d \n",num_class,length_index_reserved);
			const char* class_type_c2 = classification[num_class].c_str();
			printf("Class %s boxes_scored after thresholded is %d \n",class_type_c2,length_index_reserved);
		
			CvMat * print_box = cvCreateMat(boxes_thresholded->rows, 5, CV_32FC1);
			
			printf("Thresholded boxes: \n");
			PrintCvMatValue(boxes_thresholded);

			print_box  = nms(boxes_thresholded, overlap);

			printf("Print boxes after nms: \n");
			PrintCvMatValue(print_box);

			image_draw = drawboxes(mat, print_box, num_class);			

			CvMat temp2 = image_draw;   
			cvCopy(&temp2, mat);
			flag_box_exist = 1;
		}
		else
		{
			string class_type = classification[num_class];
			const char* class_type_c = class_type.c_str();
			printf("Class %s number is 0 \n",class_type_c);
		}
	}

	Mat image_origin = cvMatToMat(mat, true);

	if(flag_box_exist)
	{
		imwrite("./output/output_image.jpg",image_draw);
		cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
		imshow("Output Image", image_draw);
		cvWaitKey();
	}

	else
	{
		imwrite("./output/output_image.jpg",image_origin);
		cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
		imshow("Output Image", image_origin);
		cvWaitKey();
	}

    	return 0;
}
