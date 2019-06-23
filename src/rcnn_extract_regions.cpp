#include "defines.hpp"
#include "crop.hpp"
#include "rcnn_extract_regions.hpp"

float max_coord(float x1, float x2)
{
    if (x1 > x2)
		return x1;
	else
		return x2;
}

float min_coord(float x1, float x2)
{
	if (x1 <= x2)
		return x1;
	else
		return x2;
}


Mat iplImageToMat(const IplImage* img, bool copyData)
{
	Mat m;

	if (!img)
		return m;

	m.dims = 2;
	CV_DbgAssert(CV_IS_IMAGE(img) && img->imageData != 0);

	int imgdepth = IPL2CV_DEPTH(img->depth);
	size_t esz;
	m.step[0] = img->widthStep;

	if (!img->roi)
	{
		CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL);
		m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, img->nChannels);
		m.rows = img->height;
		m.cols = img->width;
		m.datastart = m.data = (uchar*)img->imageData;
		esz = CV_ELEM_SIZE(m.flags);
	}
	else
	{
		CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL || img->roi->coi != 0);
		bool selectedPlane = img->roi->coi && img->dataOrder == IPL_DATA_ORDER_PLANE;
		m.flags = Mat::MAGIC_VAL + CV_MAKETYPE(imgdepth, selectedPlane ? 1 : img->nChannels);
		m.rows = img->roi->height;
		m.cols = img->roi->width;
		esz = CV_ELEM_SIZE(m.flags);
		m.datastart = m.data = (uchar*)img->imageData +
			(selectedPlane ? (img->roi->coi - 1)*m.step*img->height : 0) +
			img->roi->yOffset*m.step[0] + img->roi->xOffset*esz;
	}

	m.datalimit = m.datastart + m.step.p[0] * m.rows;
	m.dataend = m.datastart + m.step.p[0] * (m.rows - 1) + esz*m.cols;
	m.flags |= (m.cols*esz == m.step.p[0] || m.rows == 1 ? Mat::CONTINUOUS_FLAG : 0);
	m.step[1] = esz;

	if (copyData)
	{
		Mat m2 = m;
		m.release();
		if (!img->roi || !img->roi->coi ||
			img->dataOrder == IPL_DATA_ORDER_PLANE)
			m2.copyTo(m);
		else
		{
			int ch[] = { img->roi->coi - 1, 0 };
			m.create(m2.rows, m2.cols, m2.type());
			mixChannels(&m2, 1, &m, 1, ch, 1);
		}
	}

	return m;
}


CvMat Rcnn_extract_regions(IplImage* image, CvMat* box, int batch_size, Crop crop_param)
{
		CvSize 		size;
		CvMat 		*pMat;
		CvMat  		temp_crop;
		int 		num_batches;
		double 		scale;
		const float *ptr;	
		double 		x1_coord, x2_coord, y1_coord, y2_coord;
		double 		center_x, center_y, width, height, unclipped_height, unclipped_width;
		double 		x1_coord_scaled, x2_coord_scaled, y1_coord_scaled, y2_coord_scaled;
		double 		x1_coord_clipped, x2_coord_clipped, y1_coord_clipped, y2_coord_clipped;
		double 		clipped_width, clipped_height;
		double 		scale_x, scale_y;
		double 		pad_x1,pad_y1;
		double 		crop_width, crop_height;
		double 		temp_width, temp_height;
	
		double 		means_mat_x1;
		double 		means_mat_y1;
		double 		means_mat_width;
		double 		means_mat_height;
		IplImage 	*img_resized;
		CvSize 		dst_cvsize;

		ptr              = (const float*)(box->data.ptr + 0 * box ->step);
		scale            = crop_param.crop_size / (crop_param.crop_size - crop_param.crop_padding * 2);
		x1_coord         = (*ptr);
		x2_coord         = (*(ptr + 2));
		y1_coord         = (*(ptr + 1));
		y2_coord         = (*(ptr + 3));

		width            = x2_coord - x1_coord + 1;
		height           = y2_coord - y1_coord + 1;

		center_x         = x1_coord + width / 2;
		center_y         = y1_coord + height / 2;

		x1_coord_scaled  = center_x - width  / 2 * scale;
		x2_coord_scaled  = center_x + width  / 2 * scale;
		y1_coord_scaled  = center_y - height / 2 * scale;
		y2_coord_scaled  = center_y + height / 2 * scale;

		unclipped_height = y2_coord_scaled - y1_coord_scaled + 1;
		unclipped_width  = x2_coord_scaled - x1_coord_scaled + 1;

		x1_coord_clipped = max_coord(1, x1_coord_scaled);
		x2_coord_clipped = min_coord(image->width, x2_coord_scaled);
		y1_coord_clipped = max_coord(1, y1_coord_scaled);
		y2_coord_clipped = min_coord(image->height, y2_coord_scaled);

		clipped_height   = y2_coord_clipped - y1_coord_clipped + 1;
		clipped_width    = x2_coord_clipped - x1_coord_clipped + 1;

		scale_x          = crop_param.crop_size / unclipped_width;
		scale_y          = crop_param.crop_size / unclipped_height;
		crop_width       = round(clipped_width  * scale_x);
		crop_height      = round(clipped_height * scale_y);

		pad_x1           = max_coord(1, 1 - x1_coord_scaled);
		pad_x1           = round(pad_x1 * scale_x);
		pad_y1           = max_coord(1, 1 - y1_coord_scaled);
		pad_y1           = round(pad_y1 * scale_y);      

		if ((pad_y1 + crop_height) >  crop_param.crop_size)
			crop_height  = crop_param.crop_size - pad_y1;
		if ((pad_x1 + crop_width)  >  crop_param.crop_size)
			crop_width   = crop_param.crop_size - pad_x1;

		temp_width  = x2_coord_clipped - x1_coord_clipped;
		temp_height = y2_coord_clipped - y1_coord_clipped;

		CvRect capture_rect = cvRect(x1_coord_clipped, y1_coord_clipped, temp_width, temp_height);
		pMat = cvCreateMat(crop_height, crop_width, CV_32FC3);
		cvGetSubRect(image, pMat, capture_rect);

		//******************************************** Image reseize to cropped size ****************************************
		dst_cvsize.width  = crop_width;
		dst_cvsize.height = crop_height;
		img_resized = cvCreateImage(dst_cvsize, image->depth, image->nChannels);

		cvResize(pMat, img_resized, CV_INTER_LINEAR);

		//**************************** set crop size image to crop size in crop_param with padding *******************************
		IplImage* img_out = cvCreateImage(cvSize(crop_param.crop_size, crop_param.crop_size), image->depth, image->nChannels);
		cvZero(img_out);

		CvRect roi = cvRect(pad_x1,pad_y1,(pad_x1 + crop_width),(pad_y1 + crop_height));
		cvSetImageROI(img_out, roi); 
		cvCopy(img_resized, img_out); 
		cvResetImageROI(img_out);		
		
		CvMat* temp_mat = cvGetMat(image, &temp_crop);
		
		//************************************************** image transpose  ********************************************
		Mat src  = iplImageToMat(img_out, true);

		Mat dst;  
		
    	Mat map_x;  
    	Mat map_y; 		

    	dst.create(src.size(),src.type());   
    	map_x.create( src.size(), CV_32FC1);  
    	map_y.create( src.size(), CV_32FC1);  
    	for( int i = 0; i < src.rows; ++i)  
    	{  
        	for( int j = 0; j < src.cols; ++j)  
        	{  
            	map_x.at<float>(i, j) = (float) i;//j;//(src.cols - j) ;  
            	map_y.at<float>(i, j) = (float) j;//(src.rows - i) ;  
        	}  
    	}  
    	remap(src, dst, map_x, map_y, CV_INTER_LINEAR); 
    	
		CvMat result_mat = dst;  

		return result_mat;
}
