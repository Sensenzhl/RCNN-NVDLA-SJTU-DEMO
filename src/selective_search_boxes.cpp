#include "defines.hpp"
#include "selective_search_boxes.hpp"

using namespace cv;
using namespace std;

CvMat *selective_search_boxes(IplImage* img, bool fast_mode, double im_width)
{
	double		scale;
	IplImage *	img_resized;
	IplImage *	scr = 0;
	CvMat		temp;
	CvMat		temp2;
	CvMat*		mat = cvGetMat(img, &temp);
	
	CvSize		dst_cvsize;
	scale 		  = mat->cols / im_width;
	dst_cvsize.width  = im_width;
	dst_cvsize.height = int(img->height / scale);
	img_resized = cvCreateImage(dst_cvsize, img->depth, img->nChannels);
	CvMat*  priority;

	int    ks[4];
	int    k;
	int    minSize;
	char   color_used;
	double sigma = 0.8;	

	printf("cols = %d\n", mat->cols);
	printf("im_width = %f\n", im_width);
	printf("scale = %f\n", scale);
	
	if (scale != 0)
	{
		cvResize(img, img_resized, CV_INTER_LINEAR);
	}
	
	Mat temp_Mat = cvMatToMat(mat, true);
	CvMat* mat_resized = cvGetMat(img_resized, &temp2);
	Mat temp_Mat_resized = cvMatToMat(mat_resized, true);	

	Mat matimg_resized = cvarrToMat(mat_resized);
	Mat graph_segmented;

	// Îªselective searchËã·¨Ìí¼ÓÍ¼¸îËã·¨´¦Àí½á¹û
	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	ss->setBaseImage(matimg_resized);
	ss->switchToSelectiveSearchFast();
	//ss->switchToSelectiveSearchQuality();

	// ×Ô¶¨Òå²ßÂÔ
	Ptr<SelectiveSearchSegmentationStrategy> sss_color   = createSelectiveSearchSegmentationStrategyColor();	// ÑÕÉ«ÏàËÆ¶È²ßÂÔ
	Ptr<SelectiveSearchSegmentationStrategy> sss_texture = createSelectiveSearchSegmentationStrategyTexture();	// ÎÆÀíÏàËÆ¶È²ßÂÔ
	Ptr<SelectiveSearchSegmentationStrategy> sss_size    = createSelectiveSearchSegmentationStrategySize();	// ³ß´çÏàËÆ¶È²ßÂÔ
	Ptr<SelectiveSearchSegmentationStrategy> sss_fill    = createSelectiveSearchSegmentationStrategyFill();	// Ìî³äÏàËÆ¶È²ßÂÔ
	
	// Ìí¼Ó²ßÂÔ
	Ptr<SelectiveSearchSegmentationStrategy> sss = createSelectiveSearchSegmentationStrategyMultiple(sss_color, sss_texture, sss_size, sss_fill);	// ºÏ²¢ÒÔÉÏ4ÖÖ²ßÂÔ
	ss->addStrategy(sss);
	vector<Rect> regions;
	ss->process(regions);	// ´¦Àí½á¹û
	std::cout << "Total Number of Region Proposals: " << regions.size() << endl;

	// ÏÔÊ¾½á¹û
	Mat show_img_resized = matimg_resized.clone();
	for (vector<Rect>::iterator it_r = regions.begin(); it_r != regions.end(); ++it_r)
	{
		rectangle(show_img_resized, *it_r, Scalar(0, 0, 255), 3);
	}

	cvNamedWindow("Searched_Image", CV_WINDOW_KEEPRATIO);
	imshow("Searched_Image", show_img_resized);
	cvWaitKey(0);

	//cvDestroyWindow("Ejemplo");

	CvMat* boxes = cvCreateMat(regions.size(), 4, CV_32FC1);
			
	for (int i = 0; i < boxes -> rows; i++)
	{
		//printf("x=%d y= %d width= %d height=%d\n", regions[i].x,regions[i].y,regions[i].width,regions[i].height);
		float *ptr  = (float*)(boxes->data.ptr + i * boxes->step);
		*(ptr)      = regions[i].x;
		*(ptr + 1)  = regions[i].y;
		*(ptr + 2)  = regions[i].x + regions[i].width;
		*(ptr + 3)  = regions[i].y + regions[i].height;
	}

	return boxes;
}
