#include "defines.hpp"
#include "drawboxes.hpp"

extern string classification[200];

Mat cvMatToMat(const CvMat* m, bool copyData)
{
	Mat thiz;

	if (!m)
		return thiz;

	if (!copyData)
	{
		thiz.flags = Mat::MAGIC_VAL + (m->type & (CV_MAT_TYPE_MASK | CV_MAT_CONT_FLAG));
		thiz.dims = 2;
		thiz.rows = m->rows;
		thiz.cols = m->cols;
		thiz.datastart = thiz.data = m->data.ptr;
		size_t esz = CV_ELEM_SIZE(m->type), minstep = thiz.cols*esz, _step = m->step;
		if (_step == 0)
			_step = minstep;
		thiz.datalimit = thiz.datastart + _step*thiz.rows;
		thiz.dataend = thiz.datalimit - _step + minstep;
		thiz.step[0] = _step; thiz.step[1] = esz;
	}
	else
	{
		thiz.datastart = thiz.dataend = thiz.data = 0;
		Mat(m->rows, m->cols, m->type, m->data.ptr, m->step).copyTo(thiz);
	}

	return thiz;
}


string DoubleToString(double Input)
{
	stringstream Oss;
	Oss << Input;
	return Oss.str();
}

Mat drawboxes(CvMat* im, CvMat* box, int num_class)
{
	CvFont	     font;
	double	     hScale = 0.3;
	double	     vScale = 0.3;
	int          lineWidth = 1;
	const float* ptr;
	double	     width;
	double	     height;

	if (im->rows == 0 && im->cols == 0)
		std::cout << "image is empty" << endl;
	
	CvPoint pt1, pt2, origin;

	for (int i = 0; i < box->rows; i++)
	{
		ptr = (const float*)(box->data.ptr + i * box->step);
		double score = *(ptr + 4);

		string str_score = DoubleToString(score);
		string text_scored;

		string text = classification[num_class] + " Score:  ";
		text_scored = text + str_score;
		const char* text_scored_c = text_scored.c_str();

		std::cout << text_scored << endl;
		
		height = *(ptr + 3) - *(ptr + 1);
		width  = *(ptr + 2) - *(ptr);
		CvRect rect = cvRect(*(ptr), *(ptr + 1), width, height);

		pt1.x = rect.x;
		pt1.y = rect.y;
		pt2.x = rect.x + rect.width;
		pt2.y = rect.y + rect.height;
		origin.x = pt1.x;
		origin.y = pt1.y + 10;
		cvRectangle(im, pt1, pt2, CV_RGB( 0, 300, 0), 1);

		//Print Scores on image 
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);

		cvPutText(im, text_scored_c, origin, &font, CV_RGB( 0, 300, 0));
	}

	//Mat image = im;
	Mat image = cvMatToMat(im, true);
	
	return image;
}
