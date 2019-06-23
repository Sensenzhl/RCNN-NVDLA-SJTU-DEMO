#ifndef _CROP_HPP_
#define _CROP_HPP_

#include "includes.hpp"
class Crop
{
private:

public:
	char    *crop_mode;
	int     crop_size;
	int     crop_padding;

	Crop();
	Crop(char* crop_mode, int crop_size, int padding);
	~Crop();
};

#endif
