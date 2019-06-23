#include "defines.hpp"
#include "crop.hpp"

Crop::Crop()
{
	std::cout << "Default crop called\n";
	crop_mode    = (char *)"warp";
	crop_size    = 227;
	crop_padding = 16;	
}

Crop::Crop(char* cm, int cs, int pa)
{
	std::cout << "Mannually setting crop called\n";
	crop_mode    =  cm;
	crop_size    =  cs;
	crop_padding =  pa;
}

Crop::~Crop()
{
//	std::cout << "Crop released\n";
}
