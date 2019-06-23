#ifndef _MAIN_HPP_
#define _MAIN_HPP_

#include "selective_search_boxes.hpp"
#include "filterBoxesWidth.hpp"
#include "boxRemoveDuplicates.hpp"
#include "softmax_layer.hpp"
//#include "rcnn_extract_regions.hpp"

#include "crop.hpp"
extern Mat iplImageToMat(const IplImage* img, bool copyData);
extern CvMat Rcnn_extract_regions(IplImage* image, CvMat* box, int batch_size, Crop crop_param);

#include "nms.hpp"
#include "drawboxes.hpp"

const uint32_t ms_version = 0;

string classification[200] = {
                               "accordion",                 "airplane",              "ant",             "antelope",                   "apple",
	                           "armadillo",		           "artichoke",		         "axe",             "baby bed",                "backpack",
								   "bagel",		        "balance beam",	          "banana",             "band aid",                   "banjo",
								"baseball",		          "basketball",	     "bathing cap",	              "beaker",	                   "bear",
								     "bee",	   	         "bell pepper",		       "bench",              "bicycle",	            	 "binder",
								    "bird",		           "bookshelf",		     "bow tie",	             	 "bow",		               "bowl",
							   "brassiere",		             "burrito",	             "bus",	           "butterfly",		              "camel",
							  "can opener",		                 "car",	            "cart",		          "cattle",		              "cello",									  
                               "centipede",	               "chain saw",	           "chair",		           "chime",		    "cocktail shaker",
							"coffee maker",	       "computer keyboard",	  "computer mouse",	           "corkscrew",		              "cream",
							"croquet ball",		              "crutch",	        "cucumber",	    	  "cup or mug",		             "diaper",
						   "digital clock",		          "dishwasher",	             "dog",	    	"domestic cat",		          "dragonfly",
								    "drum",		            "dumbbell",	    "electric fan",	    	    "elephant",		        "face powder",
								     "fig",		      "filing cabinet",	      "flower pot",		           "flute",		                "fox",
							 "french horn",		                "frog",	      "frying pan",		     "giant panda",		           "goldfish",
							   "golf ball",		            "golfcart",	       "guacamole",		          "guitar",		         "hair dryer",
							  "hair spray",		            "hamurger",			  "hammer",		         "hamster",		          "harmonica",
								    "harp",		"hat with a wide brim",		"head cabbage",		          "helmet",	    	   "hippopotamus",
						  "horizontal bar",			           "horse",		      "hotdog",			        "iPod",		             "isopod",
							   "jellyfish",			      "koala bear",			   "ladle",		         "ladybug",		    		   "lamp",
								  "laptop",			           "lemon",			    "lion",		        "lipstick",		    		 "lizard",
							     "lobster",			    	 "maillot",		      "maraca",		      "microphone",			      "microwave",
							    "milk can",				   "miniskirt",			  "monkey",		      "motorcycle",		    	   "mushroom",
							        "nail",			      "neck brace",			    "oboe",			      "orange",		              "otter",
						      "pencil box",	        "pencil sharpener",	         "perfume",		          "person",		    		  "piano",
						       "pineapple",			  "ping-pong ball",		     "pitcher",		           "pizza",		        "plastic bag",
						      "plate rack",			     "pomegranate",		    "popsicle",		       "porcupine",		    	"power drill",
				         		 "pretzel",			         "printer",	            "puck",		    "punching bag",		              "purse",
								  "rabbit",		              "racket",	             "ray",	           "red panda",		       "refrigerator",
						  "remote control",			   "rubber eraser",		  "rugby ball",		           "ruler",   "salt or pepper shaker",
							   "saxophone",				    "scorpion",	  	  "screwdriver",		        "seal",			          "sheep",
							         "ski",		               "skunk",	 	        "snail",			   "snake",			     "snowmobile",
							    "snowplow",			  "soap dispenser",	      "soccer ball",			    "sofa",			        "spatula",
				                "squirrel",  		        "starfish",		  "stethoscope",	    	   "stove",		           "strainer",			
							  "strawberry",				   "stretcher",        "sunglasses",     "swimming trunks",                   "swine", 
							     "syringe",					   "table",       "tape player",         "tennis ball",                    "tick",
								     "tie",                    "tiger",           "toaster",       "traffic light",                   "train", 
							    "trombone",					 "trumpet",            "turtle",       "tv or monitor",                "unicycle",                  
								  "vacuum",					  "violin",        "volleyball",         "waffle iron",                  "washer",            
					        "water bottle",				  "watercraft",			    "whale",		 "wine bottle",			          "zebra"
							};


void PrintCvMatValue(CvMat* im)
{
	const float* ptr;

	for (int i = 0; i < im->rows; i++)
	{
		ptr = (const float*)(im->data.ptr + i * im->step);//stepÊÇ×Ö½ÚÊý£¬ËùÒÔÊ×µØÖ·mat.dataÒªÓÃuchar*ÀàÐÍ£¨¼´mat.data.ptr£©£¬  
		//calculate row address and converting into real data type(float*) for further calculation
		printf("mat_row[%d]: ", i);
		for (int j = 0; j < im->cols; j++)
		{
			printf("%f  ",*(ptr + j));
		}
		printf("\n");
	}
}


#endif
