# RCNN-NVDLA-SJTU-DEMO
========================

NV-RCNN Demo
----------------------

Created by Hongli Zheng and YuXin Qian at Nvidia Deep Learning Institute & Shanghai Jiaotong Univerisity.

Introduction
----------------------

NVDLA & SJTU Open Source RCNN Demo is a successful neural network demo merging RCNN into NVDLA hardware platform. It is the first implementation of neural network on embedded system hardware IP for edge computation all over the world.

We choose RCNN as our demo since it is a classical object detection application. RCNN was implemented by Ross Girshick, Jeff Donahue, Trevor Darrel and Jitendra Malik at UC Berkeley EECS.  It uses AlexNet, the most popular benchmarking network, as its network to solve classification problems and archieved a superb performance on the object detection and classification. Nevertheless, RCNN suffers the problem of computation cost and time consumption.

To deal with this problem, we separated the modules and layers of RCNN into pieces. After that, we put the convolution layers, pooling layers, activation layers, LRN layers and Reshape function into NVDLA for acceleration and implement the others in C++ for embedded system. Through this method, NVDLA solves the problem of time-consuming in the computation of these layers and gives us a precise and a real-time prediction of object detection. It can also be smoothly migrated into an embedded system after being implemented on a SoC chip.

We have tested our demo on PASCAL VOC 2007, 2010, 2012 and ILSVRC13. The accuracy of the detection in our experiment is affirmative as RCNN.

Meanwhile, you can replace RCNN model with any of your own deep learning neural network by referring our code. NVDLA has been widly adopted on automatic pilot, face detection, gesture detection, medical application and many other computer vision field. 

License
----------------------

The use of this software is RESTRICTED to non-commercial research and educational purposes. 

NV RCNN Demo is released under the License of NVIDIA’s Open NVDLA and Shanghai JiaoTong University. 

If you find NV R-CNN useful in your research, please consider citing:

@inproceddings{

    Author = {Hongli Zheng, YuXin Qian, Dazhi He, Nvidia Deep Learning Accelerator}
  
    Title = {}
  
    Booktitle = {}
  
    Year = {2018}
}


Performance
----------------------

With this demo, you can figure out how to merge your own neural network in NVDLA hardware platform to accelerate the computation of your network. By using NVDLA, the speed of the computation can be inferred as follows:



NV R-CNN
----------------------

### Requirements: software

  1. Requirements for Caffe and pycaffe (see: Caffe installation instructions) 
    
      Note: Caffe must be built with support for Python layers!
  
    # In your Makefile.config, make sure to have this line uncommented
    
    WITH_PYTHON_LAYER := 1
       
  2. Python packages you might not have: cython, python-opencv, easydict
  
  3. [optional] MATLAB (required for PASCAL VOC evaluation only)
  
### Requirements: hardware

  1. For training smaller networks (CaffeNet, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
    
  2. For training with VGG16, you'll need a K40 (~11G of memory) 
    
  3. You can also use CPU only, while the training process will cost several hours
    
    
### Installation (sufficient for the demo)

  1. Clone the NV R-CNN Demo repository
  
    git clone 
     
  2. We'll call the directory that you cloned NV-RCNN into NVRCN_ROOT
  
  3. Build Caffe and pycaffe
    
    cd $NVRCN_ROOT/dla_amod
    
   #Now follow the Caffe installation instructions here:
   #http://caffe.berkeleyvision.org/installation.html

   #If you're experienced with Caffe and have all of the requirements installed
   #and your Makefile.config in place, then simply do:

    make -j8 && make pycaffe

  4. Put the image to detect into test folder and modify the path of it in main.cpp.
    
    cd $NVRCN_ROOT
    mv your_image.jpg $NVRCN_ROOT/images/
    vim $NVRCN_ROOT/main.cpp

   Modify  $NVRCN_ROOT/src/main.cpp   line 19 
   
    img  =  cvLoadImage("./test_image/person-bike.jpg");
    
   into
  
    img = cvLoadImage("./test_image/your_image.jpg");
   
  5. Put your .protxt file and trained model into caffe folder and modify the path of it in Python script.
    
    cd $NVRCN_ROOT
    mv your_prototxt.prototxt      $NVRCN_ROOT/dla_amod/models/your_model/
    mv your_caffemodel.caffemodel  $NVRCN_ROOT/dla_amod/models/your_model/

   And modify the line 119 and 121 of  $NVRCN_ROOT/run_amod.py

   Modify line 119 

    weights = os.path.join(_args.amod_dir, 'models/…')

   into

    weights = os.path.join(_args.amod_dir, 'models/your_model/your_caffemodel.caffemodel')

   And modify line 121
  
    weights = os.path.join(_args.amod_dir, 'models/…')
    
   into
  
    weights =  os.path.join(_args.amod_dir,'models/your_model/your_prototxt.prototxt')
    
  6. Go to $NVRCN_ROOT and make
    
    cd $NVRCN_ROOT
    make
	
  7. Run $NVRCN_ROOT/main and you will get the object detection with bounding boxes of your image!


NVDLA
---------------------------------

The NVIDIA Deep Learning Accelerator(NVDLA) is a free and open architecture that promotes a standard way to design deep learning inference accelerators. With its modular architecture, NVDLA as scalable, highly configurable, and designed to simplify integration and portability. Learn more about NVDLA on the project web page.

http://nvdla.org/

Acknowledgement
@inproceeding{

  girshick14CVPR,
  
	Author = {Girshick, Ross and Donahue, Jeff and Darrell, Trevor and Malik, Jitendra},
  
	Title = {Rich feature hierarchies for accurate object detection and semantic segmentation},
  
	Booktitle = {Computer Vision and Pattern Recognition}
  
	Year = {2014}
  
}

@inproceeding{

    NVIDIA Open NVDLA License and Agreement,
  
    Author = {NVIDIA},
  
    Year = {2018}
  
}



