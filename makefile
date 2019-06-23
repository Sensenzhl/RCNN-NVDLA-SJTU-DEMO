CXX      = g++
CXXFLAGS = -std=c++11 -g -lm `pkg-config --cflags --libs opencv` # 附加参数 pkg-config --cflags --libs opencv 包含opencv所有lib和include，可以在cmd中输入pkg-config --cflags --libs opencv查看
INC_PATH = -I ./inc  
LIBS     = -L ./ \
           -lcaffe
SRCFILES = src/*.cpp                                		# 指定目录 ./ 下的所有后缀是cpp的文件全部展开。
OBJS     = $(SRCFILES:.cpp=.o)                                 # OBJS将$(SRCS)下的.cpp文件转化为.o文件
TARGET   = main                                                # 输出程序名称

all: main

$(TARGET): $(OBJS)
	$(CXX) ./*.o $(CXXFLAGS) $(INC_PATH) -o $(TARGET)
#	$(CXX) $(CXXFLAGS) ./*.o $(INC_PATH) -o $(TARGET)

$(OBJS): $(SRCFILES) 
	$(CXX) $(CXXFLAGS) -c $(SRCFILES) $(INC_PATH)

#$(OBJS): $(SRCFILES) $(INC_PATH)/*.h
#	$(CXX) $<  $(CXXFLAGS) $(LIBS) -c -o $(TARGET) 

clean:
	rm -rf *.out *.o $(OBJS)

