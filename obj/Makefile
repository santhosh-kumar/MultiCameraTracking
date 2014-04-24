#Compiler
CC=g++

#IPP=/opt/intel/ipp/6.1.2.051/em64t#IPP location
IPP=/opt/intel/composer_xe_2013_sp1/ipp

#library
#for opencv 2.1
#LDFLAGS=-lcxcore -lcv -lhighgui -lippiem64t -lippmem64t -lippcvem64t -lippcoreem64t  -liomp5 -lpthread -lm
#for opencv2.2 and 2.3 onwards, use the following
LDFLAGS=-L$(IPP)/lib/intel64 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_ml -lippi -lipps -lippcv -lippcore

#headers
CFLAGS=-c -w -I$(IPP)/include -I/usr/local/include/opencv -I/usr/local/include/ -I/usr/local/include/boost

SRCDIR=../src
OBJDIR=./

SOURCES=$(wildcard $(SRCDIR)/*.cpp)

OBJECTS_TMP := $(SOURCES:.cpp=.o)
OBJECTS := $(subst src,obj, $(OBJECTS_TMP))

EXECUTABLE = Tracker

all: $(OBJECTS) $(EXECUTABLE)

clean: 
	rm -rf $(EXECUTABLE)
	rm ./*.o

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS)  $(LDFLAGS) -o $@

$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES)
