NVCC=nvcc
OPT=-O0
CUDACP=sm_13
CFLAGS=$(OPT) -g -Wall
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(CFLAGS)'
INCLUDE=-Iinc/
LIBRARY=-lm -lhdf5 -lGL -lglut -lGLU

all: Objects testPrograms 

Objects: obj/Screen.o obj/Camera.o obj/FileManager.o obj/lruCache.o obj/Octree.o obj/rayCaster.o obj/visualTur.o 

obj/Screen.o: src/Screen.cpp inc/Screen.hpp
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/Screen.cpp -o obj/Screen.o
obj/Camera.o: src/Camera.cu inc/Camera.hpp inc/Screen.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/Camera.cu -o obj/Camera.o
obj/FileManager.o: src/FileManager.cu inc/FileManager.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/FileManager.cu -o obj/FileManager.o
obj/lruCache.o: src/lruCache.cu inc/lruCache.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/lruCache.cu -o obj/lruCache.o
obj/Octree.o: src/Octree.cu inc/Octree.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/Octree.cu -o obj/Octree.o
obj/rayCaster.o: src/rayCaster.cu inc/rayCaster.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/rayCaster.cu -o obj/rayCaster.o
obj/visualTur.o: src/visualTur.cu inc/visualTur.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/visualTur.cu -o obj/visualTur.o

testPrograms: bin/testVisualTur

bin/testVisualTur: Objects src/testVisualTur.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/Screen.o obj/Camera.o obj/FileManager.o  obj/lruCache.o obj/Octree.o obj/rayCaster.o obj/visualTur.o src/testVisualTur.cu  -o bin/testVisualTur $(LIBRARY)

clean:
	-rm bin/* obj/* ./*.i ./*.ii ./*.cudafe* ./*.fatbin* ./*.hash ./*.module_id ./*.ptx ./*sm*.cubin ./*.o
