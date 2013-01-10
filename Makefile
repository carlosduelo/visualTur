NVCC=nvcc
OPT=-O0
CUDACP=sm_20
CFLAGS=$(OPT) -g -Wall
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(CFLAGS)'
INCLUDE=-Iinc/
LIBRARY=-lm -lhdf5 -lGL -lglut -lGLU -lfreeimage 

all: Objects testPrograms utils

Objects: obj/Screen.o obj/Camera.o obj/FileManager.o obj/lruCache.o obj/Octree.o obj/rayCaster.o obj/Octree_thread.o obj/Octree_device.o obj/visualTur_thread.o obj/visualTur_device.o 

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
obj/Octree_thread.o: src/Octree_thread.cu inc/Octree_thread.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/Octree_thread.cu -o obj/Octree_thread.o
obj/Octree_device.o: src/Octree_device.cu inc/Octree_device.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/Octree_device.cu -o obj/Octree_device.o
obj/rayCaster.o: src/rayCaster.cu inc/rayCaster.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/rayCaster.cu -o obj/rayCaster.o
obj/visualTur.o: src/visualTur.cu inc/visualTur.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/visualTur.cu -o obj/visualTur.o
obj/visualTur_thread.o: src/visualTur_thread.cu inc/visualTur_thread.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/visualTur_thread.cu -o obj/visualTur_thread.o
obj/visualTur_device.o: src/visualTur_device.cu inc/visualTur_device.hpp
	$(NVCC) -c  $(NFLAGS) $(INCLUDE) src/visualTur_device.cu -o obj/visualTur_device.o

testPrograms: bin/testVisualTur_device bin/testFileManager

bin/testFileManager: Objects src/testFileManager.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/FileManager.o src/testFileManager.cu  -o bin/testFileManager $(LIBRARY)

bin/testVisualTur_device: Objects src/testVisualTur_device.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/Screen.o obj/Camera.o obj/FileManager.o  obj/lruCache.o obj/Octree_thread.o obj/Octree_device.o obj/rayCaster.o obj/visualTur_device.o obj/visualTur_thread.o src/testVisualTur_device.cu  -o bin/testVisualTur_device $(LIBRARY)

utils: bin/cutFile

bin/cutFile: Objects src/cutFile.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/FileManager.o src/cutFile.cu  -o bin/cutFile $(LIBRARY)

clean:
	-rm bin/* obj/* ./*.i ./*.ii ./*.cudafe* ./*.fatbin* ./*.hash ./*.module_id ./*.ptx ./*sm*.cubin ./*.o
