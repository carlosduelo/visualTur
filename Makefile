NVCC=nvcc
CUDACP=sm_13
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(OPT) -g -Wall -fopenmp'
INCLUDE=-Iinc/
CPU_LIBRARY=-lm -lhdf5 -lGL -lglut -lGLU

all: Objects 

Objects: obj/Screen.o obj/Camera.o obj/FileManager.o obj/lruCache.o obj/Octree.o obj/visualTur.o 

obj/Screen.o: src/Screen.cpp inc/Screen.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/Screen.cpp -o obj/Screen.o
obj/Camera.o: src/Camera.cpp inc/Camera.hpp inc/Screen.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/Camera.cpp -o obj/Camera.o
obj/FileManager.o: src/FileManager.cpp inc/FileManager.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/FileManager.cpp -o obj/FileManager.o
obj/lruCache.o: src/lruCache.cpp inc/lruCache.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/lruCache.cpp -o obj/lruCache.o
obj/Octree.o: src/Octree.cpp inc/Octree.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/Octree.cpp -o obj/Octree.o
obj/visualTur.o: src/visualTur.cpp inc/visualTur.hpp
	$(NVCC) -c -x 'cu' $(NFLAGS) $(INCLUDE) src/visualTur.cpp -o obj/visualTur.o



clean:
	-rm bin/* obj/* ./*.i ./*.ii ./*.cudafe* ./*.fatbin* ./*.hash ./*.module_id ./*.ptx ./*sm*.cubin ./*.o
