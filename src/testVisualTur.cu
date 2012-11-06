#include "visualTur.hpp"
#include "FreeImage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include "stdlib.h"

int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		std::cerr<<"Error, testVisualTur hdf5_file dataset_name octree_file [device]"<<std::endl;
		return 0;
	}

	int device = 0;
	if (argc > 4)
	{
		device = atoi(argv[4]);
		cudaSetDevice(device);
	}

	visualTurParams_t params;
	params.W = 800;
	params.H = 800;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 1000;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 2;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	visualTur * VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(52.0f, 52.0f, 520.0f));
	VisualTur->camera_MoveForward(1.0f);

	FreeImage_Initialise();

	float * screenG = 0;
	float * screenC = new float[800*800*4];
	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*800*800*4))<<std::endl;

	VisualTur->updateVisibleCubes(5, screenG);

	std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*800*800*4, cudaMemcpyDeviceToHost))<<std::endl;

	unsigned char * picture = new unsigned char[800*800*3];
	for(int i=0; i<800; i++)
		for(int j=0; j<800; j++)
		{
			//std::cout<<screenC[i*4]<<" "<<screenC[i*4+1]<<" "<<screenC[i*4+2]<<std::endl;
			picture[i*3] = screenC[i*4]*255;
			picture[i*3+1] = screenC[i*4+1]*255;
			picture[i*3+2] = screenC[i*4+2]*255;
		} 

	FIBITMAP *img = FreeImage_ConvertFromRawBits(picture, 800, 800, 800 * 3, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);
	FreeImage_Save(FIF_PNG, img, "prueba.png", 0);

	FreeImage_DeInitialise();

	cudaFree(screenG);
	delete[] screenC;
	delete[] picture;
	delete VisualTur;
}
