#include "visualTur.hpp"
#include "FreeImage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
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

	for(int i=0; i<800; i++)
		for(int j=0; j<800; j++)
		{
			int id = i*800 + j;
			screenC[id*4] = 0.0;
			screenC[id*4+1] = 0.0f;
			screenC[id*4+2]= 0.0f;
			screenC[id*4+3]= 0.0f;
		} 
	
	FIBITMAP * bitmap = FreeImage_Allocate(800,800,24);
	RGBQUAD color;

	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*800*800*4))<<std::endl;

	std::cerr<<"Cuda mem set: "<<cudaGetErrorString(cudaMemset((void *)screenG,0,sizeof(float)*800*800*4))<<std::endl;		


	for(int it=0; it<10; it++)
	{
		VisualTur->updateVisibleCubes(5, screenG);

		std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*800*800*4, cudaMemcpyDeviceToHost))<<std::endl;

		int hits =0;
		for(int i=0; i<800; i++)
			for(int j=0; j<800; j++)
			{
				int id = i*800 + j;
				if (screenC[id*4]!=0.0f || screenC[id*4+1]!=0.0f || screenC[id*4+2]!=0.0f)
					hits++;
				/*
				picture[i*3] = screenC[i*4]*255;
				picture[i*3+1] = screenC[i*4+1]*255;
				picture[i*3+2] = screenC[i*4+2]*255;
				color.rgbRed = 0;
				color.rgbGreen = (double)i/800*255.0;
				color.rgbBlue = (double)j/800*255.0;
				*/
				color.rgbRed = screenC[id*4]*255;
				color.rgbGreen = screenC[id*4+1]*255;
				color.rgbBlue = screenC[id*4+2]*255;
				FreeImage_SetPixelColor(bitmap, i, j, &color);
			} 
		std::cout<<"--->"<<hits<<std::endl;
		std::stringstream name;
		name<<"prueba"<<it<<".png";
		FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);
	}

	cudaFree(screenG);
	delete[] screenC;
	FreeImage_DeInitialise();
}
