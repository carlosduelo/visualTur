#include "visualTur.hpp"
#include "FreeImage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
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

	int W = 1024;
	int H = 1024;

	visualTurParams_t params;
	params.W = W;
	params.H = H;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 2000;
	params.maxElementsCache_CPU = 15000;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 2;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	visualTur * VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(512.0f, 152.0f, 200.0f));
	VisualTur->camera_MoveForward(1.0f);

	FreeImage_Initialise();

	float * screenG = 0;
	float * screenC = new float[H*W*4];

	for(int i=0; i<H; i++)
		for(int j=0; j<W; j++)
		{
			int id = i*W + j;
			screenC[id*4] = 0.0;
			screenC[id*4+1] = 0.0f;
			screenC[id*4+2]= 0.0f;
			screenC[id*4+3]= 0.0f;
		} 
	
	FIBITMAP * bitmap = FreeImage_Allocate(H,W,24);
	RGBQUAD color;

	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*H*W*4))<<std::endl;

	std::cerr<<"Cuda mem set: "<<cudaGetErrorString(cudaMemset((void *)screenG,0,sizeof(float)*H*W*4))<<std::endl;		

	double total =0;
	struct timeval st, end;
	gettimeofday(&st, NULL);
	for(int m=0; m<100; m++)
	{ 
		for(int i=0; i<H; i++)
			for(int j=0; j<W; j++)
			{
				int id = i*W + j;
				screenC[id*4] = 0.0;
				screenC[id*4+1] = 0.0f;
				screenC[id*4+2]= 0.0f;
				screenC[id*4+3]= 0.0f;
			} 

		VisualTur->updateVisibleCubes(8, screenG);

		std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*W*H*4, cudaMemcpyDeviceToHost))<<std::endl;

		int hits =0;
		for(int i=0; i<H; i++)
			for(int j=0; j<W; j++)
			{
				int id = i*W + j;
				if (screenC[id*4]!=0.0f || screenC[id*4+1]!=0.0f || screenC[id*4+2]!=0.0f)
					hits++;
				color.rgbRed = screenC[id*4]*255;
				color.rgbGreen = screenC[id*4+1]*255;
				color.rgbBlue = screenC[id*4+2]*255;
				FreeImage_SetPixelColor(bitmap, i, j, &color);
			} 
		std::cout<<"--->"<<hits<<std::endl;
		std::stringstream name;
		name<<"prueba"<<m<<".png";
		FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);
		VisualTur->camera_MoveForward(10.0f);
	}
	gettimeofday(&end, NULL);
	double delta = ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
	std::cout << "Time elapsed: " << delta << " sec"<< std::endl;

	cudaFree(screenG);
	delete[] screenC;
	FreeImage_DeInitialise();
}
