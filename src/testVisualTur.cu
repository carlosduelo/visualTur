#include "visualTur.hpp"
#include "FreeImage.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv)
{
	visualTurParams_t params;
	params.W = 800;
	params.H = 800;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 100;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 1;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	visualTur * VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(52.0f, 52.0f, 520.0f));
	VisualTur->camera_MoveForward(1.0f);

	FreeImage_Initialise();

	float * screenG = 0;
	float * screenC = new float[800*800*4];
	cudaMalloc((void**)screenG, sizeof(float)*800*800*4);

	VisualTur->updateVisibleCubes(5.0, screenG);

	cudaMemcpy((void*) screenG, (const void*) screenG, sizeof(float)*800*800*4, cudaMemcpyDeviceToHost);

	//FIBITMAP *img = FreeImage_ConvertFromRawBits((BYTE*)screenC, 800, 800, 800 * 3, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);
	//FreeImage_Save(FIF_PNG, img, "prueba.png", 0);

	FreeImage_DeInitialise();
}
