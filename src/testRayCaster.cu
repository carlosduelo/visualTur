#include "config.hpp"
#include "FileManager.hpp"
#include "rayCaster.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include "FreeImage.h"
#include <iostream>
#include <fstream>


int main(int argc, char ** argv)
{
	Camera	* camera = 0;
	rayCaster * RayCaster = 0;
	FileManager * fileManager = 0;
	float iso = 1.5;

	// Camera Settings
	int W = 1920;
	int H = 1080;
	camera = new Camera(1, H, W, 50.0f, 30.0f, 35.0f);
	camera->Move(make_float3(64.0f, 64.0f, 250.0f));

	// Data Settings
	int3 coord = make_int3(256,0,256);
	int3 cubeDim = make_int3(128, 128, 128);
	int cubeINC = 2;
	int3 cubeInc = make_int3(cubeINC, cubeINC, cubeINC);
	int3 realcubeDim = cubeDim + 2 * cubeInc;
	float * dataCPU = new float[realcubeDim.x * realcubeDim.y * realcubeDim.z];
	float * dataGPU; 
	std::cerr<<"Allocating memory data buffer: "<<realcubeDim.x * realcubeDim.y * realcubeDim.z*sizeof(float)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc((void**)&dataGPU, realcubeDim.x * realcubeDim.y * realcubeDim.z*sizeof(float)))<<std::endl;
	int3 minBox = coord - cubeInc;
	int3 maxBox = minBox + realcubeDim;
	fileManager = new FileManager(argv[1], argv[2]);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, dataCPU);
	std::cerr<<"Copying data to GPU: "<<cudaGetErrorString(cudaMemcpy((void*) dataGPU, (const void*) dataCPU, realcubeDim.x * realcubeDim.y * realcubeDim.z*sizeof(float), cudaMemcpyHostToDevice))<<std::endl;
	

	// RayCaster Settings
	float * screenCPU = new float[4*camera->get_numRays()];
	float * screenGPU;
	std::cerr<<"Allocating memory screen buffer: "<<4*camera->get_numRays()*sizeof(float)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc((void**)&screenGPU, 4*camera->get_numRays()*sizeof(float)))<<std::endl;
	RayCaster = new rayCaster(iso, make_float3(0.0f, 512.0f, 0.0f));
	RayCaster->renderCube(camera, dataGPU,  make_int3(0,0,0), cubeDim, cubeInc, screenGPU);

	// Creating Image
	FIBITMAP * bitmap = FreeImage_Allocate(H,W,24);
	RGBQUAD color;
	std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenCPU, (const void*) screenGPU, sizeof(float)*camera->get_numRays()*4, cudaMemcpyDeviceToHost))<<std::endl;
	for(int i=0; i<H; i++)
		for(int j=0; j<W; j++)
		{
			int id = i*W + j;
			color.rgbRed = screenCPU[id*4]*255;
			color.rgbGreen = screenCPU[id*4+1]*255;
			color.rgbBlue = screenCPU[id*4+2]*255;
			FreeImage_SetPixelColor(bitmap, i, j, &color);
		} 
	FreeImage_Save(FIF_PNG, bitmap, "prueba.png", 0);

	delete[] dataCPU;
	delete[] screenCPU;
	cudaFree(dataGPU);
	cudaFree(screenGPU);
	delete camera;
	delete fileManager;
	delete RayCaster;
}
