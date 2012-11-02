#include "visualTur.hpp"
#include <iostream>
#include <fstream>

visualTur::visualTur(visualTurParams_t initParams)
{
	// Creating Camera
	camera = new Camera(initParams.numRayPx, initParams.H, initParams.W, initParams.distance, initParams.fov_H, initParams.fov_W);

	// Creating visible cubes array
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU: "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();

	// Cache creation
	cache = new lruCache(initParams.hdf5File, initParams.dataset_name, initParams.maxElementsCache, initParams.dimCubeCache, initParams.cubeInc);

	octree = new Octree(initParams.octreeFile, camera);
}

visualTur::~visualTur()
{
	delete 		camera;
	delete		cache;
	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	delete		octree;
}

void visualTur::resetVisibleCubes()
{
	int max = camera->get_numRays();
	for(int i=0; i<max; i++)
	{
		visibleCubesCPU[i].id = 0;
		visibleCubesCPU[i].data = 0;
	}
	std::cerr<<"Coping visibleCubes CPU to GPU: "<< cudaGetErrorString(cudaMemcpy((void*)visibleCubesGPU, (const void*)visibleCubesCPU, camera->get_numRays()*sizeof(visibleCube_t), cudaMemcpyHostToDevice))<<std::endl;

}

void visualTur::changeScreen(int pW, int pH, float pfovW, float pfovH, float pDistance)
{
	camera->set_W(pW);
	camera->set_H(pH);
	camera->set_fovW(pfovW);
	camera->set_fovH(pfovH);
	camera->set_Distance(pDistance);

	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU: "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void visualTur::changeNumRays(int pnR)
{
	camera->set_RayPerPixel(pnR);

	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU: "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void visualTur::changeCacheParameters(int nE, int3 cDim, int cInc)
{
	cache->changeDimensionCube(nE, cDim, cInc);

	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU: "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void	visualTur::camera_Move(float3 Direction)
{
	camera->Move(Direction);
}
void	visualTur::camera_RotateX(float Angle)
{
	camera->RotateX(Angle);
}
void	visualTur::camera_RotateY(float Angle)
{
	camera->RotateY(Angle);
}
void	visualTur::camera_RotateZ(float Angle)
{
	camera->RotateZ(Angle);
}
void	visualTur::camera_MoveForward(float Distance)
{
	camera->MoveForward(Distance);
}
void	visualTur::camera_MoveUpward(float Distance)
{
	camera->MoveUpward(Distance);
}
void	visualTur::camera_StrafeRight(float Distance)
{
	camera->StrafeRight(Distance);
}

void visualTur::updateVisibleCubes(int level)
{
	octree->getBoxIntersected(level, visibleCubesGPU, visibleCubesCPU);

	#if 1
	int hits = 0;
	for(int i=0; i<camera->get_numRays(); i++)
		if (visibleCubesCPU[i].id != 0)
			hits++;
	std::cout<<"Hits "<<hits<<std::endl;
	#endif

	cache->updateCache(visibleCubesCPU, camera->get_numRays(), octree->getnLevels());

}
