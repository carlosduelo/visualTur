#include "visualTur_thread.hpp"
#include "cuda_help.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

visualTur_thread::visualTur_thread(visualTurParams_thread_t initParams, Octree_device * p_octree_device)
{
	// Creating Camera
	camera = new Camera(initParams.startRay, initParams.endRay, initParams.numRayPx, initParams.H, initParams.W, initParams.distance, initParams.fov_H, initParams.fov_W);

	// Creating visible cubes array
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();

	// Create octree
	octreeLevel = initParams.octreeLevel;
	octree = new Octree_thread(p_octree_device, camera, octreeLevel);
	octree->resetState();

	// Cache creation
	cache = new lruCache(initParams.hdf5File, initParams.dataset_name, initParams.maxElementsCache, initParams.dimCubeCache, initParams.cubeInc, initParams.levelCubes, initParams.octreeLevel,initParams.maxElementsCache_CPU);
	cubeLevel = initParams.levelCubes;

	// Create rayCaster
	raycaster = new rayCaster(p_octree_device->getIsosurface(), make_float3(0.0f, 512.0f, 0.0f));
}

visualTur_thread::~visualTur_thread()
{
	delete 		camera;
	delete		cache;
	delete		octree;
	delete		raycaster;
	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
}

void visualTur_thread::resetVisibleCubes()
{
	std::cerr<<"Reset Cubes: "<< cudaGetErrorString(cudaMemset((void*)visibleCubesGPU, 0, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
}

pthread_t visualTur_thread::getID_thread()
{
	return id_thread;
}

void	visualTur_thread::camera_Move(float3 Direction)
{
	camera->Move(Direction);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_RotateX(float Angle)
{
	camera->RotateX(Angle);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_RotateY(float Angle)
{
	camera->RotateY(Angle);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_RotateZ(float Angle)
{
	camera->RotateZ(Angle);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_MoveForward(float Distance)
{
	camera->MoveForward(Distance);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_MoveUpward(float Distance)
{
	camera->MoveUpward(Distance);
	resetVisibleCubes();
	octree->resetState();
}

void	visualTur_thread::camera_StrafeRight(float Distance)
{
	camera->StrafeRight(Distance);
	resetVisibleCubes();
	octree->resetState();
}

void visualTur_thread::updateVisibleCubes(float * pixelBuffer)
{
	std::cout<<"Rendering new frame"<<std::endl;
	bool notEnd = true;
	int iterations = 0;

	while(notEnd)
	{
		octree->getBoxIntersected(visibleCubesGPU, visibleCubesCPU);

		cache->updateCache(visibleCubesCPU, camera->get_numRays(), octree->getnLevels());

		int numP = 0;
		for(int i=0; i<camera->get_numRays(); i++)
			if (visibleCubesCPU[i].state == PAINTED)
				numP++;

		if (numP == camera->get_numRays())
		{
			notEnd = false;
			break;
		}

		std::cerr<<"Coping visibleCubes to GPU: "<<cudaGetErrorString(cudaMemcpy((void*) visibleCubesGPU, (const void*) visibleCubesCPU, camera->get_numRays()*sizeof(visibleCube_t), cudaMemcpyHostToDevice))<<std::endl;

		raycaster->render(camera, octreeLevel, cubeLevel,octree->getnLevels(), visibleCubesGPU, cache->get_cubeDim(), make_int3(cache->get_cubeInc(),cache->get_cubeInc(),cache->get_cubeInc()), pixelBuffer); 

		iterations++;
	}
	std::cout<<"Iterations "<<iterations<<std::endl;
}
