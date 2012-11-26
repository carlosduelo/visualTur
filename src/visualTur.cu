#include "visualTur.hpp"
#include "cuda_help.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>


visualTur::visualTur(visualTurParams_t initParams)
{
	// Creating Camera
	camera = new Camera(initParams.numRayPx, initParams.H, initParams.W, initParams.distance, initParams.fov_H, initParams.fov_W);

	// Creating visible cubes array
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();

	octreeLevel = initParams.octreeLevel;
	// Create octree
	octree = new Octree(initParams.octreeFile, camera, octreeLevel);

	// Cache creation
	cache = new lruCache(initParams.hdf5File, initParams.dataset_name, initParams.maxElementsCache, initParams.dimCubeCache, initParams.cubeInc, initParams.levelCubes, initParams.octreeLevel,initParams.maxElementsCache_CPU);
	cubeLevel = initParams.levelCubes;

	// Create rayCaster
	raycaster = new rayCaster(octree->getIsosurface(), make_float3(0.0f, 512.0f, 0.0f));

	timingO = 0.0;
	timingC = 0.0;
	timingR = 0.0;
}

visualTur::~visualTur()
{
	delete 		camera;
	delete		cache;
	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	delete		octree;
	delete		raycaster;
}


__global__ void resetCubes(visibleCube_t * cubes, int max)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;
	if (tid<max)
	{
		cubes[tid].id = 0;
		cubes[tid].data = 0;
		cubes[tid].state = NOCUBE;
	}
}

void visualTur::resetVisibleCubes()
{
	#if 0
	int max = camera->get_numRays();
	for(int i=0; i<max; i++)
	{
		visibleCubesCPU[i].id = 0;
		visibleCubesCPU[i].data = 0;
		visibleCubesCPU[i].state = NOCUBE;
	}
	std::cerr<<"Coping visibleCubes CPU to GPU: "<< cudaGetErrorString(cudaMemcpy((void*)visibleCubesGPU, (const void*)visibleCubesCPU, camera->get_numRays()*sizeof(visibleCube_t), cudaMemcpyHostToDevice))<<std::endl;
	#else
	dim3 threads = getThreads(camera->get_numRays());
	dim3 blocks = getBlocks(camera->get_numRays());
	resetCubes<<<blocks, threads>>>(visibleCubesGPU, camera->get_numRays());
	#endif
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
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void visualTur::changeNumRays(int pnR)
{
	camera->set_RayPerPixel(pnR);

	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);
	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void visualTur::changeCacheParameters(int nE, int3 cDim, int cInc)
{
	cache->changeDimensionCube(nE, cDim, cInc);

	delete[]	visibleCubesCPU;
	cudaFree(visibleCubesGPU);

	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();
}

void	visualTur::camera_Move(float3 Direction)
{
	camera->Move(Direction);
	resetVisibleCubes();
}
void	visualTur::camera_RotateX(float Angle)
{
	camera->RotateX(Angle);
	resetVisibleCubes();
}
void	visualTur::camera_RotateY(float Angle)
{
	camera->RotateY(Angle);
	resetVisibleCubes();
}
void	visualTur::camera_RotateZ(float Angle)
{
	camera->RotateZ(Angle);
	resetVisibleCubes();
}
void	visualTur::camera_MoveForward(float Distance)
{
	camera->MoveForward(Distance);
	resetVisibleCubes();
}
void	visualTur::camera_MoveUpward(float Distance)
{
	camera->MoveUpward(Distance);
	resetVisibleCubes();
}
void	visualTur::camera_StrafeRight(float Distance)
{
	camera->StrafeRight(Distance);
	resetVisibleCubes();
}

void visualTur::updateVisibleCubes(float * pixelBuffer)
{
	std::cout<<"Rendering new frame"<<std::endl;
	bool notEnd = true;
	int iterations = 0;
	double deltaO = 0.0;
	double deltaC = 0.0;
	double deltaR = 0.0;
	while(notEnd)
	{
		struct timeval st, end;
		gettimeofday(&st, NULL);
		octree->getBoxIntersected(octreeLevel, visibleCubesGPU, visibleCubesCPU);
		gettimeofday(&end, NULL);
		deltaO += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
		timingO += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;


		gettimeofday(&st, NULL);
		cache->updateCache(visibleCubesCPU, camera->get_numRays(), octree->getnLevels());
		gettimeofday(&end, NULL);
		deltaC += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
		timingC += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;

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

		gettimeofday(&st, NULL);
		raycaster->render(camera, octreeLevel, cubeLevel,octree->getnLevels(), visibleCubesGPU, cache->get_cubeDim(), make_int3(cache->get_cubeInc(),cache->get_cubeInc(),cache->get_cubeInc()), pixelBuffer); 
		gettimeofday(&end, NULL);
		deltaR += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
		timingR += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;


		#if 0
		std::cerr<<"Coping visibleCubes to CPU: "<<cudaGetErrorString(cudaMemcpy((void*) visibleCubesCPU, (const void*) visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t), cudaMemcpyDeviceToHost))<<std::endl;
		int noHitsR = 0;
		for(int i=0; i<camera->get_numRays(); i++)
			if (visibleCubesCPU[i].state == PAINTED)
				noHitsR++;
		std::cout<<"Hits in ray casting "<<noHitsR<<std::endl;
		#endif
		iterations++;
	}
	cache->printStatistics();
	std::cout << "Time elapsed in octree: " << deltaO << " sec"<< std::endl;
	std::cout << "Time elapsed in cache: " << deltaC << " sec"<< std::endl;
	std::cout << "Time elapsed in raycasting: " << deltaR << " sec"<< std::endl;
	std::cout << "Time elapsed in total frame: " << deltaO+deltaC+deltaR << " sec"<< std::endl;
	std::cout << "Time elapsed accumulate total "<<timingO+timingC+timingR<<" Octree "<<timingO<<" Cache "<<timingC<<" RayCasting "<<timingR<<std::endl;
	std::cout<<"Iterations "<<iterations<<std::endl;
}
