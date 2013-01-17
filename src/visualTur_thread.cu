#include "visualTur_thread.hpp"
#include "cuda_help.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <pthread.h>

/* Struct parameters */


visualTur_thread::visualTur_thread(visualTurParams_thread_t initParams, Octree_device * p_octree_device, float * p_pixelBuffer)
{

	// Create multithreading stuff
	pthread_attr_init(&attr_thread);
	pthread_attr_setdetachstate(&attr_thread, PTHREAD_CREATE_JOINABLE);
	std::cerr<<"Createing cudaStream: "<< cudaGetErrorString(cudaStreamCreate(&stream))<<std::endl;
	deviceID = initParams.device;

	// Creating Camera
	camera = new Camera(initParams.startRay, initParams.endRay, initParams.numRayPx, initParams.H, initParams.W, initParams.distance, initParams.fov_H, initParams.fov_W, stream);
	std::cout<<"Camera "<<initParams.startRay <<" "<<initParams.endRay<<std::endl;

	// Creating visible cubes array
//	visibleCubesCPU = new visibleCube_t[camera->get_numRays()];
	std::cerr<<"Allocating memory visibleCubesCPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaHostAlloc((void**)&visibleCubesCPU, camera->get_numRays()*sizeof(visibleCube_t), cudaHostAllocDefault))<<std::endl;
	std::cerr<<"Allocating memory visibleCubesGPU "<<camera->get_numRays()*sizeof(visibleCube_t)/1024/1024 <<" MB : "<< cudaGetErrorString(cudaMalloc((void**)&visibleCubesGPU, camera->get_numRays()*sizeof(visibleCube_t)))<<std::endl;
	resetVisibleCubes();

	// Create octree
	octreeLevel = initParams.octreeLevel;
	octree = new Octree_thread(p_octree_device, camera, octreeLevel);
	octree->resetState(stream);

	// Cache creation
	cache = new lruCache(initParams.hdf5File, initParams.dataset_name, initParams.maxElementsCache, initParams.dimCubeCache, initParams.cubeInc, initParams.levelCubes, initParams.octreeLevel,initParams.maxElementsCache_CPU);
	cubeLevel = initParams.levelCubes;

	// Create rayCaster
	pixelBuffer = p_pixelBuffer;
	raycaster = new rayCaster(p_octree_device->getIsosurface(), make_float3(0.0f, 512.0f, 0.0f), pixelBuffer);

	paramT.object = this;
}

visualTur_thread::~visualTur_thread()
{
	delete 		camera;
	delete		cache;
	delete		octree;
	delete		raycaster;
	//delete[]	visibleCubesCPU;
	cudaFreeHost(visibleCubesCPU);
	cudaFree(visibleCubesGPU);
	pthread_attr_destroy(&attr_thread);
        cudaStreamDestroy(stream);
}

void visualTur_thread::resetVisibleCubes()
{
	std::cerr<<"Reset Cubes: "<< cudaGetErrorString(cudaMemsetAsync((void*)visibleCubesGPU, 0, camera->get_numRays()*sizeof(visibleCube_t), stream))<<std::endl;
}

pthread_t visualTur_thread::getID_thread()
{
	return id_thread;
}


/* Function executed by the thread */
void *_thread_camera_Move(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->Move((float3)p->coord,o->stream);//p->coord, o->stream);
	//o->camera->Move((float3)p->coord,o->stream);//p->coord, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}


void	visualTur_thread::camera_Move(float3 Direction)
{
	paramT.coord	= Direction;
	
	int rc = pthread_create(&id_thread, NULL, _thread_camera_Move, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
	
}

/* Function executed by the thread */
void *_thread_camera_RotateX(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->RotateX((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_RotateX(float Angle)
{
	paramT.number	= Angle;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_RotateX, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_camera_RotateY(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->RotateY((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_RotateY(float Angle)
{
	paramT.number	= Angle;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_RotateY, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_camera_RotateZ(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->RotateZ((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_RotateZ(float Angle)
{
	paramT.number	= Angle;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_RotateZ, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_camera_MoveForward(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->MoveForward((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_MoveForward(float Distance)
{
	paramT.number	= Distance;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_MoveForward, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_camera_MoveUpward(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->MoveUpward((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_MoveUpward(float Distance)
{
	paramT.number	= Distance;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_MoveUpward, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_camera_StrafeRight(void * param)
{
	param_t * 		p = (param_t*) param;
	visualTur_thread * 	o = (visualTur_thread*) p->object;
	
	cudaSetDevice(o->deviceID);
	o->camera->StrafeRight((float)p->number, o->stream);
	o->resetVisibleCubes();
	o->octree->resetState(o->stream);
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void	visualTur_thread::camera_StrafeRight(float Distance)
{
	paramT.number	= Distance;

	int rc = pthread_create(&id_thread, NULL, _thread_camera_StrafeRight, (void *)&paramT);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}

/* Function executed by the thread */
void *_thread_updateVisibleCubes(void * param)
{
	visualTur_thread * 	o = (visualTur_thread*) param;

	cudaSetDevice(o->deviceID);
	bool notEnd = true;
	int iterations = 0;

	while(notEnd)
	{
		o->octree->getBoxIntersected(o->visibleCubesGPU, o->visibleCubesCPU, o->stream);
		cudaStreamSynchronize(o->stream); 
		
		o->cache->updateCache(o->visibleCubesCPU, o->camera->get_numRays(), o->octree->getnLevels(), o->stream);
		int numP = 0;
		for(int i=0; i<o->camera->get_numRays(); i++)
			if (o->visibleCubesCPU[i].state == PAINTED)
				numP++;

		if (numP == o->camera->get_numRays())
		{
			notEnd = false;
			break;
		}

		std::cerr<<"Coping visibleCubes to GPU: "<<cudaGetErrorString(cudaMemcpyAsync((void*) o->visibleCubesGPU, (const void*) o->visibleCubesCPU, o->camera->get_numRays()*sizeof(visibleCube_t), cudaMemcpyHostToDevice, o->stream))<<std::endl;
		cudaStreamSynchronize(o->stream); 

		o->raycaster->render(o->camera, o->octreeLevel, o->cubeLevel, o->octree->getnLevels(), o->visibleCubesGPU, o->cache->get_cubeDim(), make_int3(o->cache->get_cubeInc(),o->cache->get_cubeInc(), o->cache->get_cubeInc()), o->stream); 

		iterations++;
	}

	std::cout<<"Iterations "<<iterations<<std::endl;
	//cudaStreamSynchronize(o->stream); // Posible synchronize point
	return NULL;
}
void visualTur_thread::updateVisibleCubes()
{
	int rc = pthread_create(&id_thread, NULL, _thread_updateVisibleCubes, (void *)this);
	if (rc)
	{
		std::cout << "Error:unable to create thread," << rc << std::endl;
		exit(-1);
	}
}
