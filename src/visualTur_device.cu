#include "visualTur_device.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

visualTur_device::visualTur_device(visualTurParams_device_t initParams, float * p_pixelBuffer)
{
	// Create octree
	octreeLevel 	= initParams.octreeLevel;
	octree 		= new Octree_device(initParams.octreeFile, octreeLevel);

	numThreads 	= initParams.numThreads;
	deviceThreads 	= new visualTur_thread*[numThreads];
	deviceID	= initParams.deviceID;

	offsetPixelBuffer 	= new int[numThreads];
	pixelBuffer 		= p_pixelBuffer;

	visualTurParams_thread_t initParams_thread;
	initParams_thread.W 			= initParams.W;
	initParams_thread.H 			= initParams.H;
	initParams_thread.distance		= initParams.distance;
	initParams_thread.fov_H 		= initParams.fov_H;
	initParams_thread.fov_W 		= initParams.fov_W;
	initParams_thread.numRayPx 		= initParams.numRayPx;

	// Cube Cache settings
	initParams_thread.maxElementsCache 	= initParams.maxElementsCache / numThreads;
	initParams_thread.maxElementsCache_CPU 	= initParams.maxElementsCache_CPU / numThreads;
	initParams_thread.dimCubeCache 		= initParams.dimCubeCache;
	initParams_thread.cubeInc 		= initParams.cubeInc;
	initParams_thread.levelCubes 		= initParams.levelCubes;

	// hdf5 settings
	initParams_thread.hdf5File 		= initParams.hdf5File;
	initParams_thread.dataset_name 		= initParams.dataset_name;

	// Octree
	initParams_thread.octreeLevel 		= initParams.octreeLevel;

	initParams_thread.device		= deviceID;

	int totalRays 	= initParams.endRay - initParams.startRay;
	int numRays 	= totalRays / numThreads;
	int modRays	= totalRays % numThreads;

	initParams_thread.startRay      = initParams.startRay;
        initParams_thread.endRay        = initParams.startRay + numRays + modRays;
	offsetPixelBuffer[0]		= 0;
        deviceThreads[0] 		= new visualTur_thread(initParams_thread, octree, pixelBuffer);

	for(int i=1; i<numThreads; i++)
	{
		if (i==1 && modRays != 0)
			offsetPixelBuffer[i]		= offsetPixelBuffer[i-1] + numRays + modRays;
		else
			offsetPixelBuffer[i]		= offsetPixelBuffer[i-1] + numRays;
		initParams_thread.startRay 	= initParams_thread.endRay;
		initParams_thread.endRay 	= initParams_thread.startRay + numRays;
		deviceThreads[i] = new visualTur_thread(initParams_thread, octree, pixelBuffer + offsetPixelBuffer[i]);
	}
}

visualTur_device::~visualTur_device()
{
	for(int i=0; i<numThreads; i++)
		delete deviceThreads[i];

	delete octree;
	delete[] offsetPixelBuffer;
	delete[] deviceThreads;
}
		
void	visualTur_device::camera_Move(float3 Direction)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_Move(Direction);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_RotateX(float Angle)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_RotateX(Angle);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_RotateY(float Angle)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_RotateY(Angle);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_RotateZ(float Angle)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_RotateZ(Angle);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_MoveForward(float Distance)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_MoveForward(Distance);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_MoveUpward(float Distance)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_MoveUpward(Distance);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

void	visualTur_device::camera_StrafeRight(float Distance)
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->camera_StrafeRight(Distance);

	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}


void visualTur_device::updateVisibleCubes()
{
	
	for(int i=0; i<numThreads; i++)
		deviceThreads[i]->updateVisibleCubes();
	
	void * status;
	for(int i=0; i<numThreads; i++)
	{
		int rc = pthread_join(deviceThreads[i]->getID_thread(), &status);
                if (rc)
                {
                        std::cerr << "Error:unable to join," << rc << std::endl;
                        exit(-1);
                }
	}
}

