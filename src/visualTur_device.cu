#include "visualTur_device.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

visualTur_device::visualTur_device(visualTurParams_device_t initParams)
{
	// Create octree
	octreeLevel = initParams.octreeLevel;
	octree = new Octree(initParams.octreeFile, camera, octreeLevel);
	octree->resetState();
}
