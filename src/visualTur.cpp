#include "visualTur.hpp"

visualTur::visualTur(visualTurParams_t initParams)
{
	// Creating Camera
	camera = new Camera(initParams.numRayPx, initParams.H, initParams.W, initParams.distance, initParams.fov_H, initParams.fov_W);

	// Creating visible cubes array
	visibleCubes = new visibleCube_t[camera->get_numRays()];
	resetVisibleCubes();

	// Cache creation
	cache = new lruCache(initParams.hdf5File, initParams.dataset_name, initParams.maxElementsCache, initParams.dimCubeCache, initParams.cubeInc);
}

visualTur::~visualTur()
{
	delete 		camera;
	delete		cache;
	delete[]	visibleCubes;
}

void visualTur::resetVisibleCubes()
{
	int max = camera->get_numRays();
	for(int i=0; i<max; i++)
	{
		visibleCubes[i].id = 0;
		visibleCubes[i].data = 0;
	}
}

void visualTur::changeParams(visualTurParams_t params)
{
	bool changeVisibleCubes = false;
	// Check and update if parameters change
	if (camera->get_W() != params.W)
	{
		camera->set_W(params.W);
		changeVisibleCubes = true;
	}
	if (camera->get_H() != params.H)
	{
		camera->set_H(params.H);
		changeVisibleCubes = true;
	}
	if (camera->get_fovW() != params.fov_W)
	{
		camera->set_fovW(params.fov_W);
		changeVisibleCubes = true;
	}
	if (camera->get_fovH() != params.fov_H)
	{
		camera->set_fovH(params.fov_H);
		changeVisibleCubes = true;
	}
	if (camera->get_Distance() != params.distance)
	{
		camera->set_Distance(params.distance);
		changeVisibleCubes = true;
	}
	if (camera->get_numRayPixel() != params.numRayPx)
	{
		camera->set_RayPerPixel(params.numRayPx);
		changeVisibleCubes = true;
	}

	if (cache->get_numElements() != params.maxElementsCache)
	{
		cache->changeDimensionCube(params.maxElementsCache, cache->get_cubeDim(), cache->get_cubeInc());
		changeVisibleCubes = true;
	}
	if (!(cache->get_cubeDim() == params.dimCubeCache))
	{
		cache->changeDimensionCube(cache->get_numElements(), params.dimCubeCache, cache->get_cubeInc());
		changeVisibleCubes = true;
	}
	if (cache->get_cubeInc() != params.cubeInc)
	{
		cache->changeDimensionCube(cache->get_numElements(), cache->get_cubeDim(), params.cubeInc);
		changeVisibleCubes = true;
	}

	if (changeVisibleCubes)
	{
		delete[]	visibleCubes;
		visibleCubes = new visibleCube_t[camera->get_numRays()];
		resetVisibleCubes();
	}
}
