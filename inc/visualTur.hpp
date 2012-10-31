/*
 * Camera
 *
 */

#ifndef _VISUALTUR_H_
#define _VISUALTUR_H_
#include "Camera.hpp"
#include "lruCache.hpp"

typedef struct
{
	// Camera stuff
	int 	W;
	int 	H;
	float 	distance;
	float 	fov_H;
	float	fov_W;
	int	numRayPx;
	// Cube Cache stuff
	int	maxElementsCache;
	int3	dimCubeCache;
	int	cubeInc;
	// hdf5 files
	char * hdf5File;
	char * dataset_name;
	// octree file
	char * octreeFile;

} visualTurParams_t;

class visualTur
{
	private:
		Camera * 	camera;

		visibleCube_t *	visibleCubes;

		lruCache *	cache;

		void resetVisibleCubes();
	public:
		visualTur(visualTurParams_t initParams);

		~visualTur();

		void changeParams(visualTurParams_t params);

};
#endif
