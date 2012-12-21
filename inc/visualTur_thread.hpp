/*
 * visualTur_thread 
 *
 */

#ifndef _VISUALTUR_THREAD_H_
#define _VISUALTUR_THREAD_H_
#include "Octree.hpp"
#include "lruCache.hpp"
#include "rayCaster.hpp"

typedef struct
{
	// Camera settings
	int 	W;
	int 	H;
	float 	distance;
	float 	fov_H;
	float	fov_W;
	int	numRayPx;
	// Cube Cache settings
	int	maxElementsCache;
	int	maxElementsCache_CPU;
	int3	dimCubeCache;
	int	cubeInc;
	int	levelCubes;
	// hdf5 settings
	char * hdf5File;
	char * dataset_name;
	// octree settings
	char *  octreeFile;
	int 	octreeLevel;
} visualTurParams_thread_t;

class visualTur_thread
{
	private:
		// Multithreading stuff
		pthread_t 	id_thread;
		pthread_attr_t 	attr_thread;
		cudaStream_t 	stream;

		Camera * 	camera;

		visibleCube_t *	visibleCubesCPU;
		visibleCube_t *	visibleCubesGPU;

		lruCache *	cache;
		int		cubeLevel;

		Octree *	octree;
		float 		iso;
		int		octreeLevel;

		rayCaster *	raycaster;

		void resetVisibleCubes();
	public:
		visualTur(visualTurParams_thread_t initParams);

		~visualTur();

		// Multithreading methods
		pthread_t getID_thread();

		// Change parameters
		void changeCacheParameters(int nE, int3 cDim, int cInc);

		// Move camera
		void	camera_Move(float3 Direction);
		void	camera_RotateX(float Angle);
		void	camera_RotateY(float Angle);
		void	camera_RotateZ(float Angle);
		void	camera_MoveForward(float Distance);
		void	camera_MoveUpward(float Distance);
		void	camera_StrafeRight(float Distance);	

		void updateVisibleCubes(float * pixelBuffer);
};
#endif
