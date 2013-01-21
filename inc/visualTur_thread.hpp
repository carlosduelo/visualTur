/*
 * visualTur_thread 
 *
 */

#ifndef _VISUALTUR_THREAD_H_
#define _VISUALTUR_THREAD_H_
#include "Octree_thread.hpp"
#include "lruCache_device.hpp"
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
	int	startRay;
	int	endRay;

	// Cube Cache settings
	int	levelCubes;

	// hdf5 settings
	char * hdf5File;
	char * dataset_name;

	// Octree
	int	octreeLevel;

	int	device;
} visualTurParams_thread_t;

class visualTur_thread;

struct param_t
{
	visualTur_thread * 	object;
	float			number;
	float3			coord;
	float		*	buffer;
};

class visualTur_thread
{
	public:// For multithreading stuff...
	//private:
		// Multithreading stuff
		pthread_t 	id_thread;
		pthread_attr_t 	attr_thread;
		cudaStream_t 	stream;
		int		deviceID;
		param_t		paramT;

		Camera * 	camera;

		visibleCube_t *	visibleCubesCPU;
		visibleCube_t *	visibleCubesGPU;

		lruCache_device *cache;
		int		 cubeLevel;

		Octree_thread *	octree;
		int		octreeLevel;

		rayCaster *	raycaster;
		float	  *	pixelBuffer;

		void resetVisibleCubes();
	//public:
		visualTur_thread(visualTurParams_thread_t initParams, Octree_device * p_octree_device, lruCache_device * p_cache, float * p_pixelBuffer);

		~visualTur_thread();

		// Multithreading methods
		pthread_t getID_thread();

		// Move camera
		void	camera_Move(float3 Direction);
		void	camera_RotateX(float Angle);
		void	camera_RotateY(float Angle);
		void	camera_RotateZ(float Angle);
		void	camera_MoveForward(float Distance);
		void	camera_MoveUpward(float Distance);
		void	camera_StrafeRight(float Distance);	

		void updateVisibleCubes();
};
#endif
