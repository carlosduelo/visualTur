/*
 * visualTur_device 
 *
 */

#ifndef _VISUALTUR_DEVICE_H_
#define _VISUALTUR_DEVICE_H_
#include "visualTur_thread.hpp"

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

	// Multithreading stuff
	int	numThreads;
	int	deviceID;
} visualTurParams_device_t;

class visualTur_device
{
	private:
		// Multithreading stuff
		int			deviceID;
		int			numThreads;
		visualTur_thread **	deviceThreads;
		int	*		offsetPixelBuffer;

		// Octree, shared for all threads
		Octree_device *	octree;
		float 		iso;
		int		octreeLevel;

		float * 	pixelBuffer;

	public:
		visualTur_device(visualTurParams_device_t initParams, float * p_pixelBuffer);

		~visualTur_device();

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
