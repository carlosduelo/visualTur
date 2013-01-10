/*
 * rayCaster 
 *
 */
#ifndef _RAY_CASTER_H_
#define _RAY_CASTER_H_

#include "config.hpp"
#include "Octree.hpp"
#include "Camera.hpp"

class rayCaster
{
	private:
		float 			iso;
		float	*		pixelBuffer;
		// Light 
		float3			lightPosition;
		// Material parameters

		// rayCasing Parameters
		float step;
	public:
		rayCaster(float isosurface, float3 lposition, float * p_pixelBuffer);

		~rayCaster();

		void render(Camera * camera, int levelO, int levelC, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, cudaStream_t stream);
};

#endif
