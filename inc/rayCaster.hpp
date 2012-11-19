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
		// Light 
		float3			lightPosition;
		// Material parameters
	public:
		rayCaster(float isosurface, float3 lposition);

		~rayCaster();

		void render(Camera * camera, int level, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * buffer);
		void renderCube(Camera * camera, float * cube, int3 minBox, int3 cubeDim, int3 cubeInc, float * buffer);

};

#endif
