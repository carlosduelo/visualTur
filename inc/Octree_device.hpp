/*
 * Octree CUDA 
 *
 */

#ifndef _OCTREE_DEVICE_H_
#define _OCTREE_DEVICE_H_

#include "config.hpp"

class Octree_device
{
	private:
		float isosurface;
		int3 realDim;
		int dimension;
		int nLevels;
		int maxLevel;

		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;

	public:
		/* Lee el Octree de un fichero */
		Octree_device(const char * file_name, int p_maxLevel);

		~Octree_device();

		int getnLevels();

		float getIsosurface();

		index_node_t ** getOctree();

		int * getSizes();

};
#endif

