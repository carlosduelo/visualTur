/*
 * Octree CUDA 
 *
 */

#ifndef _OCTREE_H_
#define _OCTREE_H_

#include "Camera.hpp"
#include "config.hpp"

class Octree
{
	private:
		float isosurface;
		int3 realDim;
		int dimension;
		int nLevels;
		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;
	public:
		index_node_t * 	indexGPU;
		/* Lee el Octree de un fichero */
		Octree(const char * file_name, Camera * camera);

		~Octree();

		int getnLevels();

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		bool getFirtsBoxIntersected(Camera * camera, int finalLevel, index_node_t * indexs);
};
#endif
