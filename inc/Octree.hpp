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
		Camera * camera;
		float isosurface;
		int3 realDim;
		int dimension;
		int nLevels;
		int maxLevel;
		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;
		// Octree State
		#if _OG_
		index_node_t *	GstackIndex; // cojo el puntero que apunta a memoria global
		int 	*	GstackActual; // Lo leo de memoria global
		int 	*	GcurrentLevel; // Lo leo de memoria global
		int	*	Gnumbro; // cojo el puntero que apunta a memoria global
		#else
		int 	*	GstackActual;
		index_node_t * 	GstackIndex;
		int	*	GstackLevel;
		#endif
	public:
		/* Lee el Octree de un fichero */
		Octree(const char * file_name, Camera * p_camera, int p_maxLevel);

		~Octree();

		int getnLevels();

		float getIsosurface();

		void resetState();

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		bool getBoxIntersected(int finalLevel, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU);
};
#endif
