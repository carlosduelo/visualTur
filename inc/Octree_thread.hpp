/*
 * Octree CUDA thread 
 *
 */

#ifndef _OCTREE_THREAD_H_
#define _OCTREE_THREAD_H_

#include "Octree_device.hpp"
#include "Camera.hpp"

class Octree_thread
{
	private:
		Camera * camera;

		int nLevels;
		int maxLevel;

		index_node_t ** octree;
		index_node_t * 	memoryGPU;
		int	*	sizes;

		// Octree State
		int 	*	GstackActual;
		index_node_t * 	GstackIndex;
		int	*	GstackLevel;
	public:
		Octree_thread(Octree_device * p_octree, Camera * p_camera, int p_maxLevel);

		~Octree_thread();

		int getnLevels();

		void resetState(cudaStream_t stream);

		/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
		bool getBoxIntersected(visibleCube_t * visibleGPU, visibleCube_t * visibleCPU, cudaStream_t stream);
};
#endif
