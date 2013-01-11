/*
 * Octree 
 *
 */

#ifndef _OCTREE_M_H_
#define _OCTREE_M_H_

#include "config.hpp"
#include <vector>

class OctreeLevel
{
	private:
		std::vector<index_node_t> * 	elements;
		bool 				status;
		index_node_t			nextElement;
		inline bool 	checkRange(index_node_t index, int min, int max);
		bool 		binary_search(index_node_t index, int min, int max);
		int 		binary_search_closer(index_node_t index, int min, int max);
		inline bool 	searchSecuential(index_node_t index, int min, int max, bool up);
	public:
		OctreeLevel();

		~OctreeLevel();

		int getSize();

		void printLevel();

		std::vector<index_node_t>::iterator getBegin();

		std::vector<index_node_t>::iterator getEnd();

		void completeLevel();

		bool insert(index_node_t index);

		void fill(index_node_t index);

		bool search(index_node_t index);

		/* Dado un index de nodo, devuelve los hijos que están en el nivel devolviendo el numero de hijos*/
		int searchChildren(index_node_t father, index_node_t * children);
};


class OctreeM
{
	private:
		float isosurface;
		int realDim[3];
		int dimension;
		int nLevels;

		/* Crea el nivel dado, si no es el ultimo lo crea a partir del nivel inferior */
		void 		createTree(const char * file_name, int level);
		bool 		recursiveTraceRay(float origin[3], float dir[3],  int level, int finalLevel, index_node_t index, index_node_t * finalIndex, float * finalTnear);

	public:
		OctreeLevel * octree;
		/* Crea desde cero el octree */
		OctreeM(const char * file_name, float iso);

		~OctreeM();

		float getIsosurface();

		/* Escribe en un fichero el octree */
		void writeToFile(const char * file_name);

		/* Devuelve verdadero si en la box en el nivel level a la que pertenece la coordenada (x,y,z) existe superficie */
		bool    	getValidinLevel(int x, int y, int z, int level);

		bool 		getValidNode(index_node_t index);

		/* Devuelve el numero de niveles que tiene el octree suponiendo que los nodos hojas son voxels */
		int 		getnLevels();

		/* Devuelve el numero de niveles que tiene el octree de verdad, dado que sus nodos hojas son boxs de 32x32x32 */
		int 		getrealLevels();

		/* Devuelve la dimension real del volumen que contiene el octree */
		int * 		getRealDimension();

		/* Escribe por la la salida estandar todos las box que contiene un nivel */
		void 		printLevel(int level);

};

#endif /*_OCTREE_M_H_*/
