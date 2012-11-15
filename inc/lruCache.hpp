/*
 * Cube cache
 *
 */

#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_
#include "config.hpp"
#include "FileManager.hpp"
#include "cutil_math.h"
//#include <map>
#include <boost/unordered_map.hpp>
#include <sys/time.h>


class NodeLinkedList
{
	public:
		NodeLinkedList * after;
		NodeLinkedList * before;
		unsigned int	 element;
		index_node_t 	 cubeID;
};

class LinkedList
{
	private:
		NodeLinkedList * list;
		NodeLinkedList * last;
		NodeLinkedList * memoryList;
	public:
		LinkedList(int size);
		~LinkedList();

		/* pop_front and push_last */
		NodeLinkedList * getFromFirstPosition(index_node_t newIDcube, index_node_t * removedIDcube);

		NodeLinkedList * moveToLastPosition(NodeLinkedList * node);	
};


class lruCache
{

	private:
		int 					 numElements;
		int 					 numElementsCPU;
		int3					 cubeDim;
		int3					 cubeInc;
		int3					 realcubeDim;
		int					 offsetCube;

		//std::map<index_node_t, NodeLinkedList *> indexStoredCPU;
		boost::unordered_map<index_node_t, NodeLinkedList *> indexStoredCPU;
		LinkedList	*			 queuePositionsCPU;
		//std::map<index_node_t, NodeLinkedList *> indexStored;
		boost::unordered_map<index_node_t, NodeLinkedList *> indexStored;
		LinkedList	*			 queuePositions;

		float		*			 cacheData;
		float		*			 cacheDataCPU;
		FileManager	*			 fileManager;

		// Measure propouse
		int 					access;
		int					hitsCPU;
		int					hitsGPU;
		int					missCPU;
		int					missGPU;
		double					timingAccess;
		double					timinghitsCPU;
		double					timingmissCPU;
		double					timinghitsGPU;
		double					timingmissGPU;

		#if 0
		bool insertElement(index_node_t element, unsigned int * position);
		bool insertElementCPU(index_node_t element, unsigned int * position);
		#endif
		void updateCube(visibleCube_t * cube, int nLevels, int * nEinsertedCPU, int * nEinsertedGPU);
	public:
		lruCache(char * file_name, char * dataset_name, int maxElements, int3 cDim, int cI, int maxElementsCPU);
		
		~lruCache();

		int	 get_numElements();
		int3	 get_cubeDim();
		int	 get_cubeInc();

		void changeDimensionCube(int maxElements, int3 cDim, int cI);

		void updateCache(visibleCube_t * visibleCubes, int num, int nLevels);

		void printStatistics();
};
#endif
