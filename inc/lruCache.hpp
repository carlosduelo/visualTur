/*
 * Cube cache
 *
 */

#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_
#include "config.hpp"
#include "FileManager.hpp"
#include <map>


class NodeLinkedList
{
	public:
		NodeLinkedList * after;
		NodeLinkedList * before;
		unsigned int	 element;
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
		NodeLinkedList * getFromFirstPosition();

		NodeLinkedList * moveToLastPosition(NodeLinkedList * node);	
};


class lruCache
{

	private:
		int 					 numElements;
		int3					 cubeDim;
		int3					 cubeInc;
		int3					 realcubeDim;
		int					 offsetCube;

		std::map<index_node_t, NodeLinkedList *> indexStored;
		LinkedList	*			 queuePositions;

		float		*			 cacheData;
		FileManager	*			 fileManager;

		bool insertElement(index_node_t element, unsigned int * position);
	public:
		lruCache(char * file_name, char * dataset_name, int maxElements, int3 cDim, int cI);
		
		~lruCache();

		int	 get_numElements();
		int3	 get_cubeDim();
		int	 get_cubeInc();

		void changeDimensionCube(int maxElements, int3 cDim, int cI);

		void updateCache(visibleCube_t * visibleCubes, int num, int nLevels);
};
#endif
