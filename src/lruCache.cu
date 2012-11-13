#include "lruCache.hpp"
#include "mortonCodeUtil.hpp"
#include <iostream>
#include <fstream>

LinkedList::LinkedList(int size)
{
	memoryList = new NodeLinkedList[size];
	list = memoryList;
	last = &memoryList[size-1];
	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after = &memoryList[i+1];
			memoryList[i].before = 0;
			memoryList[i].element = i;
		}
		else if (i==size-1)
		{
			memoryList[i].after = 0;
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
		}
		else
		{
			memoryList[i].after = &memoryList[i+1];
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
		}
	}
}

LinkedList::~LinkedList()
{
	delete[] memoryList;
}


NodeLinkedList * LinkedList::getFromFirstPosition()
{
	NodeLinkedList * first = list;

	list = first->after;
	list->before = 0;
	
	first->after  = 0;
	first->before = last;
	
	last->after = first;
	
	last = first;

	return first;

}

NodeLinkedList * LinkedList::moveToLastPosition(NodeLinkedList * node)
{
	if (node->before == 0)
		return LinkedList::getFromFirstPosition();
	else if (node->after == 0)
	{
		return node;
	}
	else
	{
		node->before->after = node->after;
		node->after->before = node->before;
		
		last->after = node;
		
		node->before = last;
		node->after  = 0;
		last = node;
		
		return node;
	}
}

lruCache::lruCache(char * file_name, char * dataset_name, int maxElements, int3 cDim, int cI, int maxElementsCPU)
{
	numElements 	= maxElements;
	numElementsCPU 	= maxElementsCPU;
	cubeDim 	= cDim;
	cubeInc		= make_int3(cI,cI,cI);
	realcubeDim	= cubeDim + 2 * cubeInc;
	offsetCube	= (cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);
	queuePositions  = new LinkedList(numElements);
	queuePositionsCPU = new LinkedList(numElementsCPU);

	std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMalloc((void**)&cacheData, numElements*offsetCube*sizeof(float)))<<std::endl;
	cacheDataCPU = new float[numElementsCPU*offsetCube];

	fileManager = new FileManager(file_name, dataset_name);
}

lruCache::~lruCache()
{
	delete queuePositions;
	delete queuePositionsCPU;
	cudaFree(cacheData);
	delete[] cacheDataCPU;
	delete fileManager;
}

void lruCache::changeDimensionCube(int maxElements, int3 cDim, int cI)
{
	delete queuePositions;
	cudaFree(cacheData);
	indexStored.clear();

	numElements 	= maxElements;
	cubeDim 	= cDim;
	cubeInc		= make_int3(cI,cI,cI);
	realcubeDim	= cubeDim + 2 * cubeInc;
	offsetCube	= realcubeDim.x*realcubeDim.y*realcubeDim.z ;//(cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);
	queuePositions  = new LinkedList(numElements);

	std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMalloc((void**)&cacheData, numElements*offsetCube*sizeof(float)))<<std::endl;
}

#if 0
bool lruCache::insertElement(index_node_t element, unsigned int * position)
{
	std::map<index_node_t, NodeLinkedList *>::iterator it;
	it = indexStored.find(element);

	if ( it == indexStored.end() ) // Not exists
	{
		/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
		NodeLinkedList * node = queuePositions->getFromFirstPosition();

		*position = node->element; 

		indexStored.insert(std::pair<int, NodeLinkedList *>(element, node));

		return true;
	}
	else // Exist
	{
		/* If the elements is already in the cache remove from the queue and insert at the end */
		NodeLinkedList * node = it->second;
		*position = node->element; 

		queuePositions->moveToLastPosition(node);

		return false;
	}
}

bool lruCache::insertElementCPU(index_node_t element, unsigned int * position)
{
	std::map<index_node_t, NodeLinkedList *>::iterator it;
	it = indexStoredCPU.find(element);

	if ( it == indexStoredCPU.end() ) // Not exists
	{
		/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
		NodeLinkedList * node = queuePositionsCPU->getFromFirstPosition();

		*position = node->element; 

		indexStoredCPU.insert(std::pair<int, NodeLinkedList *>(element, node));

		return true;
	}
	else // Exist
	{
		/* If the elements is already in the cache remove from the queue and insert at the end */
		NodeLinkedList * node = it->second;
		*position = node->element; 

		queuePositionsCPU->moveToLastPosition(node);

		return false;
	}
}
#endif

void lruCache::updateCube(visibleCube_t * cube, int nLevels, int * nEinsertedCPU, int * nEinsertedGPU)
{
	// Update in CPU Cache
	std::map<index_node_t, NodeLinkedList *>::iterator it;
	it = indexStoredCPU.find(cube->id);
	unsigned posC = 0;
	unsigned posG = 0;

	if ( it == indexStoredCPU.end() ) // Not exists
	{
		if (*nEinsertedCPU > numElementsCPU)
		{
			cube->state = NOCACHED;
			return;
		}

		/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
		NodeLinkedList * node = queuePositionsCPU->getFromFirstPosition();

		posC = node->element; 

		indexStoredCPU.insert(std::pair<int, NodeLinkedList *>(cube->id, node));
		int level = getIndexLevel(cube->id);
		int3 coord = getMinBoxIndex2(cube->id, level, nLevels);
		int3 minBox = coord - cubeInc;
		int3 maxBox = minBox + realcubeDim;
		fileManager->readHDF5_Voxel_Array(minBox, maxBox, cacheDataCPU + posC*offsetCube);
		(*nEinsertedCPU)++;
	}
	else // Exist
	{
		/* If the elements is already in the cache remove from the queue and insert at the end */
		NodeLinkedList * node = it->second;
		posC = node->element; 

		queuePositionsCPU->moveToLastPosition(node);
	}

	// Update in GPU cache
	it = indexStored.find(cube->id);
	if ( it == indexStored.end() ) // Not exists
	{
		if (*nEinsertedGPU > numElements)
		{
			cube->state = NOCACHED;
			std::cout<<"---------------------------------------------------------------------------------->"<<std::endl;
			return;
		}
		/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
		NodeLinkedList * node = queuePositions->getFromFirstPosition();

		posG = node->element; 

		indexStored.insert(std::pair<int, NodeLinkedList *>(cube->id, node));
		cube->data = cacheData + posG*offsetCube;
		cube->state = CACHED;
		std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMemcpy((void*) cube->data, (const void*) (cacheDataCPU + posC*offsetCube), offsetCube*sizeof(float), cudaMemcpyHostToDevice))<<std::endl;

		(*nEinsertedGPU)++;
	}
	else // Exist
	{
		/* If the elements is already in the cache remove from the queue and insert at the end */
		NodeLinkedList * node = it->second;
		posG= node->element; 
		queuePositions->moveToLastPosition(node);
		cube->data = cacheData + posG*offsetCube;
		cube->state = CACHED;
	}
}

void lruCache::updateCache(visibleCube_t * visibleCubes, int num, int nLevels)
{
	int newCubesC = 0;
	int newCubesG = 0;

	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].id != 0 && visibleCubes[i].state != PAINTED)
		{
			updateCube(&visibleCubes[i], nLevels, &newCubesC, &newCubesG);
		}
	}

	std::cerr<<"New cubes on cache: "<<newCubesC<<" CPU "<<newCubesG<<" GPU "<<std::endl;
}

int	 lruCache::get_numElements(){return numElements;}
int3	 lruCache::get_cubeDim(){return cubeDim;}
int	 lruCache::get_cubeInc(){return cubeInc.x;}
