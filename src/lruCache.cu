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

lruCache::lruCache(char * file_name, char * dataset_name, int maxElements, int3 cDim, int cI)
{
	numElements 	= maxElements;
	cubeDim 	= cDim;
	cubeInc		= make_int3(cI,cI,cI);
	realcubeDim	= cubeDim + 2 * cubeInc;
	offsetCube	= (cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);
	queuePositions  = new LinkedList(numElements);

	std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMalloc((void**)&cacheData, numElements*offsetCube*sizeof(float)))<<std::endl;

	fileManager = new FileManager(file_name, dataset_name);
}

lruCache::~lruCache()
{
	delete queuePositions;
	cudaFree(cacheData);
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

void lruCache::updateCache(visibleCube_t * visibleCubes, int num, int nLevels)
{
	unsigned int pos = 0;
	float * auxData = new float[offsetCube];

	int newCubes = 0;

	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].id != 0)
		{

			if (insertElement(visibleCubes[i].id, &pos))
			{
				int level = getIndexLevel(visibleCubes[i].id);
				int3 coord = getMinBoxIndex2(visibleCubes[i].id, level, nLevels);
				int3 minBox = coord - cubeInc;
				int3 maxBox = minBox + realcubeDim;
				fileManager->readHDF5_Voxel_Array(minBox, maxBox, auxData);

				visibleCubes[i].data = cacheData + pos*offsetCube;
				std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMemcpy((void*) visibleCubes[i].data, (const void*) auxData, offsetCube*sizeof(float), cudaMemcpyHostToDevice))<<std::endl;

				newCubes++;
			}
			else
			{
				visibleCubes[i].data = cacheData + pos*offsetCube;
			}
		}
	}

	delete[] auxData;

	std::cerr<<"New cubes on cache: "<<newCubes<<std::endl;
}
int	 lruCache::get_numElements(){return numElements;}
int3	 lruCache::get_cubeDim(){return cubeDim;}
int	 lruCache::get_cubeInc(){return cubeInc.x;}
