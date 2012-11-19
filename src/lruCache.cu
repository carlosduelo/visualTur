#include "lruCache.hpp"
#include "mortonCodeUtil.hpp"
#include "cuda_help.hpp"
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
			memoryList[i].cubeID = 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after = 0;
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
			memoryList[i].cubeID = 0;
		}
		else
		{
			memoryList[i].after = &memoryList[i+1];
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
			memoryList[i].cubeID = 0;
		}
	}
}

LinkedList::~LinkedList()
{
	delete[] memoryList;
}


NodeLinkedList * LinkedList::getFromFirstPosition(index_node_t newIDcube, index_node_t * removedIDcube)
{
	NodeLinkedList * first = list;

	list = first->after;
	list->before = 0;
	
	first->after  = 0;
	first->before = last;
	
	last->after = first;
	
	last = first;
	*removedIDcube = last->cubeID;
	last->cubeID = newIDcube;

	return first;

}

NodeLinkedList * LinkedList::moveToLastPosition(NodeLinkedList * node)
{
	if (node->before == 0)
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

	std::cerr<<"Creating cache in GPU "<< numElements*offsetCube*sizeof(float)/1024/1024<<" MB: "<<cudaGetErrorString(cudaMalloc((void**)&cacheData, numElements*offsetCube*sizeof(float)))<<std::endl;
	cacheDataCPU = new float[numElementsCPU*offsetCube];
	//std::cerr<<"Creating cache in CPU: "<<cudaGetErrorString(cudaHostAlloc((void**)&cacheDataCPU, numElementsCPU*offsetCube*sizeof(float),cudaHostAllocDefault))<<std::endl;

	fileManager = new FileManager(file_name, dataset_name);
	access = 0;
	hitsCPU = 0;
	hitsGPU = 0;
	missCPU = 0;
	missGPU = 0;
	timingAccess = 0.0;
	timinghitsCPU = 0.0;
	timingmissCPU = 0.0;
	timinghitsGPU = 0.0;
	timingmissGPU = 0.0;
}

lruCache::~lruCache()
{
	delete queuePositions;
	delete queuePositionsCPU;
	cudaFree(cacheData);
	delete[] cacheDataCPU;
	//cudaFreeHost(cacheDataCPU);
	delete fileManager;
}

void lruCache::changeDimensionCube(int maxElements, int3 cDim, int cI)
{
	delete queuePositions;
	cudaFree(cacheData);
	indexStored.clear();
	indexStoredCPU.clear();
	delete[] cacheDataCPU;
	//cudaFreeHost(cacheDataCPU);
	delete queuePositionsCPU;

	numElements 	= maxElements;
	cubeDim 	= cDim;
	cubeInc		= make_int3(cI,cI,cI);
	realcubeDim	= cubeDim + 2 * cubeInc;
	offsetCube	= realcubeDim.x*realcubeDim.y*realcubeDim.z ;//(cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);
	queuePositions  = new LinkedList(numElements);

	std::cerr<<"Creating cache in GPU "<< numElements*offsetCube*sizeof(float)/1024/1024<<" MB: "<<cudaGetErrorString(cudaMalloc((void**)&cacheData, numElements*offsetCube*sizeof(float)))<<std::endl;
	cacheDataCPU = new float[numElementsCPU*offsetCube];
	//std::cerr<<"Creating cache in CPU: "<<cudaGetErrorString(cudaHostAlloc((void**)&cacheDataCPU, numElementsCPU*offsetCube*sizeof(float),cudaHostAllocDefault))<<std::endl;
	access = 0;
	hitsCPU = 0;
	hitsGPU = 0;
	missCPU = 0;
	missGPU = 0;
}

void lruCache::updateCube(visibleCube_t * cube, int nLevels, int * nEinsertedCPU, int * nEinsertedGPU)
{
	
	struct timeval stA, endA;
	access++;
	//std::map<index_node_t, NodeLinkedList *>::iterator it;
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
	unsigned posC = 0;
	unsigned posG = 0;

	gettimeofday(&stA, NULL);
	// Update in GPU cache
	it = indexStored.find(cube->id);
	if ( it == indexStored.end() ) // Not exists
	{
		if (*nEinsertedGPU >= numElements)
		{
			cube->state = NOCACHED;
			gettimeofday(&endA, NULL);
			timingAccess += ((endA.tv_sec  - stA.tv_sec) * 1000000u + endA.tv_usec - stA.tv_usec) / 1.e6;
			//std::cout<<"---------------------------------------------------------------------------------->"<<std::endl;
			return;
		}
		else
		{
			// Update in CPU Cache
			it = indexStoredCPU.find(cube->id);

			if ( it == indexStoredCPU.end() ) // Not exists
			{

				if (*nEinsertedCPU >= numElementsCPU)
				{
					cube->state = NOCACHED;
					gettimeofday(&endA, NULL);
					timingAccess += ((endA.tv_sec  - stA.tv_sec) * 1000000u + endA.tv_usec - stA.tv_usec) / 1.e6;
					return;
				}

				missCPU++;
				struct timeval stC, endC;
				gettimeofday(&stC, NULL);

				index_node_t removedIDcube = 0;
				/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
				NodeLinkedList * node = queuePositionsCPU->getFromFirstPosition(cube->id, &removedIDcube);

				posC = node->element; 

				indexStoredCPU.insert(std::pair<int, NodeLinkedList *>(cube->id, node));
				if (removedIDcube != 0)
					indexStoredCPU.erase(indexStoredCPU.find(removedIDcube));
				int level = getIndexLevel(cube->id);
				int3 coord = getMinBoxIndex2(cube->id, level, nLevels);
				int3 minBox = coord - cubeInc;
				int3 maxBox = minBox + realcubeDim;
				fileManager->readHDF5_Voxel_Array(minBox, maxBox, cacheDataCPU + posC*offsetCube);
				(*nEinsertedCPU)++;

				gettimeofday(&endC, NULL);
				timingmissCPU += ((endC.tv_sec  - stC.tv_sec) * 1000000u + endC.tv_usec - stC.tv_usec) / 1.e6;
				
			}
			else // Exist
			{
				hitsCPU++;		

				missCPU++;
				struct timeval stC, endC;
				gettimeofday(&stC, NULL);

				/* If the elements is already in the cache remove from the queue and insert at the end */
				NodeLinkedList * node = it->second;
				posC = node->element; 

				queuePositionsCPU->moveToLastPosition(node);

				gettimeofday(&endC, NULL);
				timinghitsCPU += ((endC.tv_sec  - stC.tv_sec) * 1000000u + endC.tv_usec - stC.tv_usec) / 1.e6;
			}

			struct timeval st, end;
			gettimeofday(&st, NULL);

			index_node_t removedIDcube = 0;
			/* Get from the queue the first element, add to hashtable (index, lastPosition) and enqueue the position */
			NodeLinkedList * node = queuePositions->getFromFirstPosition(cube->id, &removedIDcube);

			posG = node->element; 

			indexStored.insert(std::pair<int, NodeLinkedList *>(cube->id, node));
			if (removedIDcube != 0)
				indexStored.erase(indexStored.find(removedIDcube));
			cube->data = cacheData + posG*offsetCube;
			cube->state = CACHED;
			std::cerr<<"Creating cache in GPU: "<<cudaGetErrorString(cudaMemcpy((void*) cube->data, (const void*) (cacheDataCPU + posC*offsetCube), offsetCube*sizeof(float), cudaMemcpyHostToDevice))<<std::endl;

			(*nEinsertedGPU)++;

			gettimeofday(&end, NULL);
			timingmissGPU += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
			missGPU++;
		}
	}
	else // Exist
	{
		hitsGPU++;
		struct timeval st, end;
		gettimeofday(&st, NULL);

		/* If the elements is already in the cache remove from the queue and insert at the end */
		NodeLinkedList * node = it->second;
		posG= node->element; 
		queuePositions->moveToLastPosition(node);
		cube->data = cacheData + posG*offsetCube;
		cube->state = CACHED;

		gettimeofday(&end, NULL);
		timinghitsGPU += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
	}

	
	gettimeofday(&endA, NULL);
	timingAccess += ((endA.tv_sec  - stA.tv_sec) * 1000000u + endA.tv_usec - stA.tv_usec) / 1.e6;
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
void 	lruCache::printStatistics()
{ 
	std::cout<<"Access "<<access<<" hits CPU "<<hitsCPU<<" miss CPU "<<missCPU<<" hits GPU "<<hitsGPU<<" miss GPU "<<missGPU<<std::endl;
	std::cout<<"Seconds spend in Accesses "<<timingAccess<<" hits CPU "<<timinghitsCPU<<" miss CPU "<<timingmissCPU<<" hits GPU "<<timinghitsGPU<<" miss GPU "<<timingmissGPU<<std::endl;
}
