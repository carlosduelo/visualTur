#include "Octree.hpp"
#include "mortonCodeUtil.hpp"
#include "cuda_help.hpp"
#include <iostream>
#include <fstream>

#define STACK_DIM 32

/*
 **********************************************************************************************
 ****** GPU Octree functions ******************************************************************
 **********************************************************************************************
 */

__device__ inline bool _cuda_checkRange(index_node_t * elements, index_node_t index, int min, int max)
{
	return  index == elements[min] 	|| 
		index == elements[max]	||
		(elements[min] < index && elements[max] > index);
}

__device__ bool _cuda_binary_search(index_node_t * elements, index_node_t index, int min, int max)
{
#if 0
	bool end = false;
	bool found = false;

	while(!end && !found)
	{
		int diff 	= max-min;
		int middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		end 		= diff < 1;
		found 		=  _cuda_checkRange(elements, index, middle, middle+1);
		if (index < elements[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}
	return found;
#endif
#if 1
	int minX = min;
	int maxX = max;
	while(1)
	{
		int diff = maxX-minX;
		if (diff < 1)
			return false;
		else if (diff == 1)
			return _cuda_checkRange(elements, index, minX, maxX);
		else
		{
			unsigned int middle = minX + (diff / 2);
			if (middle % 2 == 1) middle--;

			if (_cuda_checkRange(elements, index, middle, middle+1))
				return true;
			else if (index < elements[middle])
			{
				maxX = middle-1;
			}
			else if (index > elements[middle+1])
			{
				minX = middle + 2;
			}
			#if 0
			// XXX in cuda sin 2.x capability not printf
			else
				std::cout<<"Errro"<<std::endl;
			#endif
		}
	}
#endif
}

__device__ int _cuda_binary_search_closer(index_node_t * elements, index_node_t index, int min, int max)
{
#if 1
	bool end = false;
	bool found = false;
	int middle = 0;

	while(!end && !found)
	{
		int diff 	= max-min;
		middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		end 		= diff <= 1;
		found 		=  _cuda_checkRange(elements, index, middle, middle+1);
		if (index < elements[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}
	return middle;
#endif
#if 0
	while(1)
	{
		int diff = max-min;
		unsigned int middle = min + (diff / 2);
		if (diff <= 1)
		{
			if (middle % 2 == 1) middle--;
			return middle;
		}
		else
		{
			if (middle % 2 == 1) middle--;

			if (_cuda_checkRange(elements, index, middle, middle+1))
				return middle;
			else if (index < elements[middle])
			{
				max = middle-1;
			}
			else if (index > elements[middle+1])
			{
				min = middle + 2;
			}
			#if 0
			// XXX en cuda me arriesgo... malo...
			else
				std::cout<<"Errro"<<std::endl;
			#endif
		}
	}
#endif
}

__device__  bool _cuda_searchSecuential(index_node_t * elements, index_node_t index, int min, int max)
{
	bool find = false;
	for(int i=min; i<max; i+=2)
		if (_cuda_checkRange(elements, index, i, i+1))
			find = true;

	return find;
}

__device__ int _cuda_searchChildren(index_node_t * elements, int size, index_node_t father, index_node_t * children)
{
	index_node_t childrenID = father << 3;
	int numChild = 0;

	if (size==2)
	{
		for(int i=0; i<8; i++)
		{
			if (_cuda_checkRange(elements, childrenID,0,1))
			{
				children[numChild] = childrenID;
				numChild++;
			}
			childrenID++;
		}
	}
	else
	{
		unsigned int closer1 = _cuda_binary_search_closer(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		for(int i=0; i<8; i++)
		{
			if (_cuda_searchSecuential(elements, childrenID, closer1, closer8))
			{
				children[numChild] = childrenID;
				numChild++;
			}
			childrenID++;
		}
	}
	return numChild;
}

__device__ bool _cuda_RayAABB(index_node_t index, float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels)
{
	int3 minBox;
	int3 maxBox;
	int level;
	minBox = getMinBoxIndex(index, &level, nLevels); 
	int dim = (1<<(nLevels-level));
	maxBox.x = dim + minBox.x;
	maxBox.y = dim + minBox.y;
	maxBox.z = dim + minBox.z;
	bool hit = true;

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1 / dir.x;
	if (divx >= 0)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1 / dir.y;
	if (divy >= 0)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
	{
		hit = false;
	}

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1 / dir.z;
	if (divz >= 0)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
	{
		hit = false;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0)
	 	*tnear=0.0;
	else
		*tnear=tmin;
	*tfar=tmax;

	return hit;
}

__device__ bool _cuda_RayAABB2(float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, int3 minBox, int level)
{
	int3 maxBox;
	int dim = (1<<(nLevels-level));
	maxBox.x = dim + minBox.x;
	maxBox.y = dim + minBox.y;
	maxBox.z = dim + minBox.z;
	bool hit = true;

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1 / dir.x;
	if (divx >= 0)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1 / dir.y;
	if (divy >= 0)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
	{
		hit = false;
	}

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1 / dir.z;
	if (divz >= 0)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
	{
		hit = false;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0)
	 	*tnear=0.0;
	else
		*tnear=tmin;
	*tfar=tmax;

	return hit;

}

__device__ int _cuda_searchChildrenValidAndHit(index_node_t * elements, int size, float3 origin, float3 ray, index_node_t father, index_node_t * children, float * tnears, float * tfars, int nLevels)
{
	index_node_t childrenID = father << 3;
	int numChild = 0;
/*
	if ((childrenID+7) < elements[0] || (childrenID) > elements[size-1])
		return 0;
*/
	if (size==2)
	{
		for(int i=0; i<8; i++)
		{
			if (	_cuda_checkRange(elements, childrenID,0,1) &&
				_cuda_RayAABB(childrenID, origin, ray,  &tnears[numChild], &tfars[numChild], nLevels))
			{
				children[numChild] = childrenID;
				numChild++;
			}
			childrenID++;
		}
	}
	else
	{
		unsigned int closer1 = _cuda_binary_search_closer(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		for(int i=0; i<8; i++)
		{
			if (	_cuda_searchSecuential(elements, childrenID, closer1, closer8) &&
				_cuda_RayAABB(childrenID, origin, ray, &tnears[numChild], &tfars[numChild], nLevels) && tfars[numChild]>=0.0f)
			{
				children[numChild] = childrenID;
				numChild++;
			}
			childrenID++;
		}
	}
	return numChild;
}

__device__ int _cuda_searchChildrenValidAndHit2(index_node_t * elements, int size, float3 origin, float3 ray, index_node_t father, index_node_t * children, float * tnears, float * tfars, int nLevels, int level, int3 minB)
{
	index_node_t childrenID = father << 3;
	int numChild = 0;
	int dim = (1<<(nLevels-level));
	int3 minBox = make_int3(minB.x, minB.y, minB.z);
/*
	if ((childrenID+7) < elements[0] || (childrenID) > elements[size-1])
		return 0;
*/
	if (size==2)
	{
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_checkRange(elements, childrenID,0,1))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
	}
	else
	{
		unsigned int closer1 = _cuda_binary_search_closer(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) &&  tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
		minBox.z+=dim;
		if (	_cuda_RayAABB2(origin, ray,  &tnears[numChild], &tfars[numChild], nLevels, minBox, level) && tfars[numChild]>=0.0 &&
			_cuda_searchSecuential(elements, childrenID, closer1, closer8))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
	}
	return numChild;
}

__device__ void _cuda_sortList(index_node_t * list, float * order, int size)
{
	int n = size;
	while(n != 0)
	{
		int newn = 0;
		for(int id=1; id<size; id++)
		{
			if (order[id-1] > order[id])
			{
				index_node_t auxID = list[id];
				list[id] = list[id-1];
				list[id-1] = auxID;

				float aux = order[id];
				order[id] = order[id-1];
				order[id-1] = aux;

				newn=id;
			}
		}
		n = newn;
	}
}


/* Return number of children and children per thread */
__global__ void cuda_searchChildren(index_node_t ** elements, int * size, int level, int numfathers, index_node_t * father, int * numChildren, index_node_t * children)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	int idC = id*8;

	if (id < numfathers)
	{
		numChildren[id] = _cuda_searchChildren(elements[level], size[level], father[id], &children[idC]);
	}

	return;
}


__global__ void cuda_search(index_node_t * elements, index_node_t * index, int numIndex, int min, int max, bool * search_array)
{
	int blockD = blockDim.x * blockDim.y;
	int blockFila = blockD*gridDim.y;

	int id = (blockIdx.y * blockD + blockIdx.x*blockFila) + threadIdx.y + threadIdx.x*blockDim.y;

	if (id < numIndex)
		search_array[id] = _cuda_binary_search(elements, index[id], min, max);
}

__device__ int3 _cuda_updateCoordinates(int maxLevel, int cLevel, index_node_t cIndex, int nLevel, index_node_t nIndex, int3 minBox)
{
	if ( 0 == nIndex)
	{
		return make_int3(0,0,0);
	}
	else if (cLevel < nLevel)
	{
		index_node_t mask = (index_node_t) 1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		return minBox;

	}
	else if (cLevel > nLevel)
	{
		return	getMinBoxIndex2(nIndex, nLevel, maxLevel);
	}
	else
	{
		index_node_t mask = (index_node_t)1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.z -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.y -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.x -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		return minBox;
	}
}

__device__ void _cuda_getFirtsVoxel(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 ray, int finalLevel, visibleCube_t * indexNode, int * p_stackActual, index_node_t * p_stackIndex, int * p_stackLevel)
{

	if (indexNode->state ==  NOCUBE && (*p_stackActual) >= 0)
	{
		index_node_t * 	stackIndex 	= p_stackIndex;
		int	     *	stackLevel 	= p_stackLevel;
		int 		stackActual 	= *p_stackActual;

		int 		currentLevel 	= stackLevel[stackActual];
		index_node_t 	current 	= stackIndex[stackActual];
		int3		minBox 		= getMinBoxIndex2(current, currentLevel, nLevels);

		index_node_t 	children[8];
		float		tnears[8];
		float		tfars[8];
	
		while(stackActual >= 0)
		{
			#if _DEBUG_
			if (stackActual >= STACK_DIM)
			{
				printf("ERROR");
				return;
			}
			#endif

			if (stackLevel[stackActual] == finalLevel)
			{
				indexNode->id = stackIndex[stackActual];
				//printf("-->%d %d %lld\n", finalLevel, stackActual, stackIndex[stackActual]);
				indexNode->state = CUBE;
				*p_stackActual = stackActual - 1;
				return;
			}

			minBox = _cuda_updateCoordinates(nLevels, currentLevel, current, stackLevel[stackActual], stackIndex[stackActual], minBox);

			current 	= stackIndex[stackActual];
			currentLevel 	= stackLevel[stackActual];
			stackActual--;


			int nValid  = _cuda_searchChildrenValidAndHit2(octree[currentLevel+1], sizes[currentLevel+1], origin, ray, current, children, tnears, tfars, nLevels, currentLevel+1, minBox);
			// Sort the list mayor to minor
			int n = nValid;
			while(n != 0)
			{
				int newn = 0;
				#pragma unroll
				for(int id=1; id<nValid; id++)
				{
					if (tnears[id-1] > tnears[id])
					{
						index_node_t auxID = children[id];
						children[id] = children[id-1];
						children[id-1] = auxID;

						float aux = tnears[id];
						tnears[id] = tnears[id-1];
						tnears[id-1] = aux;

						aux = tfars[id];
						tfars[id] = tfars[id-1];
						tfars[id-1] = aux;

						newn=id;
					}
					else if (tnears[id-1] == tnears[id] && tfars[id-1] > tfars[id])
					{
						index_node_t auxID = children[id];
						children[id] = children[id-1];
						children[id-1] = auxID;

						float aux = tnears[id];
						tnears[id] = tnears[id-1];
						tnears[id-1] = aux;

						aux = tfars[id];
						tfars[id] = tfars[id-1];
						tfars[id-1] = aux;

						newn=id;
					}
				}
				n = newn;
			}
			for(int i=nValid-1; i>=0; i--)
			{
				stackActual++;
				stackIndex[stackActual] = children[i];
				stackLevel[stackActual] = currentLevel+1;
			}
		}

		// NO CUBE FOUND
		indexNode->id 	= 0;
		*p_stackActual = stackActual;
		return;
	}
}

__global__ void cuda_getFirtsVoxel(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 * rays, int finalLevel, visibleCube_t * indexNode, int numElements, int * stackActual, index_node_t * stackIndex, int * stackLevel)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
		float tnear, tfar;
		if (!_cuda_checkRange(octree[0],1,0,1) || !_cuda_RayAABB(1, origin, rays[i],  &tnear, &tfar, nLevels))
		{
			indexNode[i].id = 0;
			return;
		}
		else
			_cuda_getFirtsVoxel(octree, sizes, nLevels, origin, rays[i], finalLevel, &indexNode[i], &stackActual[i], &stackIndex[i*STACK_DIM], &stackLevel[i*STACK_DIM]);
	}

	return;
}

__global__ void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}

#if _OG_
__global__ void cuda_resetState(int nLevels, int numElements, index_node_t * GstackIndex, int * GstackActual, int * GcurrentLevel, int * Gnumbro)
{
	
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
		GstackIndex[i*HPN*nLevels] = 1;
		Gnumbro[i*nLevels] = 1; 
		GstackActual[i] = 0;
		GcurrentLevel[i] = 0;
	}
}
#else
__global__ void cuda_resetState(int numElements, int * stackActual, index_node_t * stackIndex, int * stackLevel)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
		stackActual[i] = 0;
		stackIndex[i*STACK_DIM] = 1;
		stackLevel[i*STACK_DIM] = 0;
	}
}
#endif

/*
******************************************************************************************************
************ METHODS OCTREEMCUDA *********************************************************************
******************************************************************************************************
*/

/* Lee el Octree de un fichero */
Octree::Octree(const char * file_name, Camera * p_camera, int p_maxLevel)
{
	camera = p_camera;
	maxLevel = p_maxLevel;

        /* Read octree from file */
	std::ifstream file;

        file.open(file_name, std::ifstream::binary);
        int magicWord;

        file.read((char*)&magicWord, sizeof(magicWord));
        if (magicWord != 919278872)
        {
                std::cerr<<"Error, invalid file format"<<std::endl;
                throw std::exception();
        }

        file.read((char*)&isosurface, sizeof(isosurface));
        file.read((char*)&dimension, sizeof(dimension));
        file.read((char*)&realDim.x, sizeof(realDim.x));
        file.read((char*)&realDim.y, sizeof(realDim.y));
        file.read((char*)&realDim.z, sizeof(realDim.z));
        file.read((char*)&nLevels, sizeof(int));

        std::cout<<"Octree de dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" niveles "<<nLevels<<std::endl;

        index_node_t ** octreeCPU	= new index_node_t*[nLevels+1];
        int 	*	sizesCPU	= new int[nLevels+1];

        for(int i=nLevels; i>=0; i--)
        {
                int numElem = 0;
                file.read((char*)&numElem, sizeof(numElem));
                //std::cout<<"Dimension del node en el nivel "<<i<<" es de "<<powf(2.0,*nLevels-i)<<std::endl;
                //std::cout<<"Numero de elementos de nivel "<<i<<" "<<numElem<<std::endl;
                sizesCPU[i] = numElem;
                octreeCPU[i] = new index_node_t[numElem];
                for(int j=0; j<numElem; j++)
                {
                        index_node_t node = 0;
                        file.read((char*) &node, sizeof(index_node_t));
                        octreeCPU[i][j]= node;
                }
        }

        file.close();
	/* end reading octree from file */

	std::cerr<<"Copying octree to GPU"<<std::endl;

	int total = 0;
	for(int i=0; i<=maxLevel; i++)
		total+=sizesCPU[i];

	std::cerr<<"Allocating memory octree CUDA octree "<<(maxLevel+1)*sizeof(index_node_t*)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&octree, (maxLevel+1)*sizeof(index_node_t*))) << std::endl;
	std::cerr<<"Allocating memory octree CUDA memory "<<total*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&memoryGPU, total*sizeof(index_node_t))) << std::endl;
	std::cerr<<"Allocating memory octree CUDA sizes "<<(maxLevel+1)*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&sizes,   (maxLevel+1)*sizeof(int))) << std::endl;

	/* Compiando sizes */
	std::cerr<<"Coping to device: "<< cudaGetErrorString(cudaMemcpy((void*)sizes, (void*)sizesCPU, (maxLevel+1)*sizeof(int), cudaMemcpyHostToDevice)) << std::endl;
	/* end sizes */

	/* Copying octree */

	int offset = 0;
	for(int i=0; i<=maxLevel; i++)
	{
		std::cerr<<"Coping to device: "<< cudaGetErrorString(cudaMemcpy((void*)(memoryGPU+offset), (void*)octreeCPU[i], sizesCPU[i]*sizeof(index_node_t), cudaMemcpyHostToDevice)) << std::endl;
		offset+=sizesCPU[i];
	}	
	
	dim3 blocks(1);
	dim3 threads(maxLevel+1);
	
	insertOctreePointers<<<blocks,threads>>>(octree, sizes, memoryGPU);
	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	std::cerr<<"Synchronizing: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
	std::cerr<<"End copying octree to GPU"<<std::endl;

	delete[] sizesCPU;
        for(int i=0; i<=nLevels; i++)
        {
                delete[] octreeCPU[i];
        }
        delete[]        octreeCPU;

	// Create octree State
	#if _OG_
		std::cerr<<"Allocating stackIndex "<<HPN*nLevels*camera->get_numRays()*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GstackIndex, HPN*nLevels*camera->get_numRays()*sizeof(index_node_t))) << std::endl;
		std::cerr<<"Allocating stackActual "<<camera->get_numRays()*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GstackActual, camera->get_numRays()*sizeof(int))) << std::endl;
		std::cerr<<"Allocating stackcurrentLevel "<<camera->get_numRays()*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GcurrentLevel, camera->get_numRays()*sizeof(int))) << std::endl;
		std::cerr<<"Allocating numbro "<<nLevels*camera->get_numRays()*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&Gnumbro,nLevels*camera->get_numRays()*sizeof(int))) << std::endl;
		resetState();
	#else
	std::cerr<<"Allocating memory octree state stackIndex "<<camera->get_numRays()*STACK_DIM*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GstackIndex, camera->get_numRays()*STACK_DIM*sizeof(index_node_t))) << std::endl;
	std::cerr<<"Allocating memory octree state stackActual "<<camera->get_numRays()*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GstackActual, camera->get_numRays()*sizeof(int))) << std::endl;
	std::cerr<<"Allocating memory octree state stackLevel "<<camera->get_numRays()*STACK_DIM*sizeof(int)/1024.0f/1024.0f<<" MB: "<< cudaGetErrorString(cudaMalloc(&GstackLevel, camera->get_numRays()*STACK_DIM*sizeof(int))) << std::endl;
	#endif

	resetState();

}

Octree::~Octree()
{
	cudaFree(octree);	
	cudaFree(memoryGPU);	
	cudaFree(sizes);	
}

float Octree::getIsosurface()
{
	return isosurface; 
}

int Octree::getnLevels()
{
	return nLevels;
}

/* Dado un rayo devuelve true si el rayo impacta contra el volumen, el primer box del nivel dado contra el que impacta y la distancia entre el origen del rayo y la box */
bool Octree::getBoxIntersected(int finalLevel, visibleCube_t * visibleGPU, visibleCube_t * visibleCPU)
{
	std::cerr<<"Getting firts box intersected"<<std::endl;

	int numElements = camera->get_numRays();
	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	//std::cerr<<"Set HEAP size: "<< cudaGetErrorString(cudaThreadSetLimit(cudaLimitMallocHeapSize , numElements*1216)) << std::endl;

	#if _OG_
	cuda_getFirtsVoxel<<<blocks,threads>>>(octree, sizes, nLevels, camera->get_position(), camera->get_rayDirections(), finalLevel, visibleGPU, numElements,GstackIndex, GstackActual, GcurrentLevel, Gnumbro);
	#else
	cuda_getFirtsVoxel<<<blocks,threads>>>(octree, sizes, nLevels, camera->get_position(), camera->get_rayDirections(), finalLevel, visibleGPU, numElements, GstackActual, GstackIndex, GstackLevel);
	#endif
	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;

	std::cerr<<"Coping to host visibleCubes: "<< cudaGetErrorString(cudaMemcpy((void*)visibleCPU, (const void*)visibleGPU, numElements*sizeof(visibleCube_t), cudaMemcpyDeviceToHost)) << std::endl;

	std::cerr<<"End Getting firts box intersected"<<std::endl;
	return true;
}

void Octree::resetState()
{
#if _OG_
	int numElements = camera->get_numRays();
	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_resetState<<<blocks, threads>>>(nLevels, numElements, GstackIndex, GstackActual, GcurrentLevel, Gnumbro); 
	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
#else
	int numElements = camera->get_numRays();
	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_resetState<<<blocks,threads>>>(numElements, GstackActual, GstackIndex, GstackLevel);
	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
#endif
}
