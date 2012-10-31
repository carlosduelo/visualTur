/*
 * morton Code functions 
 *
 */

#ifndef _MORTON_H_
#define _MORTON_H_
#include "config.hpp"

inline __host__ __device__ index_node_t dilateInteger(index_node_t x)
{
	x = (x | (x << 20)) & 0x000001FFC00003FF;
	x = (x | (x << 10)) & 0x0007E007C00F801F;
	x = (x | (x << 4))  & 0x00786070C0E181C3;
	x = (x | (x << 2))  & 0x0199219243248649;
	x = (x | (x << 2))  & 0x0649249249249249;
	x = (x | (x << 2))  & 0x1249249249249249;
	return x;
}

inline __host__ __device__ int getIndexLevel(index_node_t index)
{
	if (index >= 144115188075855872)
		return 19;
	else if (index >= 18014398509481984)
		return 18;
	else if (index >= 2251799813685248)
		return 17;
	else if (index >= 281474976710656)
		return 16;
	else if (index >= 35184372088832)
		return 15;
	else if (index >= 4398046511104)
		return 14;
	else if (index >= 549755813888)
		return 13;
	else if (index >= 68719476736)
		return 12;
	else if (index >= 8589934592)
		return 11;
	else if (index >= 1073741824)
		return 10;
	else if (index >= 134217728)
		return 9;
	else if (index >= 16777216)
		return 8;
	else if (index >= 2097152)
		return 7;
	else if (index >= 262144)
		return 6;
	else if (index >= 32768)
		return 5;
	else if (index >= 4096)
		return 4;
	else if (index >= 512)
		return 3;
	else if (index >= 64)
		return 2;
	else if (index >= 8)
		return 1;
	else if (index == 1)
		return 0;
	else
		return -1;
}

inline __host__ __device__ int3 getMinBoxIndex(index_node_t index, int * level, int nLevels)
{
	int3 Box;
	Box.x  = 0;
	Box.y  = 0;
	Box.z  = 0;
	*level = 0;

	if (index == 1)
		return Box; // minBOX (0,0,0) and level 0

	*level = getIndexLevel(index);

	index_node_t mask = 1;

	for(int l=(*level); l>0; l--)
	{
		Box.z +=  (index & mask) << (nLevels-l); index>>=1;
		Box.y +=  (index & mask) << (nLevels-l); index>>=1;
		Box.x +=  (index & mask) << (nLevels-l); index>>=1;
	}

	#if 0
	// XXX en cuda me arriesgo.... malo...
	if (index!=1)
		std::cerr<<"Error getting minBox from index"<<std::endl;
	#endif

	return Box;
	
}

inline __host__ __device__ int3 getMinBoxIndex2(index_node_t index, int level, int nLevels)
{
	int3 minBox = make_int3(0,0,0);

	if (index == 1)
		return minBox;// minBOX (0,0,0) and level 0

	index_node_t mask = 1;

	for(int l=level; l>0; l--)
	{
		minBox.z +=  (index & mask) << (nLevels-l); index>>=1;
		minBox.y +=  (index & mask) << (nLevels-l); index>>=1;
		minBox.x +=  (index & mask) << (nLevels-l); index>>=1;
	}

	#if 0
	// XXX en cuda me arriesgo.... malo...
	if (index!=1)
		std::cerr<<"Error getting minBox from index"<<std::endl;
	#endif

	return minBox;
	
}

inline __host__ __device__ index_node_t coordinateToIndex(int3 pos, int level, int nLevels)
{
	if (level==0)
		return 1;

	index_node_t code 	= (index_node_t)1 << (nLevels*3);
	index_node_t xcode 	= dilateInteger((index_node_t) pos.x) << 2;
	index_node_t ycode 	= dilateInteger((index_node_t) pos.y) << 1;
	index_node_t zcode 	= dilateInteger((index_node_t) pos.z);

	code = code | xcode | ycode | zcode;

	code>>=(nLevels-level)*3;

	/* XXX CUDA not supported
	if (code==0)
	{
		std::cout<<"Error, index cannot be zero "<<x<<","<<y<<","<<z<<" level "<<level<<std::endl;
		std::exception();
		exit(1);
	}
	*/
	return code;
}

#endif
