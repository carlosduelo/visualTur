#ifndef _CONFIG_H_
#define _CONFIG_H_

/* Necesary includes */

// CUDA 
#include <cuda_runtime.h>
#include <cutil_math.h>

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

typedef struct
{
	index_node_t id;
	float * data;
} visibleCube_t;

#endif
