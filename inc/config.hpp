#ifndef _CONFIG_H_
#define _CONFIG_H_

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

#define PAINTED 	(unsigned char)4
#define CACHED 		(unsigned char)2
#define NOCACHED 	(unsigned char)1
#define NOCUBE		(unsigned char)0

typedef struct
{
	index_node_t 	id;
	float * 	data;
	unsigned char   state;
} visibleCube_t;

#endif
