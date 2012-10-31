/*
 * CUDA Help 
 *
 */

#ifndef _CUDA_HELP_H_
#define _CUDA_HELP_H_

#define BLOCK_SIZE 128

dim3 getBlocks(int dim)
{
	if (dim <= BLOCK_SIZE)
	{
		dim3 blocks(1);//,0,0);
		return blocks;
	}
	else// if (dim<=(BLOCK_SIZE*BLOCK_SIZE))
	{
		int numBlocks = dim / 128;
		if (dim % BLOCK_SIZE !=0) numBlocks++;
		int bpA = sqrt(numBlocks);
		int bp  = floorf(bpA) + 1;
		dim3 blocks(bp,bp);//,0); 
		return blocks;
	}
}

dim3 getThreads(int dim)
{
	int t = 32;
	while(dim>t && t<BLOCK_SIZE)
	{
		t+=32;
	}

	dim3 threads(t);//,0,0);
	return threads;
}
#endif
