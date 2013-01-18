#include "visualTur_device.hpp"
#include "hdf5.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>


#define WHERESTR  "Error[file "<<__FILE__<<", line "<<__LINE__<<"]: "
int getnLevelFile(char * file_name, char * dataset_name)
{

	hid_t           file_id;
	hid_t           dataset_id;
	hid_t           spaceid;
	int             ndim;
	hsize_t         dims[3];
	if ((file_id    = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		std::cerr<< WHERESTR<<" unable to open the requested file"<<std::endl;
		exit(0);
	}

	if ((dataset_id = H5Dopen1(file_id, dataset_name)) < 0 )
	{
		std::cerr<<WHERESTR<<" unable to open the requested data set"<<std::endl;
		exit(0);
	}

	if ((spaceid    = H5Dget_space(dataset_id)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to open the requested data space"<<std::endl;
		exit(0);
	}

	if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" handling file"<<std::endl;
		exit(0);
	}

	herr_t      status;

	if ((status = H5Dclose(dataset_id)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to close the data set"<<std::endl;
		exit(0);
	}


	if ((status = H5Fclose(file_id)) < 0);
	{
		std::cerr<<WHERESTR<<" unable to close the file"<<std::endl;
	}

	int dimension;
	if (dims[0]>dims[1] && dims[0]>dims[2])
                dimension = dims[0];
        else if (dims[1]>dims[2])
                dimension = dims[1];
        else
                dimension = dims[2];

        /* Calcular dimension del árbol*/
        float aux = logf(dimension)/logf(2.0);
        float aux2 = aux - floorf(aux);
        int nLevels = aux2>0.0 ? aux+1 : aux;

	return nLevels; 
}

int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		std::cerr<<"Error, testVisualTur hdf5_file dataset_name octree_file [device]"<<std::endl;
		return 0;
	}

	int device = 0;
	if (argc > 4)
	{
		device = atoi(argv[4]);
	}

	int numT = 1;
	std::cout<<"Number of threads:"<<std::endl;
	std::cin >> numT;

	int W = 1024;
	int H = 1024;

	int nLevel = getnLevelFile(argv[1], argv[2]);

	cudaSetDevice(device);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	//get the amount of free memory on the graphics card  
    	size_t free;  
	size_t total;  
    	cudaMemGetInfo(&free, &total); 

	visualTurParams_device_t params;
	params.W = W;
	params.H = H;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 3*(free / (38*38*38*4)) /4;
	params.maxElementsCache_CPU = 5000;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 2;
	params.levelCubes = nLevel - 5;
	params.octreeLevel = (nLevel - 5) + 3;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	params.numThreads = numT;
	params.deviceID = device;
	params.startRay = 0;
	params.endRay = params.W*params.H*params.numRayPx;

	float * screenC = new float[H*W*4];
	float * screenG = 0;
	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*H*W*4))<<std::endl;

	visualTur_device * VisualTur = new visualTur_device(params,screenG); 

	VisualTur->camera_Move(make_float3(512,512,512));

	struct timeval st, end;

	double totalT = 0.0;
	for(int m=0; m<1000; m++)
	{
		
		gettimeofday(&st, NULL);
		VisualTur->updateVisibleCubes();
		std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*W*H*4, cudaMemcpyDeviceToHost))<<std::endl;
		VisualTur->camera_StrafeRight(0.5f);

		gettimeofday(&end, NULL);
		double delta = ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
		std::cout << "Time elapsed iteration "<<m<<": " << delta << " sec"<< std::endl;
		totalT+=delta;
	}
	
	std::cout << "Time elapsed: " << totalT<< " sec to 1000 iterations"<< std::endl;
	std::cout << "Average time elapsed: " << totalT/1000.0f<< " sec"<< std::endl;
	

	cudaFree(screenG);
	delete[] screenC;
	delete VisualTur;

}
