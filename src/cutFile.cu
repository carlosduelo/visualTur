#include "FileManager.hpp"
#include <stdexcept>
#include <iostream>
#include <stdlib.h>

int main(int argc, char ** argv)
{
	if (argc < 8)
	{
	std::cerr<<"reduceVolume hdf5_name data_set out_name X Y Z dim"<<std::endl;
	return -1;
	}
	FileManager 	fileManager(argv[1],argv[2]);
	int  x = atoi(argv[4]);
	int  y = atoi(argv[5]);
	int  z = atoi(argv[6]);
	int  dim = atoi(argv[7]);

	float * data = new float[dim*dim*dim];
	fileManager.readHDF5_Voxel_Array(make_int3(x,y,z), make_int3(x+dim,y+dim,z+dim), data);
	fileManager.CreateFile(data, argv[3], (char*)"value", dim, dim, dim);
	delete[] data;
	return 0;
}
