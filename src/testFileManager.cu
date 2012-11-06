#include "FileManager.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>



int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testFileManger hdf5_file dataset_name"<<std::endl;
		return 0;
	}

	FileManager * fileManager = new FileManager(argv[1], argv[2]);

	int  dim = 32;
	int3 cero = make_int3(0,0,0);
	int3 dimI = make_int3(dim,dim,dim); 
	int offset = 10;
	int3 offsetI = make_int3(offset, offset, offset);
	int3 dimM = make_int3(fileManager->dims[0], fileManager->dims[1], fileManager->dims[2]); 
	int3 start2 = dimM - dimI;

	float * data1 = new float[dim*dim*dim];
	float * data2 = new float[dim*dim*dim];

	fileManager->readHDF5_Voxel_Array(cero, dimI, data1);
	fileManager->readHDF5_Voxel_Array(cero-offsetI,dimI-offsetI, data2);
	for(unsigned int i=0; i<dim-offset; i++)
		for(unsigned int j=0; j<dim-offset; j++)
			for(unsigned int k=0; k<dim-offset; k++)
			{
				if (data1[k+j*dim+i*dim*dim] != data2[(k+offset)+(j+offset)*dim+(i+offset)*dim*dim])
					std::cout<<"Error"<<std::endl;
			}

	fileManager->readHDF5_Voxel_Array(start2, dimM, data1);
	fileManager->readHDF5_Voxel_Array(start2+offsetI,dimM+offsetI, data2);
	for(unsigned int i=0; i<dim-offset; i++)
		for(unsigned int j=0; j<dim-offset; j++)
			for(unsigned int k=0; k<dim-offset; k++)
			{
				if (data1[(k+offset)+(j+offset)*dim+(i+offset)*dim*dim] != data2[k+j*dim+i*dim*dim])
					std::cout<<"Error"<<std::endl;
			}
	delete[] data1;
	delete[] data2;
	delete fileManager;
}
