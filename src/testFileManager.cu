#include "FileManager.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>


int checkValues(float * dat1, int dim1, float *dat2, int dim2, int3 minBox, int3 maxBox)
{
	for(int i=0;i<abs(minBox.x);i++)
		for(int j=0;j<abs(minBox.y);j++)
			for(int k=0;k<abs(minBox.z);k++)
				if (dat2[k+j*dim2+i*dim2*dim2] != 0.0)
				{
					std::cout<<"Error test"<<std::endl;
					std::cout<<"Position ("<<i<<","<<j<<","<<k<<") not 0.0"<<std::endl;
					return 0;
				}

	for(int i=abs(minBox.x);i<maxBox.x;i++)
		for(int j=abs(minBox.y);j<maxBox.y;j++)
			for(int k=abs(minBox.z);k<maxBox.z;k++)
			{
				if (dat2[k+j*dim2+i*dim2*dim2] != dat1[(k+minBox.z)+(j+minBox.y)*dim1+(i+minBox.x)*dim1*dim1])
				{
					std::cout<<"Error test"<<std::endl;
					std::cout<<"Elements not equal ";
					std::cout<<"data1("<<(i+minBox.x)<<","<<(j+minBox.y)<<","<<(k+minBox.z)<<")="<<dat1[(k+minBox.z)+(j+minBox.y)*dim1+(i+minBox.x)*dim1*dim1]<<" ";
					std::cout<<"data2("<<i<<","<<j<<","<<k<<")="<<dat2[k+j*dim2+i*dim2*dim2]<<std::endl;
					exit(0);
					return 0;
				}
			}

	return 0;
}


int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testFileManger hdf5_file dataset_name"<<std::endl;
		return 0;
	}

	FileManager * fileManager = new FileManager(argv[1], argv[2]);

	int cubeInc = 2;
	int  dim = 32;
	int  dim2 = dim +2*cubeInc;
	
	float * data1 = new float[dim2*dim2*dim2];
	float * data2 = new float[dim2*dim2*dim2];

	// NORMAL READ just try it
	int3 minBox = make_int3(0,0,0);
	int3 maxBox = make_int3(dim2,dim2,dim2);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, data1);
	std::cout<<"Cube minBox ("<<minBox.x<<","<<minBox.y<<","<<minBox.z<<") maxBox ("<<maxBox.x<<","<<maxBox.y<<","<<maxBox.z<<") Read "<<std::endl;

	// TEST 1
	std::cout<<"TEST 1"<<std::endl;
	// READ negatives
	minBox = make_int3(-2*cubeInc, -2*cubeInc, -2*cubeInc);
	maxBox = minBox + make_int3(dim2,dim2,dim2);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, data2);
	std::cout<<"Cube minBox ("<<minBox.x<<","<<minBox.y<<","<<minBox.z<<") maxBox ("<<maxBox.x<<","<<maxBox.y<<","<<maxBox.z<<") Read "<<std::endl;

	checkValues(data1, dim2, data2, dim2, minBox, maxBox);

	// TEST 2
	std::cout<<"TEST 2"<<std::endl;
	// READ ONE NEGATIVE
	minBox = make_int3(-2*cubeInc, 0, 0);
	maxBox = minBox + make_int3(dim2,dim2,dim2);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, data2);
	std::cout<<"Cube minBox ("<<minBox.x<<","<<minBox.y<<","<<minBox.z<<") maxBox ("<<maxBox.x<<","<<maxBox.y<<","<<maxBox.z<<") Read "<<std::endl;

	checkValues(data1, dim2, data2, dim2, minBox, maxBox);

	// TEST 3
	std::cout<<"TEST 3"<<std::endl;
	// READ ONE NEGATIVE
	minBox = make_int3(0, -2*cubeInc, 0);
	maxBox = minBox + make_int3(dim2,dim2,dim2);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, data2);
	std::cout<<"Cube minBox ("<<minBox.x<<","<<minBox.y<<","<<minBox.z<<") maxBox ("<<maxBox.x<<","<<maxBox.y<<","<<maxBox.z<<") Read "<<std::endl;

	checkValues(data1, dim2, data2, dim2, minBox, maxBox);

	// TEST 4
	std::cout<<"TEST 3"<<std::endl;
	// READ ONE NEGATIVE
	minBox = make_int3(0, 0, 0);
	maxBox = minBox + make_int3(dim2,dim2,dim2);
	fileManager->readHDF5_Voxel_Array(minBox, maxBox, data2);
	std::cout<<"Cube minBox ("<<minBox.x<<","<<minBox.y<<","<<minBox.z<<") maxBox ("<<maxBox.x<<","<<maxBox.y<<","<<maxBox.z<<") Read "<<std::endl;

	checkValues(data1, dim2, data2, dim2, minBox, maxBox);
	delete fileManager;
	delete[] data1;
	delete[] data2;
}
