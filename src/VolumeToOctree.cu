
#include "CreateOctree.hpp"
#include <iostream>
#include <fstream>


int main(int argc, char ** argv)
{
	if (argc != 3)
	{
		std::cout<<"Error, volumeToOctree hdf5_file file_output"<<std::endl;
		return -1;
	}

	float iso;
	std::cout<<"******** CREATING OCTREE *******"<<std::endl;

	std::cout<<"Give the threshold"<<std::endl;
	std::cin>>iso;

	OctreeM * o = new OctreeM(argv[1], iso);

	std::cout<<"Writing to a file....."<<std::endl;

	o->writeToFile(argv[2]);

	delete o;

	std::cout<<"Finish, test OK!"<<std::endl;

	std::cout<<"******** OCTREE CREATED ********"<<std::endl;

	return 0;
}
