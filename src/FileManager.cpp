#include "FileManager.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <iostream>

#define WHERESTR  "Error[file "<<__FILE__<<", line "<<__LINE__<<"]: "
#define WHEREARG  __FILE__, __LINE__

FileManager::FileManager(const char * file_name, const char * dataset_name)
{
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
	
}

FileManager::~FileManager()
{
	herr_t      status;

	if ((status = H5Dclose(dataset_id)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to close the data set"<<std::endl;
		exit(0);
	}


	if ((status = H5Fclose(file_id)) < 0);
	{
		std::cerr<<WHERESTR<<" unable to close the file"<<std::endl;
		#if 0
		std::cerr<<"Cannot close file with open handles: " << 
		H5Fget_obj_count( file_id, H5F_OBJ_FILE )  	<<" file, " 	<<
		H5Fget_obj_count( file_id, H5F_OBJ_DATASET )  	<<" data, " 	<<
		H5Fget_obj_count( file_id, H5F_OBJ_GROUP )  	<<" group, "	<<
		H5Fget_obj_count( file_id, H5F_OBJ_DATATYPE )	<<" type, "	<<
		H5Fget_obj_count( file_id, H5F_OBJ_ATTR )	<<" attr"	<<std::endl;
		exit(0);
		#endif
		//return -1;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
	}
	
}

int FileManager::readHDF5(float * volume)
{

	herr_t      status;

	if ((status = H5Dread(dataset_id, H5T_IEEE_F32LE, spaceid/*H5S_ALL*/, H5S_ALL, H5P_DEFAULT, volume)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to read the file"<<std::endl;
		return 0;
	}


	return 0;		
}

void FileManager::readHDF5_Voxel_Array(int3 s, int3 e, float * data)
{
	herr_t	status;
	hid_t	memspace; 
	hsize_t	dim[3];
	hsize_t	dimD[3];
	hsize_t offset_out[3] 	= {0,0,0};
	hsize_t offset[3] 	= {s.x, s.y, s.z};

	dimD[0] = e.x - s.x;
	dimD[1] = e.y - s.y;
	dimD[2] = e.z - s.z;

	if (dimD[0] != dimD[1] || dimD[0] != dimD[2] )//|| dimD[0] < 0)
	{
		std::cerr<<"Error: No valid dimensions reading a voxel"<<std::endl;
		throw std::exception();
	}

	//std::cout<<"Real Size GigaVoxel "<<dimD[0]<<std::endl;

	dim[0] = e.x > this->dims[0] ? this->dims[0] - s.x : e.x - s.x;
	dim[1] = e.y > this->dims[1] ? this->dims[1] - s.y : e.y - s.y;
	dim[2] = e.z > this->dims[2] ? this->dims[2] - s.z : e.z - s.z;
	
	float * aux = new float[dim[0]*dim[0]*dim[0]];

	// Container todo a 0's
	if (s.x >= this->dims[0] || s.y >= this->dims[1] || s.z >= this->dims[2])
	{
		for(int i=0; i<(dimD[0]*dimD[0]*dimD[0]); i++)
			data[i] = 0.0;
		return;
	}

	//std::cout<<sx<<","<<sy<<","<<sz<<","<<ex<<","<<ey<<","<<ez<<std::endl;
	//std::cout<<sx<<","<<sy<<","<<sz<<","<<dim[0]<<","<<dim[1]<<","<<dim[2]<<std::endl;

	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(spaceid, H5S_SELECT_SET, offset, NULL, dim, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining hyperslab in the dataset"<<std::endl;
		return;
	}

	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining the memory dataspace"<<std::endl;
		return; 
	}


	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dim, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining the memory hyperslab"<<std::endl;
		return;
	}

	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(dataset_id, H5T_IEEE_F32LE, memspace, spaceid, H5P_DEFAULT, aux)) < 0)
	{
		std::cerr<<WHERESTR<<" reading data from hyperslab un the file"<<std::endl;
		return;
	}


	for(unsigned int i=0; i<dim[0]; i++)
		for(unsigned int j=0; j<dim[1]; j++)
			for(unsigned int k=0; k<dim[2]; k++)
				data[k+j*dimD[0]+i*dimD[0]*dimD[0]] = aux[k+j*dim[2]+i*dim[1]*dim[2]];
		
	for(unsigned int i=dim[0]; i<dimD[0]; i++)
		for(unsigned int j=dim[1]; j<dimD[0]; j++)
			for(unsigned int k=dim[2]; k<dimD[0]; k++)
				data[k+j*dimD[0]+i*dimD[0]*dimD[0]] = 0.0; 
		
	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<WHERESTR<<" closing dataspace"<<std::endl;
		return;
	}


	delete[] aux;
}
