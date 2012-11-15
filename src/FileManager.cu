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
	hsize_t dim[3] = {abs(e.x-s.x),abs(e.y-s.y),abs(e.z-s.z),};

	// Container todo a 0's
	if (s.x >= (int)this->dims[0] || s.y >= (int)this->dims[1] || s.z >= (int)this->dims[2] || e.x < 0 || e.y < 0 || e.z < 0)
	{
		std::cerr<<"Warning: reading cube outsite the volume "<<std::endl;
		#if 1
		std::cerr<<"Dimension valume "<<this->dims[0]<<" "<<this->dims[1]<<" "<<this->dims[2]<<std::endl;
		std::cerr<<"start "<<s.x<<" "<<s.y<<" "<<s.z<<std::endl;
		std::cerr<<"end "<<e.x<<" "<<e.y<<" "<<e.z<<std::endl;
		std::cerr<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
		#endif
		for(unsigned int i=0; i<(dim[0]*dim[1]*dim[2]); i++)
			data[i] = 0.0;
		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {0,0,0};
	hsize_t offset[3] 	= {s.x < 0 ? 0 : s.x, s.y < 0 ? 0 : s.y, s.z < 0 ? 0 : s.z};
	hsize_t dimR[3]		= {e.x > (int)this->dims[0] ? this->dims[0] - offset[0] : e.x - offset[0],
				   e.y > (int)this->dims[1] ? this->dims[1] - offset[1] : e.y - offset[1],
				   e.z > (int)this->dims[2] ? this->dims[2] - offset[2] : e.z - offset[2]};

	float * aux = new float[dimR[0]*dimR[1]*dimR[2]];

	#if 0
	std::cout<<"--"<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	std::cout<<"--"<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	#endif

	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining hyperslab in the dataset"<<std::endl;
		return;
	}

	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining the memory dataspace"<<std::endl;
		return; 
	}


	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
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

	hsize_t dimS[3] = {s.x >= 0 ? 0 : abs(s.x), s.y >= 0 ? 0 : abs(s.y), s.z >= 0 ? 0 : abs(s.z)};
	hsize_t dimI[3] = {dimS[0]+dimR[0], dimS[1]+dimR[1], dimS[2]+dimR[2]};

	#if 0
	std::cout<<dimS[0]<<" "<<dimS[1]<<" "<<dimS[2]<<std::endl;
	std::cout<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	std::cout<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	#endif

	for(unsigned int i=0; i<dimS[0]; i++)
		for(unsigned int j=0; j<dimS[1]; j++)
			for(unsigned int k=0; k<dimS[2]; k++)
				data[k+j*dim[2]+i*dim[2]*dim[1]] = 0.0; 


	for(unsigned int i=dimS[0]; i<dimI[0]; i++)
		for(unsigned int j=dimS[1]; j<dimI[1]; j++)
			for(unsigned int k=dimS[2]; k<dimI[2]; k++)
				data[k+j*dim[2]+i*dim[2]*dim[1]] = aux[(k-dimS[0])+(j-dimS[1])*dimR[2]+(i-dimS[0])*dimR[1]*dimR[2]];
		
	for(unsigned int i=dimI[0]; i<dim[0]; i++)
		for(unsigned int j=dimI[1]; j<dim[0]; j++)
			for(unsigned int k=dimI[2]; k<dim[0]; k++)
				data[k+j*dim[2]+i*dim[2]*dim[1]] = 0.0; 
		
	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<WHERESTR<<" closing dataspace"<<std::endl;
		return;
	}


	delete[] aux;
}
