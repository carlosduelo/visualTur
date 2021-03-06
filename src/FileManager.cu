#include "FileManager.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <iostream>

#define WHERESTR  "Error[file "<<__FILE__<<", line "<<__LINE__<<"]: "

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
	hsize_t dim[3] = {abs(e.x-s.x),abs(e.y-s.y),abs(e.z-s.z)};

	// Set zeros's
	bzero(data, dim[0]*dim[1]*dim[2]*sizeof(float));

	// The data required is completly outside of the dataset
	if (s.x >= (int)this->dims[0] || s.y >= (int)this->dims[1] || s.z >= (int)this->dims[2] || e.x < 0 || e.y < 0 || e.z < 0)
	{
		std::cerr<<"Warning: reading cube outsite the volume "<<std::endl;
		#if 1
		std::cerr<<"Dimension valume "<<this->dims[0]<<" "<<this->dims[1]<<" "<<this->dims[2]<<std::endl;
		std::cerr<<"start "<<s.x<<" "<<s.y<<" "<<s.z<<std::endl;
		std::cerr<<"end "<<e.x<<" "<<e.y<<" "<<e.z<<std::endl;
		std::cerr<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
		#endif
		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {s.x < 0 ? abs(s.x) : 0, s.y < 0 ? abs(s.y) : 0, s.z < 0 ? abs(s.z) : 0};
	hsize_t offset[3] 	= {s.x < 0 ? 0 : s.x, s.y < 0 ? 0 : s.y, s.z < 0 ? 0 : s.z};
	hsize_t dimR[3]		= {e.x > (int)this->dims[0] ? this->dims[0] - offset[0] : e.x - offset[0],
				   e.y > (int)this->dims[1] ? this->dims[1] - offset[1] : e.y - offset[1],
				   e.z > (int)this->dims[2] ? this->dims[2] - offset[2] : e.z - offset[2]};

	#if 0
	std::cout<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	std::cout<<"Dimension hyperSlab "<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	std::cout<<"Offset in "<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	std::cout<<"Offset out "<<offset_out[0]<<" "<<offset_out[1]<<" "<<offset_out[2]<<std::endl;
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
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" defining the memory space"<<std::endl;
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
	if ((status = H5Dread(dataset_id, H5T_IEEE_F32LE, memspace, spaceid, H5P_DEFAULT, data)) < 0)
	{
		std::cerr<<WHERESTR<<" reading data from hyperslab un the file"<<std::endl;
		return;
	}


	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<WHERESTR<<" closing dataspace"<<std::endl;
		return;
	}
}

int FileManager::CreateFile(float * volume, char * newname, char * dataset_name, int x, int y, int z)
{
	hid_t       file, dataset;         /* file and dataset handles */
	hid_t       datatype, dataspace;   /* handles */
	hsize_t     dimsf[3];              /* dataset dimensions */
	herr_t      status;

	if ((file = H5Fcreate(newname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to create the HDF5 file"<<std::endl;
		return -1;
	}

	dimsf[0] = (hsize_t)x;
	dimsf[1] = (hsize_t)y;
	dimsf[2] = (hsize_t)z;

	fprintf(stdout, "New data set size (%d,%d,%d)\n",x,y,z);

	if ((dataspace = H5Screate_simple(3, dimsf, NULL)) < 0)
	{
		std::cerr<<WHERESTR" unable to create the data space"<<std::endl;
		exit(0);
	}

	if ((datatype = H5Tcopy(H5T_IEEE_F32LE)) < 0)
	{
		std::cerr<<WHERESTR" unable to create the data type"<<std::endl;
		exit(0);
	}

	if ((status = H5Tset_order(datatype, H5T_ORDER_LE)) < 0)
	{
		std::cerr<<WHERESTR" unable to set order"<<std::endl;
		exit(0);
	}

	if (( dataset = H5Dcreate2(file, dataset_name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
	{
		std::cerr<<WHERESTR" unable to create the data set"<<std::endl;
		exit(0);
	}

	if ((status = H5Dwrite(dataset, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume)) < 0)
	{
		std::cerr<<WHERESTR" unable to write data set"<<std::endl;
		exit(0);
	}

	if (H5Sclose(dataspace) < 0)
	{
		std::cerr<<WHERESTR" unable to close the dataspace"<<std::endl;
		exit(0);
	}

	if (H5Tclose(datatype) < 0)
	{
		std::cerr<<WHERESTR" unable to close the datatype"<<std::endl;
		exit(0);
	}

	if (H5Dclose(dataset) < 0)
	{
		std::cerr<<WHERESTR" unable to close the data set"<<std::endl;
		exit(0);
	}

	if (H5Fclose(file) < 0)
	{
		std::cerr<<WHERESTR" unable to close the file"<<std::endl;
		exit(0);
	}

	return 0;
}
