/*
 * FileManager
 *
 */

#ifndef _FILE_MANAGER_H_
#define _FILE_MANAGER_H_

#include "config.hpp"
#include "hdf5.h"

class FileManager
{	
	private:
		hid_t           file_id;
		hid_t           dataset_id;
		hid_t           spaceid;
		int             ndim;

	public:
		hsize_t         dims[3];

		FileManager(const char * file_name, const char * dataset_name);

		~FileManager();

		int readHDF5(float * volume);

		/*
		 * Lee del finchero una parte 
		 */
		void readHDF5_Voxel_Array(int3 s, int3 e, float * data);

		int CreateFile(float * volume, char * newname, char * dataset_name, int x, int y, int z);
};

#endif/*_FILE_MANAGER_H_*/
