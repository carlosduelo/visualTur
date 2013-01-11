#include "CreateOctree.hpp"
#include "FileManager.hpp"
#include "mortonCodeUtil.hpp"
#include "cutil_math.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

/*
 **********************************************************************************************
 ****** OctreeLevel ***************************************************************************
 **********************************************************************************************
 */

OctreeLevel::OctreeLevel()
{
	elements = 0;
	status	= false;
	nextElement = 0;
}

OctreeLevel::~OctreeLevel()
{
	delete elements;
}

int OctreeLevel::getSize()
{
	if (elements!=0)
		return elements->size();
	else
		return 0;
}

void OctreeLevel::printLevel()
{
	std::vector<index_node_t>::iterator it;

	for ( it=elements->begin() ; it != elements->end(); it++ )
	{
		std::cout << "From " << *it;
		it++;
		std::cout << " to " << *it<<std::endl;
	}
	std::cout<<std::endl;
}

std::vector<index_node_t>::iterator OctreeLevel::getBegin()
{
	if (elements != 0)
		return elements->begin();

	std::cout<<"Error level not created"<<std::endl;
	std::exception();
	return elements->begin();
}

std::vector<index_node_t>::iterator OctreeLevel::getEnd()
{
	if (elements != 0)
		return elements->end();

	std::cout<<"Error level not created"<<std::endl;
	std::exception();
	return elements->begin();
}


void OctreeLevel::fill(index_node_t index)
{
	if (elements == 0)
		elements = new std::vector<index_node_t>();

	elements->push_back(index);
	status = true;
}

bool OctreeLevel::insert(index_node_t index)
{
	if (index < nextElement)
	{
		std::cout<<"Error: inserting a element smaller than the last"<<std::endl;
		std::exception();
		return false;
	}
	if (elements == 0)
	{
		elements = new std::vector<index_node_t>();
		elements->push_back(index);
		status = false;
		nextElement = index + 1;
		//std::cout<<"creando elementos meto el inidece "<<index<<"siguiente es " <<nextElement<<std::endl;
		//std::cout<<index<<std::endl;
		return false;
	}
	// Es decir va seguido
	else if (nextElement == index)
	{
		nextElement++;
		//std::cout<<"contiguo "<<index<<" siguiente sera"<<nextElement<<std::endl;
		return true;
	}
	else
	{
		//std::cout<<"meto "<<nextElement-1<<" y "<<index<<"siguiente ";
		elements->push_back(nextElement-1);
		//std::cout<<nextElement-1<<std::endl;
		//std::cout<<index<<std::endl;
		elements->push_back(index);
		status = false;
		nextElement = index + 1;
		//std::cout<<nextElement<<std::endl;
		return false;
	}
}

void OctreeLevel::completeLevel()
{
	if (elements == 0)
		status = false;
	else
	{
		std::cout<<"Para completar meto el ultimo "<<nextElement-1<<std::endl;
		elements->push_back(nextElement-1);
		status = true;
	}
}

inline bool OctreeLevel::checkRange(index_node_t index, int min, int max)
{
	if (
		(elements->at(min) < index && elements->at(max) > index) || 
		(index == elements->at(min) || index == elements->at(max)))
			return true;
	else
			return false;
	
}

bool OctreeLevel::binary_search(index_node_t index, int min, int max)
{
	int diff = max-min;
	if (diff < 1)
		return false;
	else if (diff == 1)
		return checkRange(index, min, max);
	else
	{
		unsigned int middle = min + (diff / 2);
		if (middle % 2 == 1) middle--;

		if (checkRange(index, middle, middle+1))
			return true;
		else if (index < elements->at(middle))
		{
			max = middle-1;
			return binary_search(index, min, max);
		}
		else if (index > elements->at(middle+1))
		{
			min = middle + 2;
			return binary_search(index, min, max);
		}
		else
			std::cout<<"Errro"<<std::endl;
	}

	return false;
}

bool OctreeLevel::search(index_node_t index)
{
	if (status)
	{
		return binary_search(index, 0, elements->size()-1);
	}

	return false;
}

int OctreeLevel::binary_search_closer(index_node_t index, int min, int max)
{
	int diff = max-min;
	unsigned int middle = min + (diff / 2);
	#if 0 
	if (diff < 1)
	{
	//	std::cout<<"cacaa "<<max<<" "<<min<<" "<<middle<<std::endl;
		return -1;
	}
	else if (diff == 1)
		return middle;
	#endif
	if (diff <= 1)
	{
		if (middle % 2 == 1) middle--;
		return middle;
	}
	else
	{
		if (middle % 2 == 1) middle--;

		if (checkRange(index, middle, middle+1))
			return middle;
		else if (index < elements->at(middle))
		{
			max = middle-1;
			return binary_search_closer(index, min, max);
		}
		else if (index > elements->at(middle+1))
		{
			min = middle + 2;
			return binary_search_closer(index, min, max);
		}
		else
			std::cout<<"Errro"<<std::endl;
	}

	return false;
}

inline bool OctreeLevel::searchSecuential(index_node_t index, int min, int max, bool up)
{
	if (up)
		for(int i=min; i<max; i+=2)
		{
			//std::cout<<i<<std::endl;
			if (checkRange(index, i, i+1))
				return true;
			if (index < elements->at(i+1))
				break;
		}
	else
		for(int i=max; i>min; i-=2)
		{
			if (checkRange(index, i, i+1))
				return true;
			if (index < elements->at(i+1))
				break;
		}
		
	return false;
}

/* Dado un index de nodo, devuelve los hijos que están en el nivel */
int OctreeLevel::searchChildren(index_node_t father, index_node_t * children)
{
	if (!status)
		return false;

	index_node_t childrenID = father << 3;
	int numChild = 0;

	if (elements->size()==2)
	{
		for(int i=0; i<8; i++)
		{
			if (checkRange(childrenID,0,1))
				children[numChild++] = childrenID;
			childrenID++;
		}

		return numChild;
	}

	unsigned int closer1 = binary_search_closer(childrenID,   0, elements->size());
	unsigned int closer8 = binary_search_closer(childrenID+7, closer1, elements->size()) + 1;

	if (closer8 >= elements->size())
		closer8 = elements->size()-1;

	for(int i=0; i<8; i++)
	{
		if (searchSecuential(childrenID, closer1, closer8, true))
		{
			children[numChild] = childrenID;
			numChild++;
		}
		childrenID++;
	}

	return numChild;
}

/*
 **********************************************************************************************
 ****** Octree ********************************************************************************
 **********************************************************************************************
 */

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

#define MAX_DIM_BRICK_FOR_READING 512
//#define BRICK_DIM 32
//#define BRICK_EXP 5
#define BRICK_DIM 1
#define BRICK_EXP 0

OctreeM::OctreeM(const char * file_name, float iso)
{
	FileManager * fm = new FileManager(file_name, "value");
	realDim[0] = fm->dims[0];
	realDim[1] = fm->dims[1];
	realDim[2] = fm->dims[2];
	delete fm;
	
	if (realDim[0]>realDim[1] && realDim[0]>realDim[2])
		dimension = realDim[0];
	else if (realDim[1]>realDim[2])
		dimension = realDim[1];
	else
		dimension = realDim[2];

	/* Calcular dimension del árbol*/
	float aux = logf(dimension)/logf(2.0);
	float aux2 = aux - floorf(aux);
	nLevels = aux2>0.0 ? aux+1 : aux;
	dimension = pow(2,nLevels);

	std::cout<<"Octree de dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" niveles "<<nLevels<<std::endl;

	isosurface = iso;

	octree = new OctreeLevel[nLevels-BRICK_EXP+1]();

	/* Create last level */
	for(int i=nLevels-BRICK_EXP; i>=0; i--)
	{
		createTree(file_name, i);
		std::cout<<"Numero de elementos de nivel "<<i<<" "<<octree[i].getSize()<<std::endl;

	}
}


OctreeM::~OctreeM()
{
	delete[] octree;
}

void OctreeM::writeToFile(const char * file_name)
{
	/* Save isosurface, real dimension, nLevels, octree */
	std::ofstream file(file_name, std::ofstream::binary);
	
	int magicWord = 919278872;

	file.write((char*)&magicWord,  sizeof(magicWord));
	file.write((char*)&isosurface, sizeof(isosurface));
	file.write((char*)&dimension, sizeof(dimension));
	file.write((char*)&realDim[0], sizeof(realDim[0]));
	file.write((char*)&realDim[1], sizeof(realDim[1]));
	file.write((char*)&realDim[2], sizeof(realDim[2]));
	file.write((char*)&nLevels, sizeof(nLevels));

	for(int i=nLevels-BRICK_EXP; i>=0; i--)
	{
		int numElem = octree[i].getSize();
		file.write((char*)&numElem, sizeof(numElem));
		std::vector<index_node_t>::iterator it;
		for(it=octree[i].getBegin() ; it != octree[i].getEnd(); it++)
		{
			index_node_t node = *it;
			file.write((char*) &node, sizeof(index_node_t));
		}
			
	}

	file.close();	
}

int 	OctreeM::getnLevels()
{
	return nLevels;
}

int 	OctreeM::getrealLevels()
{
	return nLevels-BRICK_EXP+1;
}

int *	OctreeM::getRealDimension()
{
	return realDim;
}

float OctreeM::getIsosurface()
{
	return isosurface;
}

void OctreeM::printLevel(int level)
{
	std::cout<<"Level "<< level <<std::endl;
	std::vector<index_node_t>::iterator it;

	for ( it=octree[level].getBegin() ; it != octree[level].getEnd(); it++ )
	{
	        std::cout << "From " << *it;
		it++;
	        std::cout << " to " << *it<<std::endl;
	}
	std::cout<<std::endl;
}

bool OctreeM::getValidinLevel(int x, int y, int z, int level)
{
	if (level < 0 || x < 0 || x > realDim[0] || y < 0 || y > realDim[1] || z < 0 || z > realDim[2])
		return false;
	
	if (nLevels-level < BRICK_EXP)
		level=BRICK_EXP-1;

	return octree[level].search(coordinateToIndex(make_int3(x,y,z),level,nLevels));
}

bool OctreeM::getValidNode(index_node_t index)
{
	int level = getIndexLevel(index); 
	return octree[level].search(index);
}


void OctreeM::createTree(const char * file_name, int level)
{
	int dim = powf(2.0,nLevels-level);
	std::cout<<"Dimension del node en el nivel "<<level<<" es de "<<dim<<std::endl;

	if (nLevels-level==BRICK_EXP)
	{
		FileManager * fileManager = new FileManager(file_name, "value");

		int brick_dim_reading = MAX_DIM_BRICK_FOR_READING; 
		while (dimension < brick_dim_reading)
			brick_dim_reading >>= 1;

		float voxel[8];

		index_node_t 	inicio 		= coordinateToIndex(make_int3(0,0,0),nLevels,nLevels);
		index_node_t 	fin 		= coordinateToIndex(make_int3(dimension-1,dimension-1,dimension-1),nLevels,nLevels);
		long long int 	numero 		= 0;
		long long int 	contiguo 	= 0;
		int3 startBox, endBox;

		int dim  = brick_dim_reading+1;
		float * container = new float[dim*dim*dim];
		long int total = fin - inicio;
		long int count = 0;
		std::cout<<"0 %";

		while(inicio<=fin)
		{
			startBox = getMinBoxIndex2(inicio, nLevels, nLevels);

			index_node_t interFin = inicio + brick_dim_reading * brick_dim_reading * brick_dim_reading - 1;

			endBox = getMinBoxIndex2(interFin, nLevels, nLevels);

			fileManager->readHDF5_Voxel_Array(startBox, endBox + make_int3(2,2,2), container);

			int3 		current 	= startBox;
			index_node_t 	anterior 	= inicio;

			int3 		cArray;

			while(inicio<=interFin)
			{
				current 	= updateCoordinates(nLevels, nLevels, anterior, nLevels, inicio, current);
				anterior 	= inicio;

				cArray 		= current - startBox;
				voxel[0] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(0,0,1);
				voxel[1] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(0,1,0);
				voxel[2] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(0,1,1);
				voxel[3] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(1,0,0);
				voxel[4] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(1,0,1);
				voxel[5] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(1,1,0);
				voxel[6] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
				cArray 		= current - startBox + make_int3(1,1,1);
				voxel[7] 	= container[posToIndex(cArray.x,cArray.y, cArray.z, dim)];
			#if 0	
				voxel[0] = container->getElement(x,y,z);
				voxel[1] = container->getElement(x,y,z+1);
				voxel[2] = container->getElement(x,y+1,z);
				voxel[3] = container->getElement(x,y+1,z+1);
				voxel[4] = container->getElement(x+1,y,z);
				voxel[5] = container->getElement(x+1,y,z+1);
				voxel[6] = container->getElement(x+1,y+1,z);
				voxel[7] = container->getElement(x+1,y+1,z+1);
			#endif
				bool has = false; // XXX Quitar de la comprobacion del bucle y anadir break
				bool sign = (voxel[0] - isosurface) < 0;
				for (int i = 1; i < 8 && !has; ++i)
					if (((voxel[i] - isosurface) < 0) != sign)
						has = true;

				if (has)
				{
					numero++;
					//Generar código y meter!!
				//	if (octree[level].insert(coordinateToIndex(make_int3(x,y,z),level,nLevels)))
					if (octree[level].insert(inicio))
						contiguo++;
				}
				inicio++;
				count++;
				if(count % 1000 == 0)
				{
					std::cout<<'\r';
					std::cout<<(count*100)/total<<" %";
				}
			}

			std::cout<<'\r';
			std::cout<<(count*100)/total<<" %";
		}

		std::cout<<"Level complete"<<std::endl;
		octree[level].completeLevel();

		delete fileManager;
		delete[] container;

		std::cout<<numero<< " " << contiguo<<" "<<std::endl;
	}
	else if (level==0)
	{
		index_node_t children[8];
		if (octree[1].searchChildren(1, children) > 0)
		{
			octree[0].insert(1);
			octree[0].completeLevel();
		}
	}
	else
	{
		index_node_t 	inicio 		= coordinateToIndex(make_int3(0,0,0),level,nLevels);
		index_node_t 	fin 		= coordinateToIndex(make_int3(dimension-1,dimension-1,dimension-1),level,nLevels);
		long long int 	numero 		= 0;
		long long int 	contiguo 	= 0;
		long int 	total 		= fin - inicio;
		long int count = 0;
		std::cout<<"0 %";
		while(inicio!=fin)
		{
			index_node_t children[8];
			if (octree[level+1].searchChildren(inicio, children) > 0)
			{
				numero++;
				//Generar código y meter!!
				if (octree[level].insert(inicio))
					contiguo++;
			}
			inicio++;
			count++;
			if(count % 1000 == 0)
			{
				std::cout<<'\r';
				std::cout<<(count*100)/total<<" %";
			}
		}
		std::cout<<"Level complete"<<std::endl;
		octree[level].completeLevel();
		std::cout<<numero<< " " << contiguo<<" "<<std::endl;
	}

}
