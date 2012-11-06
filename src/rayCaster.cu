#include "rayCaster.hpp"
#include "mortonCodeUtil.hpp"
#include "cuda_help.hpp"
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iostream>
#include <fstream>

#define posToIndex(i,j,k,d) ((k)+(j)*d+(i)*d*d)


inline __device__ bool _cuda_RayAABB(float3 origin, float3 dir,  float * tnear, float * tfar, int3 minBox, int3 maxBox)
{
	bool hit = true;

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1 / dir.x;
	if (divx >= 0)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1 / dir.y;
	if (divy >= 0)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
	{
		hit = false;
	}

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1 / dir.z;
	if (divz >= 0)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
	{
		hit = false;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0)
	 	*tnear=0.0;
	else
		*tnear=tmin;
	*tfar=tmax;

	return hit;

}

inline __device__ float getElement(int x, int y, int z, float * data, int3 dim)
{
	return data[posToIndex(x,y,z,dim.x)]; 
}

__device__ float getElementInterpolate(float3 pos, float * data, int3 minBox, int3 dim)
{
	float3 posR = make_float3(pos.x - minBox.x, pos.y -minBox.y, pos.z - minBox.z);

	int x0 = posR.x < 0.0f ? 0 : (posR.x>=dim.x-1 ? dim.x-2 : posR.x);
	int y0 = posR.y < 0.0f ? 0 : (posR.y>=dim.y-1 ? dim.y-2 : posR.y);
	int z0 = posR.z < 0.0f ? 0 : (posR.z>=dim.z-1 ? dim.z-2 : posR.z);
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	int z1 = z0 + 1;

	float p000 = getElement(x0,y0,z1,data,dim);
	float p001 = getElement(x0,y1,z1,data,dim);
	float p010 = getElement(x0,y0,z0,data,dim);
	float p011 = getElement(x0,y1,z0,data,dim);
	float p100 = getElement(x1,y0,z1,data,dim);
	float p101 = getElement(x1,y1,z1,data,dim);
	float p110 = getElement(x1,y0,z0,data,dim);
	float p111 = getElement(x1,y1,z0,data,dim);

	float3 pi = make_float3(posR.x-(float)x0, posR.y-(float)y0, posR.z-(float)z0);

	return  p000 * (1.0-pi.x) * (1.0-pi.y) * (1.0-pi.z) +       \
		p001 * (1.0-pi.x) * (1.0-pi.y) * pi.z       +       \
		p010 * (1.0-pi.x) * pi.y       * (1.0-pi.z) +       \
		p011 * (1.0-pi.x) * pi.y       * pi.z       +       \
		p100 * pi.x       * (1.0-pi.y) * (1.0-pi.z) +       \
		p101 * pi.x       * (1.0-pi.y) * pi.z       +       \
		p110 * pi.x       * pi.y       * (1.0-pi.z) +       \
		p111 * pi.x       * pi.y       * pi.z;
}

inline __device__ float3 getNormal(float3 pos, float * data, int3 minBox, int3 maxBox)
{
	return make_float3(	(getElementInterpolate(make_float3(pos.x-1.0f,pos.y,pos.z),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x+1.0f,pos.y,pos.z),data,minBox,maxBox))        /2.0f,
				(getElementInterpolate(make_float3(pos.x,pos.y-1.0f,pos.z),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x,pos.y+1.0f,pos.z),data,minBox,maxBox))        /2.0f,
			        (getElementInterpolate(make_float3(pos.x,pos.y,pos.z-1.0f),data,minBox,maxBox) - getElementInterpolate(make_float3(pos.x,pos.y,pos.z+1.0f),data,minBox,maxBox))        /2.0f);
}			

__global__ void cuda_rayCaster(int W, int H, float3 ligth, float3 origin, float3 * rays, float iso, visibleCube_t * cube, int3 dimCube, int3 cubeInc, int level, int nLevel, float * screen)
{
	unsigned int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (cube[id].id != 0 && id < (W*H))
	{
		float tnear;
		float tfar;
		// To do test intersection real cube position
		int3 minBox = getMinBoxIndex2(cube[id].id, level, nLevel);
		int3 maxBox = minBox + dimCube;

		if  (_cuda_RayAABB(origin, rays[id],  &tnear, &tfar, minBox, maxBox))
		{
			// To ray caster is needed bigger cube, so add cube inc
			minBox -= cubeInc;
			maxBox = dimCube + 2*cubeInc;
			//float * cubeD = cube[id].data;
			float3 Xnear = origin + tnear * rays[id];
			float3 Xfar  = Xnear;
			float3 Xnew  = Xnear;
			bool 				primera 	= true;
			float 				ant		= 0.0;
			float				sig		= 0.0;
			float steps = 0;
			
			bool hit = false;

			while(steps < 500)
			{
				if (primera)
				{
					primera = false;
					ant = getElementInterpolate(Xnear, cube[id].data, minBox, maxBox);
					Xfar = Xnear;
				}
				else
				{
					sig = getElementInterpolate(Xnear, cube[id].data, minBox, maxBox);
					if (( ((iso-ant)<0.0) && ((iso-sig)<0.0)) || ( ((iso-ant)>0.0) && ((iso-sig)>0.0)))
					{
						ant = sig;
						Xfar=Xnear;
					}
					else
					{
						#if 0
						// Intersection Refinament using an iterative bisection procedure
						float valueE = 0.0;
						for(int k = 0; k<5;k++) // XXX Cuanto más grande mejor debería ser el renderizado
						{
							Xnew = (Xfar - Xnear)*((iso-sig)/(ant-sig))+Xnear;
							valueE = getElementInterpolate(Xnew, cube[id].data, minBox, maxBox);
							if (valueE>iso)
								Xnear=Xnew;
							else
								Xfar=Xnew;
						}
						#endif
						Xnew = Xnear;
						steps = 500;
						hit = true;
					}
					
				}

				Xnear += (0.1 * rays[id]);
				steps++;
			}

			if (hit)
			{
				#if 0	
				float3 n = getNormal(Xnew, cube[id].data, minBox,  maxBox);
				float3 l = Xnew - ligth;
				normalize(Xnew);	
				float3 v = Xnew - origin;
				normalize(v);
				float nl		= n.x * l.x + n.y * l.y + n.z * l.z;	
				float lambertTerm  	= fabsf(nl);
				float3 r = l - 2.0 * n * nl;
				//float rv		= r.x*rays[id].x + r.y * rays[id].y + r.z * rays[id].z; 
				float rv		= r.x*v.x + r.y*v.y + r.z*v.z; 
			
				#endif
				screen[id*4]   = ((Xnew.x/maxBox.x ));// * lambertTerm) + (0.2 * powf(rv,8.0));
				screen[id*4+1] = ((Xnew.y/maxBox.y ));// * lambertTerm) + (0.2 * powf(rv,8.0));
				screen[id*4+2] = ((Xnew.z/maxBox.z ));// * lambertTerm) + (0.1 * powf(rv,8.0));
				screen[id*4+3] = 1.0f;
			}
			else
			{
				screen[id*4] = 0.0f; 
				screen[id*4+1] = 0.0f; 
				screen[id*4+2] = 0.0f; 
				screen[id*4+3] = 1.0f; 
			}
			
		}
		else
		{
			screen[id*4] = 0.0f; 
			screen[id*4+1] = 0.0f; 
			screen[id*4+2] = 0.0f; 
			screen[id*4+3] = 1.0f; 
		}
	}
}

/*
******************************************************************************************************
************ rayCaster methods ***************************************************************************
******************************************************************************************************
*/

rayCaster::rayCaster(float isosurface, float3 lposition)
{
	iso    = isosurface;
	lightPosition = lposition;
}

rayCaster::~rayCaster()
{
}

void rayCaster::render(Camera * camera, int level, int nLevel, visibleCube_t * cube, int3 cubeDim, int3 cubeInc, float * buffer)
{
	dim3 threads = getThreads(camera->get_numRays());
	dim3 blocks = getBlocks(camera->get_numRays());
	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;

	cuda_rayCaster<<<blocks, threads>>>(camera->get_W(), camera->get_H(), lightPosition, camera->get_position(), camera->get_rayDirections(), iso, cube, cubeDim, cubeInc, level, nLevel, buffer);
	std::cerr<<"Synchronizing rayCaster: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
	return;
}
