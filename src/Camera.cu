/*
 * Camera
 *
 */

#include "Camera.hpp"
#include "cuda_help.hpp"
#include <math.h>
#include <iostream>
#include <fstream>

Camera::Camera(int sRay, int eRay, int nRP, int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w, cudaStream_t stream) //inits the values (Position: (0|0|0) Target: (0|0|-1) )
{
	// Init screen
	screen		= new Screen(p_H, p_W, p_d, p_fov_h, p_fov_w);

	// Antialising, supersampling, number rays per pixel nRP*nRP, matrix of rays
	numRayPixel	= nRP;

	// Range of rays
	startRay 	= sRay;
	endRay		= eRay; 

	
	look 	= make_float3(0.0, 0.0, -1.0);
	up 	= make_float3(0.0, 1.0, 0.0);
	position= make_float3(0.0, 0.0, 0.0);
	right   = cross(look, up);
	RotatedX	= 0.0;
	RotatedY	= 0.0;
	RotatedZ	= 0.0;	

	numRays	= endRay - startRay;
	std::cerr<<"Allocating memory camera directions buffer: "<<3*numRays*sizeof(float)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, 3*numRays*sizeof(float))) << std::endl;

	UpdateRays(stream);
}

Camera::~Camera()
{
	cudaFree(rayDirections);
	delete screen;
}

__global__ void cuda_updateRays(float * rays, int numRays, int sR, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numRays)
	{
		int i  = (id+sR) / W;
		int j  = (id+sR) % W;

		float ih  = h/H;
		float iw  = w/W;

		float3 A = (look * distance);
		A += up * (-(h/2.0f) + (ih*(i + 0.5f)));
		A += right * (-(w/2.0f) + (iw*(j + 0.5f)));
		A = normalize(A);

		rays[id] 		= A.x;
		rays[id+numRays] 	= A.y;
		rays[id+2*numRays] 	= A.z;
	}
}


// NEED TO COMPLETE CODE, THINK HOW CARRY OUT THE ANTIALISING
__global__ void cuda_updateRaysAntiAliassing(float * rays, int numRays, int sR, int numRP, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numRays)
	{
		int i  = ((id+sR)/(numRP*numRP)) / W;
		int j  = ((id+sR)/(numRP*numRP)) % W;
		//int ra = (id+sR) % (numRP*numRP);
		float ih  = h/H;
		float iw  = w/W;

		float3 A = (look * distance);
		A += up * ((h/2.0f) - (ih*(i + 0.5f)));
		A += right * (-(w/2.0f) + (iw*(j + 0.5f)));
		A = normalize(A);

		rays[id] 		= A.x;
		rays[id+numRays] 	= A.y;
		rays[id+2*numRays] 	= A.z;
	}
}
void Camera::UpdateRays(cudaStream_t stream)
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);
	std::cerr<<"Launching kernek ray generation blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	if (numRayPixel == 1)
		cuda_updateRays<<<blocks, threads, 0, stream>>>(rayDirections, numRays, startRay, up, right, look, screen->get_H(), screen->get_W(), screen->get_h(), screen->get_w(), screen->get_Distance());
	else
		cuda_updateRaysAntiAliassing<<<blocks, threads>>>(rayDirections, numRays, startRay, numRayPixel, up, right, look, screen->get_H(), screen->get_W(), screen->get_h(), screen->get_w(), screen->get_Distance());

	//std::cerr<<"Synchronizing calculating rays: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
}

void Camera::Move(float3 Direction, cudaStream_t 	stream)
{
	position += Direction;
	UpdateRays(stream);
}

void Camera::RotateX(float Angle, cudaStream_t 	stream)
{	
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedX += Angle;

	//Rotate viewdir around the right vector:
	look = look*cPI180 + up*sPI180;
	normalize(look);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);

	UpdateRays(stream);                                                                
}

void Camera::RotateY(float Angle, cudaStream_t 	stream)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedY += Angle;

	//Rotate viewdir around the up vector:
	look = look*cPI180 - right*sPI180;
	normalize(look);

	//now compute the new RightVector (by cross product)
	right = cross(look, up);
	UpdateRays(stream);                                                                
}

void Camera::RotateZ(float Angle, cudaStream_t 	stream)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedZ += Angle;

	//Rotate viewdir around the right vector:
	right = right*cPI180 + up*sPI180;
	normalize(right);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);
	UpdateRays(stream);                                                                
}

void Camera::MoveForward(float Distance, cudaStream_t 	stream)
{
	position += look*(-Distance);
	UpdateRays(stream);                                                                
}

void Camera::MoveUpward(float Distance, cudaStream_t 	stream)
{
	position += right*Distance;
	UpdateRays(stream);                                                                
}

void Camera::StrafeRight(float Distance, cudaStream_t 	stream)
{
	position += up*Distance;
	UpdateRays(stream);                                                                
}

int		Camera::get_H(){return screen->get_H();}
int		Camera::get_W(){return screen->get_W();}
float		Camera::get_h(){return screen->get_h();}
float		Camera::get_w(){return screen->get_w();}
float		Camera::get_Distance(){return screen->get_Distance();}
float		Camera::get_fovH(){return screen->get_fovH();}
float		Camera::get_fovW(){return screen->get_fovW();}
int		Camera::get_numRayPixel(){return numRayPixel;}
int		Camera::get_numRays(){return numRays;}
float *		Camera::get_rayDirections(){return rayDirections;}
float3		Camera::get_look(){return look;}
float3		Camera::get_up(){return up;}
float3		Camera::get_right(){return right;}
float3		Camera::get_position(){return position;}
float           Camera::get_RotatedX(){return RotatedX;}
float           Camera::get_RotatedY(){return RotatedY;}
float           Camera::get_RotatedZ(){return RotatedZ;}
