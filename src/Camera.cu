/*
 * Camera
 *
 */

#include "Camera.hpp"
#include "cuda_help.hpp"
#include <math.h>
#include <iostream>
#include <fstream>

Camera::Camera(int nRP, int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w) //inits the values (Position: (0|0|0) Target: (0|0|-1) )
{
	screen		= new Screen(p_H, p_W, p_d, p_fov_h, p_fov_w);
	numRayPixel	= nRP;

	look 	= make_float3(0.0, 0.0, -1.0);
	up 	= make_float3(0.0, 1.0, 0.0);
	position= make_float3(0.0, 0.0, 0.0);
	right   = cross(look, up);
	RotatedX	= 0.0;
	RotatedY	= 0.0;
	RotatedZ	= 0.0;	

	numRays	= screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

	UpdateRays();
}

Camera::~Camera()
{
	cudaFree(rayDirections);
	delete screen;
}



__global__ void cuda_updateRays(float3 * rays, int numRayPixel, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance, float fov_H, float fov_W, float3 position)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

#if _IMG_
	if (id < (W*H))
	{
		int i  = id / W;
		int j  = id % W;

		float ih  = h/H;
		float iw  = w/W;

		float3 A = (look * distance);
		A += up * ((h/2.0f) - (ih*(i + 0.5f)));
		A += right * (-(w/2.0f) + (iw*(j + 0.5f)));

		rays[id] = normalize(A);
	}
#else
	if (id < (W*H))
	{
		int i  = id / W;
		int j  = id % W;

		float ih  = h/H;
		float iw  = w/W;

		float3 A = (look * distance);
		A += up * (-(h/2.0f) + (ih*(i + 0.5f)));
		A += right * (-(w/2.0f) + (iw*(j + 0.5f)));

		rays[id] = normalize(A);
	}
#endif
}

void Camera::UpdateRays()
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);
	std::cerr<<"Launching kernek ray generation blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	cuda_updateRays<<<blocks, threads>>>(rayDirections, numRayPixel, up, right, look, screen->get_H(), screen->get_W(), screen->get_h(), screen->get_w(), screen->get_Distance(), screen->get_fovH(), screen->get_fovW(), position);
	std::cerr<<"Synchronizing calculating rays: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
}

void Camera::Move(float3 Direction)
{
	position += Direction;
	UpdateRays();                                                                
}

void Camera::RotateX(float Angle)
{	
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedX += Angle;

	//Rotate viewdir around the right vector:
	look = look*cPI180 + up*sPI180;
	normalize(look);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);

	UpdateRays();                                                                
}

void Camera::RotateY(float Angle)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedY += Angle;

	//Rotate viewdir around the up vector:
	look = look*cPI180 - right*sPI180;
	normalize(look);

	//now compute the new RightVector (by cross product)
	right = cross(look, up);
	UpdateRays();                                                                
}

void Camera::RotateZ(float Angle)
{
	float  sPI180 = sin(Angle*(M_PI/180.0));
	float  cPI180 = cos(Angle*(M_PI/180.0));

	RotatedZ += Angle;

	//Rotate viewdir around the right vector:
	right = right*cPI180 + up*sPI180;
	normalize(right);

	//now compute the new UpVector (by cross product)
	up = (-1.0f) * cross(look,right);
	UpdateRays();                                                                
}

void Camera::MoveForward(float Distance)
{
	position += look*(-Distance);
	UpdateRays();                                                                
}

void Camera::MoveUpward(float Distance)
{
	position += right*Distance;
	UpdateRays();                                                                
}

void Camera::StrafeRight(float Distance)
{
	position += up*Distance;
	UpdateRays();                                                                
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
float3 *	Camera::get_rayDirections(){return rayDirections;}
float3		Camera::get_look(){return look;}
float3		Camera::get_up(){return up;}
float3		Camera::get_right(){return right;}
float3		Camera::get_position(){return position;}
float           Camera::get_RotatedX(){return RotatedX;}
float           Camera::get_RotatedY(){return RotatedY;}
float           Camera::get_RotatedZ(){return RotatedZ;}
void		Camera::set_H(int pH)
{
	screen->set_H(pH);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_W(int pW)
{
	screen->set_W(pW);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_Distance(float pd)
{
	screen->set_Distance(pd);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_fovH(float pfh)
{
	screen->set_fovH(pfh);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_fovW(float pfw)
{
	screen->set_fovW(pfw);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void 	Camera::set_RayPerPixel(int rpp)
{
	numRayPixel = rpp;
	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
	std::cerr<<"Allocating memory camera directions buffer: "<<numRays*sizeof(float3)/1024/1024 << " MB: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}
