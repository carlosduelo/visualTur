/*
 * Camera
 *
 */

#include "Camera.hpp"
#include <math.h>
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 16

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
	std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

	UpdateRays();
}

Camera::~Camera()
{
	cudaFree(rayDirections);
	delete screen;
}



__global__ void cuda_updateRays(float3 * rays, int numRayPixel, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance, float fov_H, float fov_W, float3 position)
{
		#if 0 //XXX CPU CODE
		float ih  = screen->h/screen->H;
		float iw  = screen->w/screen->W;

		vmml::vector<3, float> A = position + look * screen->distance;
		A += up * ((screen->h/2.0) + ih/2.0);
		A -= right * ((screen->w/2.0) + iw/2.0);
		for(int i=0; i < screen->H; i++)
			for(int j=0; j < screen->W; j++)
			{
				rayDirections[i*screen->W+j] = (A + (j*iw)*right - (i*ih*up)) - position;
				rayDirections[i*screen->W+j].normalize();
			}
		#endif

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j  = blockIdx.y * blockDim.y + threadIdx.y;
	int id = i * W + j;

	float ih  = h/H;
	float iw  = w/W;

	float3 A = position + (look * distance);
	A += up * ((h/2.0) + ih/2.0);
	A -= right * ((w/2.0) + iw/2.0);
	A += (j*iw)*right; 
	A -= (i*ih)*up;
	A -= position;
	rays[id] = normalize(A);

	#if 0
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j  = blockIdx.y * blockDim.y + threadIdx.y;

	int id = i * W + j;
	id *= numRayPixel*numRayPixel;
	
	float ih  = h/H;
	float iw  = w/W;

	float3 ph;
	float3 pw;
	float fov_H_PI = distance * tanf(fov_H*(M_PI/180.0));
	float fov_W_PI = distance * tanf(fov_W*(M_PI/180.0));
	ph = fov_H_PI * up;
	pw = fov_W_PI * right;
	float  ihr             = ih/(2.0*numRayPixel);
	float  iwr             = iw/(2.0*numRayPixel);
	float  matDisPixel_W[4][4];
	float  matDisPixel_H[4][4];
	for(int rpW = 0; rpW<numRayPixel; rpW++)
	{
		for(int rpH = 0; rpH<numRayPixel; rpH++)
		{
			matDisPixel_H[rpW][rpH] = (2.0*rpH*ihr+ihr);
			matDisPixel_W[rpW][rpH] = (2.0*rpW*iwr+iwr);
		}
	}
	int indice2 = 0;
	for(int rpW = 0; rpW<numRayPixel; rpW++)
	{
		for(int rpH = 0; rpH<numRayPixel; rpH++)
		{
			//rays[id + indice2].x = i;
			//rays[id + indice2].y = j;
			
			rays[id + indice2]         = (distance * look) + ph - pw;
			rays[id + indice2].x      -= i*ih+matDisPixel_H[rpW][rpH];
			rays[id + indice2].y      -= j*iw+matDisPixel_W[rpW][rpH];
			rays[id + indice2] 	= gpu_normalize(rays[id + indice2]);
			
			indice2++;
		}
	}
	#endif

}

void Camera::UpdateRays()
{
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(screen->get_H()/BLOCK_SIZE, screen->get_W()/BLOCK_SIZE);
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
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_W(int pW)
{
	screen->set_W(pW);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_Distance(float pd)
{
	screen->set_Distance(pd);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_fovH(float pfh)
{
	screen->set_fovH(pfh);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void		Camera::set_fovW(float pfw)
{
	screen->set_fovW(pfw);

	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}

void 	Camera::set_RayPerPixel(int rpp)
{
	numRayPixel = rpp;
	numRays = screen->get_H()*screen->get_W()*numRayPixel*numRayPixel;

	cudaFree(rayDirections);
        std::cerr<<"Allocating memory camera directions buffer: "<< cudaGetErrorString(cudaMalloc(&rayDirections, numRays*sizeof(float3))) << std::endl;

        UpdateRays();
}
