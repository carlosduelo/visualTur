/*
 * Camera
 *
 */

#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Screen.hpp"
#include "cutil_math.h"

class Camera
{
	private:
		Screen * screen;
		int			numRayPixel;
		int			startRay;
		int			endRay;
		int			numRays;
		float 	 *		rayDirections;

		// Camera
		float3			look;
		float3			up;
		float3			right;
		float3			position;
		float                  	RotatedX;
		float                  	RotatedY;
		float                  	RotatedZ;

		// Private Methods
		void UpdateRays(cudaStream_t 	stream);
	public:

		Camera(int sRay, int eRay, int nRP, int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w, cudaStream_t stream); //inits the values (Position: (0|0|0) Target: (0|0|-1) )

		~Camera();

		void		Move(float3 Direction, cudaStream_t 	stream);
		void		RotateX(float Angle, cudaStream_t 	stream);
		void		RotateY(float Angle, cudaStream_t 	stream);
		void		RotateZ(float Angle, cudaStream_t 	stream);
		void		MoveForward(float Distance, cudaStream_t 	stream);
		void		MoveUpward(float Distance, cudaStream_t 	stream);
		void		StrafeRight(float Distance, cudaStream_t 	stream);	
		int		get_H();
		int		get_W();
		float		get_h();
		float		get_w();
		float		get_Distance();
		float		get_fovH();
		float		get_fovW();
		int		get_numRayPixel();
		int		get_numRays();
		float *		get_rayDirections();
		float3		get_look();
		float3		get_up();
		float3		get_right();
		float3		get_position();
		float           get_RotatedX();
		float           get_RotatedY();
		float           get_RotatedZ();
};
#endif/*_CAMERA_H_*/
