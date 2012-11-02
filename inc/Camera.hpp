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
		void UpdateRays();
		Screen * screen;
		int			numRayPixel;
		int			numRays;
		float3 	 *		rayDirections;
		float3			look;
		float3			up;
		float3			right;
		float3			position;
		float                  	RotatedX;
		float                  	RotatedY;
		float                  	RotatedZ;
	public:

		Camera(int nRP, int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w); //inits the values (Position: (0|0|0) Target: (0|0|-1) )

		~Camera();

		void		Move(float3 Direction);
		void		RotateX(float Angle);
		void		RotateY(float Angle);
		void		RotateZ(float Angle);
		void		MoveForward(float Distance);
		void		MoveUpward(float Distance);
		void		StrafeRight(float Distance);	
		int		get_H();
		int		get_W();
		float		get_h();
		float		get_w();
		float		get_Distance();
		float		get_fovH();
		float		get_fovW();
		int		get_numRayPixel();
		int		get_numRays();
		float3 *	get_rayDirections();
		float3		get_look();
		float3		get_up();
		float3		get_right();
		float3		get_position();
		float           get_RotatedX();
		float           get_RotatedY();
		float           get_RotatedZ();
		void		set_H(int pH);
		void		set_W(int pW);
		void		set_Distance(float pd);
		void		set_fovH(float pfh);
		void		set_fovW(float pfw);
		void 		set_RayPerPixel(int rpp);

};
#endif/*_CAMERA_H_*/
