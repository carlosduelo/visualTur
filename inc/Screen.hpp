/*
 * Screen
 *
 */

#ifndef _SCREEN_H_
#define _SCREEN_H_

#include "config.hpp"

class Screen
{
	private:
		int H;	
		int W;
		float distance;
		float fov_H;  	
		float fov_W;
		float h;
		float w;

		void createScreen();
	public:
		Screen(int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w);

		~Screen();

		int get_H();

		int get_W();

		float get_h();

		float get_w();

		float get_Distance();

		float get_fovH();

		float get_fovW();

		void set_H(int pH);

		void set_W(int pW);

		void set_Distance(float pd);

		void set_fovH(float pfh);

		void set_fovW(float pfw);
};
#endif /*_SCREEN_H_*/

