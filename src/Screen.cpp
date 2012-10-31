/*
 * Screen
 *
 */

#include "Screen.hpp"
#include <iostream>
#include <math.h>


Screen::Screen(int p_H, int p_W, float p_d, float p_fov_h, float p_fov_w)
{
	H       = p_H;
	W       = p_W;
	distance= p_d;
	fov_H   = p_fov_h;
	fov_W   = p_fov_w;
	h       = 2*p_d*tanf(p_fov_h*(M_PI/180.0));
	w       = 2*p_d*tanf(p_fov_w*(M_PI/180.0));

	std::cout << "Screen (" << p_H << "," << p_W << ")" << std::endl;
	std::cout << "Distance " << p_d << "  fov(" << p_fov_h << "," << p_fov_w << ")" << std::endl;
	std::cout << "Tamaño pantalla en mundo (" << h << "," << w << ")" << std::endl;
}

Screen::~Screen()
{
}

void Screen::createScreen()
{
	h       = 2*distance*tanf(fov_H*(M_PI/180.0));
	w       = 2*distance*tanf(fov_W*(M_PI/180.0));

	std::cout << "Screen (" << H << "," << W << ")" << std::endl;
	std::cout << "Distance " << distance << "  fov(" << fov_H << "," << fov_W << ")" << std::endl;
	std::cout << "Tamaño pantalla en mundo (" << h << "," << w << ")" << std::endl;
}

int Screen::get_H()
{
	return H;
}

int Screen::get_W()
{
	return W;
}

float Screen::get_h()
{
	return h;
}

float Screen::get_w()
{
	return w;
}

float Screen::get_Distance()
{
	return distance;
}

float Screen::get_fovH()
{
	return fov_H;
}

float Screen::get_fovW()
{
	return fov_W;
}

void Screen::set_H(int pH)
{
	H = pH;
}

void Screen::set_W(int pW)
{
	W = pW;
}

void Screen::set_Distance(float pd)
{
	distance = pd;
	createScreen();
}

void Screen::set_fovH(float pfh)
{
	fov_H = pfh;
	createScreen();
}

void Screen::set_fovW(float pfw)
{
	fov_W = pfw;
	createScreen();
}
