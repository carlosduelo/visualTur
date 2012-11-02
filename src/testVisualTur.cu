#include "visualTur.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char ** argv)
{
	visualTurParams_t params;
	params.W = 800;
	params.H = 800;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 100;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 1;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	visualTur * VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(52.0f, 52.0f, 520.0f));
	VisualTur->camera_MoveForward(1.0f);

	VisualTur->updateVisibleCubes(9);
}
