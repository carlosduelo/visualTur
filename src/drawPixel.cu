#include "visualTur.hpp"
#include "hdf5.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "helper_timer.h"
#include <timer.h>

float * screenG;
float * screenC;

int W = 1024;
int H = 1024;
visualTur * VisualTur; 

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;
#define MAX(a,b) ((a > b) ? a : b)
StopWatchInterface *timer = NULL;

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "%3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

#define WHERESTR  "Error[file "<<__FILE__<<", line "<<__LINE__<<"]: "
int getnLevelFile(char * file_name, char * dataset_name)
{

	hid_t           file_id;
	hid_t           dataset_id;
	hid_t           spaceid;
	int             ndim;
	hsize_t         dims[3];
	if ((file_id    = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		std::cerr<< WHERESTR<<" unable to open the requested file"<<std::endl;
		exit(0);
	}

	if ((dataset_id = H5Dopen1(file_id, dataset_name)) < 0 )
	{
		std::cerr<<WHERESTR<<" unable to open the requested data set"<<std::endl;
		exit(0);
	}

	if ((spaceid    = H5Dget_space(dataset_id)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to open the requested data space"<<std::endl;
		exit(0);
	}

	if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
	{
		std::cerr<<WHERESTR<<" handling file"<<std::endl;
		exit(0);
	}

	herr_t      status;

	if ((status = H5Dclose(dataset_id)) < 0)
	{
		std::cerr<<WHERESTR<<" unable to close the data set"<<std::endl;
		exit(0);
	}


	if ((status = H5Fclose(file_id)) < 0);
	{
		std::cerr<<WHERESTR<<" unable to close the file"<<std::endl;
	}

	int dimension;
	if (dims[0]>dims[1] && dims[0]>dims[2])
                dimension = dims[0];
        else if (dims[1]>dims[2])
                dimension = dims[1];
        else
                dimension = dims[2];

        /* Calcular dimension del árbol*/
        float aux = logf(dimension)/logf(2.0);
        float aux2 = aux - floorf(aux);
        int nLevels = aux2>0.0 ? aux+1 : aux;

	return nLevels; 
}

void display()
{
	sdkStartTimer(&timer);

	VisualTur->updateVisibleCubes(screenG);

	std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*W*H*4, cudaMemcpyDeviceToHost))<<std::endl;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawPixels(W, H, GL_RGBA, GL_FLOAT, screenC);

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();

}

void KeyDown(unsigned char key, int x, int y)
{
        switch (key)
        {
//              case 27:                //ESC
//                      PostQuitMessage(0);
//                      break;
                case 'a':
                        VisualTur->camera_RotateY(5.0);
                        break;
                case 'd':
                        VisualTur->camera_RotateY(-5.0);
                        break;
                case 'w':
                        VisualTur->camera_MoveForward( -5.0 ) ;
                        break;
                case 's':
                        VisualTur->camera_MoveForward( 5.0 ) ;
                        break;
                case 'x':
                        VisualTur->camera_RotateX(5.0);
                        break;
                case 'y':
                        VisualTur->camera_RotateX(-5.0);
                        break;
                case 'c':
                        VisualTur->camera_StrafeRight(-5.0);
                        break;
                case 'v':
                        VisualTur->camera_StrafeRight(5.0);
                        break;
                case 'f':
                        VisualTur->camera_MoveUpward(-5.0);
                        break;
                case 'r':
                        VisualTur->camera_MoveUpward(5.0);
                        break;
                case 'm':
                        VisualTur->camera_RotateZ(-5.0);
                        break;
                case 'n':
                        VisualTur->camera_RotateZ(5.0);
                        break;
        }
	display();
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cerr<<"Error, testVisualTur hdf5_file dataset_name octree_file [device]"<<std::endl;
		return 0;
	}

	int device = 0;
	if (argc > 4)
	{
		device = atoi(argv[4]);
	}

	int x,y,z;

	std::cout<<"Display Resolution:"<<std::endl;
	std::cout<<"Width: ";
	std::cin >> W;
	std::cout<<"Height: ";
	std::cin >> H;
	std::cout<<"Camera position (X,Y,Z):"<<std::endl;
	std::cout<<"X: ";
	std::cin >> x;
	std::cout<<"Y: ";
	std::cin >> y;
	std::cout<<"Z: ";
	std::cin >> z;
	
	int nLevel = getnLevelFile(argv[1], argv[2]);

	cudaSetDevice(device);
	screenG = 0;
	screenC = new float[H*W*4];
	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*H*W*4))<<std::endl;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	sdkCreateTimer(&timer);
	
	//get the amount of free memory on the graphics card  
    	size_t free;  
	size_t total;  
    	cudaMemGetInfo(&free, &total); 

	visualTurParams_t params;
	params.W = W;
	params.H = H;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 3*(free / (38*38*38*4)) /4;
	params.maxElementsCache_CPU = 5000;
	params.dimCubeCache = make_int3(32,32,32);
	params.cubeInc = 2;
	params.levelCubes = nLevel - 5;
	params.octreeLevel = (nLevel - 5) + 3;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(x,y,z));

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(W, H);
	glutCreateWindow(argv[1]);

	glutDisplayFunc(display);
	glutKeyboardFunc(KeyDown);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glutMainLoop();
}
