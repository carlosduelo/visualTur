#include "visualTur.hpp"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>

int W = 1024;
int H = 1024;
visualTur * VisualTur; 
float * screenG;
float * screenC;


//FPS
long long int 	frameCount = 0;
int 		currentTime;
int 		previousTime;
float		fps;

void drawFPS()
{
	std::cout<<"FPS: "<<fps<<std::endl;
}

void display()
{

	VisualTur->updateVisibleCubes(screenG);

	std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*W*H*4, cudaMemcpyDeviceToHost))<<std::endl;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawPixels(W, H,GL_RGBA, GL_FLOAT, screenC);

	drawFPS();

	glutSwapBuffers();
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
                        VisualTur->camera_MoveForward( -1.0 ) ;
                        break;
                case 's':
                        VisualTur->camera_MoveForward( 1.0 ) ;
                        break;
                case 'x':
                        VisualTur->camera_RotateX(5.0);
                        break;
                case 'y':
                        VisualTur->camera_RotateX(-5.0);
                        break;
                case 'c':
                        VisualTur->camera_StrafeRight(-0.1);
                        break;
                case 'v':
                        VisualTur->camera_StrafeRight(0.1);
                        break;
                case 'f':
                        VisualTur->camera_MoveUpward(-0.3);
                        break;
                case 'r':
                        VisualTur->camera_MoveUpward(0.3);
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

//-------------------------------------------------------------------------
// Calculates the frames per second
//-------------------------------------------------------------------------
void calculateFPS()
{
    //  Increase frame count
    frameCount++;

    //  Get the number of milliseconds since glutInit called
    //  (or first call to glutGet(GLUT ELAPSED TIME)).
    currentTime = glutGet(GLUT_ELAPSED_TIME);

    //  Calculate time passed
    int timeInterval = currentTime - previousTime;

    if(timeInterval > 1000)
    {
        //  calculate the number of frames per second
        fps = frameCount / (timeInterval / 1000.0f);

        //  Set time
        previousTime = currentTime;

        //  Reset frame count
        frameCount = 0;
    }
}

void idle (void)
{
    //  Animate the object

    //  Calculate FPS
    calculateFPS();

    //  Call display function (draw the current frame)
    glutPostRedisplay ();
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
		cudaSetDevice(device);
	}


	visualTurParams_t params;
	params.W = W;
	params.H = H;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 3500;
	params.maxElementsCache_CPU = 5000;
	params.dimCubeCache = make_int3(64,64,64);
	params.cubeInc = 2;
	params.levelCubes = 3;
	params.octreeLevel =8;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	VisualTur = new visualTur(params); 

	VisualTur->camera_Move(make_float3(128.0f, 128.0f, 550.0f));
	VisualTur->camera_MoveForward(1.0f);

	screenG = 0;
	screenC = new float[H*W*4];

	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*H*W*4))<<std::endl;

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(W, H);
	glutCreateWindow(argv[1]);

	glutDisplayFunc(display);
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mouse_button);
	//glutMotionFunc(mouse_motion);
	glutKeyboardFunc(KeyDown);
	glutIdleFunc(idle);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glutMainLoop();
}
