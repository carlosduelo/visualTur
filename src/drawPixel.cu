#include "visualTur_device.hpp"
#include "hdf5.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _INTERGL_
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
GLuint  positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;
#else
float * screenG;
float * screenC;
#endif

int W = 1024;
int H = 1024;
visualTur_device * VisualTur; 


//FPS
long long int 	frameCount = 0;
int 		currentTime;
int 		previousTime;
float		fps;

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
		#if 0
		std::cerr<<"Cannot close file with open handles: " << 
		H5Fget_obj_count( file_id, H5F_OBJ_FILE )  	<<" file, " 	<<
		H5Fget_obj_count( file_id, H5F_OBJ_DATASET )  	<<" data, " 	<<
		H5Fget_obj_count( file_id, H5F_OBJ_GROUP )  	<<" group, "	<<
		H5Fget_obj_count( file_id, H5F_OBJ_DATATYPE )	<<" type, "	<<
		H5Fget_obj_count( file_id, H5F_OBJ_ATTR )	<<" attr"	<<std::endl;
		exit(0);
		#endif
		//return -1;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
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


void drawFPS()
{
	std::cout<<"FPS: "<<fps<<std::endl;
}

void display()
{

#ifdef _INTERGL_
	// Map buffer object for writing from CUDA
	float * positions;
	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);

	//VisualTur->updateVisibleCubes(positions);

	// Unmap buffer object
	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
	// Render from buffer object
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
/*
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, positionsVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
*/
	drawFPS();

	// Swap buffers
	glutSwapBuffers();
	glutPostRedisplay();


#else
	VisualTur->updateVisibleCubes();

	std::cerr<<"Retrieve screen from GPU: "<< cudaGetErrorString(cudaMemcpy((void*) screenC, (const void*) screenG, sizeof(float)*W*H*4, cudaMemcpyDeviceToHost))<<std::endl;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawPixels(W, H, GL_RGBA, GL_FLOAT, screenC);

	drawFPS();

	glutSwapBuffers();
#endif
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

	int numT = 1;
	std::cout<<"Number of threads:"<<std::endl;
	std::cin >> numT;
	
	int nLevel = getnLevelFile(argv[1], argv[2]);

	#ifdef _INTERGL_
	cudaGLSetGLDevice(device);

	// Create buffer object and register it with CUDA
	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, positionsVBO);
	unsigned int size = W * H * 4 * sizeof(float);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
/*
glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
unsigned int size = W* H* 4 * sizeof(float);
glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
glBindBuffer(GL_ARRAY_BUFFER, 0);
*/
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);

	#else

	cudaSetDevice(device);
	screenG = 0;
	screenC = new float[H*W*4];
	std::cerr<<"Allocating memory octree CUDA screen: "<< cudaGetErrorString(cudaMalloc((void**)&screenG, sizeof(float)*H*W*4))<<std::endl;
	std::cerr<<"Cuda mem set: "<<cudaGetErrorString(cudaMemset((void *)screenG,0,sizeof(float)*H*W*4))<<std::endl;		
	#endif

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	//get the amount of free memory on the graphics card  
    	size_t free;  
	size_t total;  
    	cudaMemGetInfo(&free, &total); 

	visualTurParams_device_t params;
	params.W = W;
	params.H = H;
	params.fov_H = 35.0f;
	params.fov_W = 35.0f;
	params.distance = 50.0f;
	params.numRayPx = 1;
	params.maxElementsCache = 3*(free / (66*66*66*4)) /4;
	params.maxElementsCache_CPU = 5000;
	params.dimCubeCache = make_int3(64,64,64);
	params.cubeInc = 2;
	params.levelCubes = nLevel - 6;
	params.octreeLevel = (nLevel - 6) + 3;
	params.hdf5File = argv[1];
	params.dataset_name = argv[2];
	params.octreeFile = argv[3];

	params.numThreads = numT;
	params.deviceID = device;
	params.startRay = 0;
	params.endRay = params.W*params.H*params.numRayPx;

	VisualTur = new visualTur_device(params, screenG); 

	VisualTur->camera_Move(make_float3(x,y,z));

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
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
