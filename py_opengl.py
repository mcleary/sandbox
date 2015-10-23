#! /usr/bin/env python

# This code is based on PyOpenGL_Demo shader demo code

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL.GL.ARB.shader_objects import *
from OpenGL.GL.ARB.vertex_shader import *
from OpenGL.GL.ARB.fragment_shader import *


def init_gl(width, height):
    """
    We call this right after our OpenGL window is created.
    A general OpenGL initialization function.  Sets all of the initial parameters.

    :param width: width of the window, used to calculate the projection matrix
    :param height: height of the window, used to calcule the projection matrix
    """
    glClearColor(0.0, 0.0, 0.0, 0.0)        # This Will Clear The Background Color To Black
    glClearDepth(1.0)                       # Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)                    # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)                 # Enables Depth Testing
    glShadeModel(GL_SMOOTH)                 # Enables Smooth Color Shading
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()                        # Reset The Projection Matrix

    # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
 
    if not glInitShaderObjectsARB():
        print('Missing Shader Objects!')
        sys.exit(1)
    if not glInitVertexShaderARB():
        print('Missing Vertex Shader!')
        sys.exit(1)
    if not glInitFragmentShaderARB():
        print('Missing Fragment Shader!')
        sys.exit(1)


def resize_gl_scene(width, height):
    """
    # The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
    :param width: new window width
    :param height: new window height
    """
    if height == 0:                        # Prevent A Divide By Zero If The Window Is Too Small
        height = 1

    glViewport(0, 0, width, height)        # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def draw_gl_scene():
    """
    The main drawing function.
    """
    # Clear The Screen And The Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()                    # Reset The View 

    # Move Left 1.5 units and into the screen 6.0 units.
    glTranslatef(-1.5, 0.0, -6.0)

    glutSolidSphere(1.0, 32, 32)
    glTranslate(1, 0, 2)
    glutSolidCube(1.0)

    #  since this is double buffered, swap the buffers to display what just got drawn. 
    glutSwapBuffers()


def key_pressed(*args):
    """
    The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
    :param args: keys that were pressed
    """
    # If escape is pressed, kill everything.
    if args[0] == '\x1b':
        sys.exit()


def main():
    global window

    # For now we just pass glutInit one empty argument. I wasn't sure what should or
    #  could be passed in (tuple, list, ...)
    # Once I find out the right stuff based on reading the PyOpenGL source, I'll address this.
    glutInit(sys.argv)

    # Select type of Display mode:   
    #  Double buffer 
    #  RGBA color
    # Alpha components supported 
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    
    # get a 640 x 480 window 
    glutInitWindowSize(640, 480)
    
    # the window starts at the upper left corner of the screen 
    glutInitWindowPosition(0, 0)
    
    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    window = glutCreateWindow("Jeff Molofee's GL Code Tutorial ... NeHe '99")

    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.    
    glutDisplayFunc(draw_gl_scene)
    
    # Uncomment this line to get full screen.
    # glutFullScreen()

    # When we are doing nothing, redraw the scene.
    glutIdleFunc(draw_gl_scene)
    
    # Register the function called when our window is resized.
    glutReshapeFunc(resize_gl_scene)
    
    # Register the function called when the keyboard is pressed.  
    glutKeyboardFunc(key_pressed)

    # Initialize our window. 
    init_gl(640, 480)

    # Start Event Processing Engine    
    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.

if __name__ == "__main__":
    print("Hit ESC key to quit.")
    main()

