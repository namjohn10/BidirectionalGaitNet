cmake_minimum_required(VERSION 3.5.1)
project(mysim)

add_compile_options(-fvisibility=hidden)

link_directories(../sim/)
include_directories(../sim/)

find_package(DART REQUIRED COMPONENTS gui collision-bullet  CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PythonLibs REQUIRED)

# find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)

find_package(pybind11 REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})

# file(GLOB srcs "GLfunctions.h" "GLfunctions.cpp" "GLFWApp.h" "GLFWApp.cpp" "ShapeRenderer.h" "ShapeRenderer.cpp" "InverseGaitNet.h" "InverseGaitNet.cpp" "stb_image_write.h") 
file(GLOB srcs "*.h" "*.cpp") 

add_executable(viewer "main.cpp" ${srcs})
target_link_libraries(viewer ${DART_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut glad pybind11::embed glfw imgui sim)

# add_executable(collector "collector.cpp" ${srcs})
# target_link_libraries(collector ${DART_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut glad pybind11::embed glfw imgui sim)