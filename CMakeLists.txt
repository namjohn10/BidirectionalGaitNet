cmake_minimum_required(VERSION 3.5.1)
project(mysim)
set(OpenGL_GL_PREFERENCE LEGACY)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

link_libraries(stdc++fs)

add_compile_options(-fPIC)
# add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

add_compile_options(-DHAVE_CSTDDEF)

add_definitions(-DMASS_ROOT_DIR="..")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

# if(SERVER_BUILD)
find_package(PythonLibs 3.6 EXACT REQUIRED)
find_package(PythonInterp 3.6 EXACT REQUIRED)
# endif()


if(NOT SERVER_BUILD)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
endif()
add_subdirectory( libs )
add_subdirectory( sim )
add_subdirectory( python )
if(NOT SERVER_BUILD)
add_subdirectory( viewer )
endif()
