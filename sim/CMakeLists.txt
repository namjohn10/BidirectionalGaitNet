cmake_minimum_required(VERSION 3.5.1)
project(sim)


add_compile_options(-Wdeprecated-declarations)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(DART REQUIRED COMPONENTS collision-bullet  CONFIG)
find_package(TinyXML2 REQUIRED)
# find_package(PythonLibs REQUIRED)
# find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")

add_library(sim ${srcs})
target_link_libraries(sim ${DART_LIBRARIES} ${TinyXML_LIBRARIES})
