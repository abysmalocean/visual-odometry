
# PCL library
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories( "/usr/include/eigen3" )
# opencv
find_package( OpenCV REQUIRED ) 

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

ADD_LIBRARY( utility utility.cpp )
TARGET_LINK_LIBRARIES( utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( main main.cpp )
TARGET_LINK_LIBRARIES( main
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( vo vo.cpp )
TARGET_LINK_LIBRARIES( 
    vo 
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( incParameter incParameter.cpp )
TARGET_LINK_LIBRARIES( incParameter
    utility )



ADD_EXECUTABLE( pointClould pointClould.cpp )
TARGET_LINK_LIBRARIES( pointClould
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE( measureTwo measureTwo.cpp )
TARGET_LINK_LIBRARIES( measureTwo
    utility
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )