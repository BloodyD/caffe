# - Try to find LibOpenCV
# Once done this will define
#  LIBOpenCV_FOUND - System has opencv
#  LIBOpenCV_INCLUDE_DIRS - The opecv include directories
#  LIBOpenCV_LIBRARIES - The libraries needed to use opencv
#  LIBOpenCV_DEFINITIONS - Compiler switches required for using opencv

set(CMAKE_VERBOSE_MAKEFILE "TRUE")

find_package(PkgConfig)
pkg_check_modules(PC_LIBOpenCV opencv)

find_path(LIBOpenCV_INCLUDE_DIR  opencv2/opencv.hpp
          HINTS 
	/opt/local/include
	${PC_LIBOpenCV_INCLUDE_DIRS} 
	${PC_LIBOpenCV_INCLUDE_DIR} )
set(LIBOpenCV_INCLUDE_DIRS ${LIBOpenCV_INCLUDE_DIR} )

set(OPENCV_LIB_COMPONENTS core imgproc highgui ml video features2d calib3d objdetect contrib legacy flann nonfree)
#set(OpenCV_VERSION ${PC_LIBOpenCV_VERSION})

set(OpenCV_LIBS "")
set(OpenCV_FOUND_TMP true)
foreach(__CVLIB ${OPENCV_LIB_COMPONENTS})
	#DEBUG
	#message(STATUS ${__CVLIB})

	#find_library(OpenCV_${__CVLIB}_LIBRARY_DEBUG NAMES "opencv_${__CVLIB}d" "libopencv_${__CVLIB}d" PATHS "${PC_LIBOpenCV_LIBDIR} ${PC_LIBOpenCV_LIBRARY_DIRS}" NO_DEFAULT_PATH)
	find_library(OpenCV_${__CVLIB}_LIBRARY_RELEASE NAMES "opencv_${__CVLIB}" "libopencv_${__CVLIB}" 
		HINTS 
		/opt/local/lib
		${PC_LIBOpenCV_LIBDIR} 
		${PC_LIBOpenCV_LIBRARY_DIRS})
	#DEBUG
	#message(STATUS ${OpenCV_${__CVLIB}_LIBRARY_RELEASE})
                
	#Remove the cache value
	set(OpenCV_${__CVLIB}_LIBRARY "" CACHE STRING "" FORCE)
        
	#both debug/release
	if(OpenCV_${__CVLIB}_LIBRARY_DEBUG AND OpenCV_${__CVLIB}_LIBRARY_RELEASE)
		set(OpenCV_${__CVLIB}_LIBRARY debug ${OpenCV_${__CVLIB}_LIBRARY_DEBUG} optimized ${OpenCV_${__CVLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#only debug
	elseif(OpenCV_${__CVLIB}_LIBRARY_DEBUG)
                        set(OpenCV_${__CVLIB}_LIBRARY ${OpenCV_${__CVLIB}_LIBRARY_DEBUG}  CACHE STRING "" FORCE)
	#only release
	elseif(OpenCV_${__CVLIB}_LIBRARY_RELEASE)
		set(OpenCV_${__CVLIB}_LIBRARY ${OpenCV_${__CVLIB}_LIBRARY_RELEASE}  CACHE STRING "" FORCE)
	#no library found
	else()
        set(OpenCV_FOUND_TMP false)
	endif()
                
    #Add to the general list
    if(OpenCV_${__CVLIB}_LIBRARY)
		set(OpenCV_LIBS ${OpenCV_LIBS} ${OpenCV_${__CVLIB}_LIBRARY})
    endif(OpenCV_${__CVLIB}_LIBRARY)
                
endforeach(__CVLIB)
set(LIBOpenCV_FOUND ${OpenCV_FOUND_TMP} CACHE BOOL "" FORCE)
set(LIBOpenCV_LIBRARIES ${OpenCV_LIBS} )


message(STATUS "[FindOpenCV] Using OpenCV include dir: ${LIBOpenCV_INCLUDE_DIRS}")
message(STATUS "[FindOpenCV] Using OpenCV libraries: ${LIBOpenCV_LIBRARIES}")


##====================================================
## Print message
##----------------------------------------------------
if(NOT LIBOpenCV_FOUND)
  # make FIND_PACKAGE friendly
  if(NOT OpenCV_FIND_QUIETLY)
        if(OpenCV_FIND_REQUIRED)
          message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
        else(OpenCV_FIND_REQUIRED)
          message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
        endif(OpenCV_FIND_REQUIRED)
  endif(NOT OpenCV_FIND_QUIETLY)
endif(NOT LIBOpenCV_FOUND)

#~ include(FindPackageHandleStandardArgs)
#~ # handle the QUIETLY and REQUIRED arguments and set LIBOpenCV_FOUND to TRUE
#~ # if all listed variables are TRUE
#~ find_package_handle_standard_args(LIBOpenCV  DEFAULT_MSG LIBOpenCV_LIBRARIES LIBOpenCV_INCLUDE_DIR)
#~ mark_as_advanced(LIBOpenCV_INCLUDE_DIR LIBOpenCV_LIBRARY )
