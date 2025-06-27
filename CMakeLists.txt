cmake_minimum_required(VERSION 3.16...3.26)
project(kiss-lidarodometry)

find_package(OpenGL REQUIRED)

# Option to enable ROS 2 support (Linux only)
option(WITH_ROS2 "Enable ROS 2 rosbag support" OFF)

# ROS 2 dependencies (Linux only)
if(WITH_ROS2 AND NOT WIN32)
    find_package(ament_cmake QUIET)
    find_package(rclcpp QUIET)
    find_package(sensor_msgs QUIET)
    find_package(rosbag2_cpp QUIET)
    find_package(rosbag2_storage QUIET)
    
    if(ament_cmake_FOUND AND rclcpp_FOUND AND sensor_msgs_FOUND AND rosbag2_cpp_FOUND)
        set(ROS2_FOUND TRUE)
        message(STATUS "ROS 2 dependencies found - enabling rosbag support")
        add_definitions(-DWITH_ROS2)
    else()
        set(ROS2_FOUND FALSE)
        message(WARNING "ROS 2 dependencies not found - rosbag support disabled")
        set(WITH_ROS2 OFF)
    endif()
endif()

include_directories(
        3rdparty/
        3rdparty/imgui
        3rdparty/imgui/backends
        3rdparty/ImGuizmo
        3rdparty/glm
        3rdparty/portable-file-dialogs-master
)


message("Windows, building laszip from source")
add_subdirectory(3rdparty/LASzip/)
include_directories(3rdparty/LASzip/include)

# LASzip sources


set(IMGUI_SRCS
        3rdparty/imgui/imgui.cpp
        3rdparty/imgui/imgui_demo.cpp
        3rdparty/imgui/imgui_draw.cpp
        3rdparty/imgui/imgui_tables.cpp
        3rdparty/imgui/imgui_widgets.cpp
        3rdparty/imgui/backends/imgui_impl_glut.cpp
        3rdparty/imgui/backends/imgui_impl_opengl2.cpp)

set(GIZMO 3rdparty/ImGuizmo/ImGuizmo.cpp )

if (WIN32)
        message("Windows, building freeglut from source")
        include_directories(3rdparty/freeglut-3.2.2/include/)
        add_subdirectory(3rdparty/freeglut-3.2.2)
else()
        message("Linux, using system freeglut")
        find_package(GLUT REQUIRED)
        include_directories(${GLUT_INCLUDE_DIRS})

endif()

#kiss-icp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kiss-icp/cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/kiss_icp)
message("LASzip library")


add_executable(kiss_icp_lidar_odometry main.cpp  lidar_odometry_utils.cpp ${IMGUI_SRCS} ${GIZMO})
set_target_properties(kiss_icp_lidar_odometry PROPERTIES CXX_STANDARD 20)
if (WIN32)
    include(3rdparty/kiss-icp/cpp/kiss_icp/cmake/CompilerOptions.cmake)
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip3 freeglut_static ${OPENGL_LIBRARIES})
else ()
    set(LINUX_LIBS kiss_icp_pipeline laszip ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
    
    # Add ROS 2 libraries if available
    if(ROS2_FOUND)
        ament_target_dependencies(kiss_icp_lidar_odometry
            rclcpp
            sensor_msgs
            rosbag2_cpp
            rosbag2_storage)
    endif()
    
    target_link_libraries(kiss_icp_lidar_odometry ${LINUX_LIBS})
endif ()

install(TARGETS kiss_icp_lidar_odometry DESTINATION bin)

# CPack configuration
set(CPACK_PACKAGE_NAME "hd_mapping")
set(CPACK_PACKAGE_VERSION "${HDMAPPING_VERSION_MAJOR}.${HDMAPPING_VERSION_MINOR}.${HDMAPPING_VERSION_PATCH}")

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Kiss ICP Lidar Odometry")
set(CPACK_PACKAGE_CONTACT "michalpelka@gmail.com")
set(CPACK_GENERATOR "ZIP")

include(CPack)
