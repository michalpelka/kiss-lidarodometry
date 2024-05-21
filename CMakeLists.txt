cmake_minimum_required(VERSION 3.16...3.26)
project(kiss-lidarodometry)

find_package(OpenGL REQUIRED)

include_directories(
        3rdparty/
        3rdparty/imgui
        3rdparty/imgui/backends
        3rdparty/ImGuizmo
        3rdparty/freeglut/include
        3rdparty/glew-2.2.0/include/GL
        3rdparty/glm
        3rdparty/portable-file-dialogs-master
)

add_subdirectory(3rdparty/LASzip/)
include_directories(
        3rdparty/LASzip/include
)

set(IMGUI_SRCS
        3rdparty/imgui/imgui.cpp
        3rdparty/imgui/imgui_demo.cpp
        3rdparty/imgui/imgui_draw.cpp
        3rdparty/imgui/imgui_tables.cpp
        3rdparty/imgui/imgui_widgets.cpp
        3rdparty/imgui/backends/imgui_impl_glut.cpp
        3rdparty/imgui/backends/imgui_impl_opengl2.cpp
        )

set(GIZMO
        3rdparty/ImGuizmo/ImGuizmo.cpp
        )

include_directories(3rdparty/freeglut-3.2.2/include/)
add_subdirectory(3rdparty/freeglut-3.2.2)

#kiss-icp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kiss-icp/cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/kiss_icp)
message("LASzip library")
MESSAGE(STATUS "Opengl : ${OPENGL_LIBRARIES}")

add_executable(kiss_icp_lidar_odometry main.cpp  lidar_odometry_utils.cpp ${IMGUI_SRCS} ${GIZMO})
set_target_properties(kiss_icp_lidar_odometry PROPERTIES CXX_STANDARD 20)
if (WIN32)
    include(3rdparty/kiss-icp/cpp/kiss_icp/cmake/CompilerOptions.cmake)
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip3 freeglut_static ${OPENGL_LIBRARIES})
else ()
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip freeglut_static ${OPENGL_LIBRARIES})
endif ()