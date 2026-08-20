cmake_minimum_required(VERSION 3.16...3.26)
project(kiss-lidarodometry)

# Some bundled/fetched third-party projects (e.g. Sophus, tessil, fetched by
# kiss-icp) declare cmake_minimum_required() versions older than what recent
# CMake releases accept. Tell CMake to treat those as this minimum instead of
# hard-erroring.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

find_package(OpenGL REQUIRED)

include_directories(
        3rdparty/
        3rdparty/imgui
        3rdparty/imgui/backends
        3rdparty/ImGuizmo
        3rdparty/glm
        3rdparty/portable-file-dialogs-master
        3rd/Fusion/
)


message("Windows, building laszip from source")
if(APPLE)
    # macOS's off_t is already 64-bit and off64_t is not defined by the
    # system headers, unlike Linux/MinGW. LASzip's non-Windows fseeko()
    # path casts to off64_t, so alias it to off_t for Apple builds instead
    # of patching the pinned submodule source.
    add_compile_definitions(off64_t=off_t)
endif()
add_subdirectory(3rdparty/LASzip/)
include_directories(3rdparty/LASzip/include)
add_subdirectory(3rdparty/Fusion)
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

if (WIN32 OR APPLE)
        message("Building freeglut from source")
        if (APPLE)
            # Apple's system GLUT.framework is a frozen, pre-2001 implementation
            # with no scroll wheel/trackpad support and a close button that does
            # not work. freeglut's Cocoa backend (still opt-in upstream) fixes
            # both, matching the same approach used by the sister HDMapping
            # project (MapsHD/HDMapping), which validated this exact freeglut
            # commit with -DFREEGLUT_COCOA=ON.
            set(FREEGLUT_COCOA ON CACHE BOOL "" FORCE)
        endif ()
        include_directories(3rdparty/freeglut/include/)
        add_subdirectory(3rdparty/freeglut)
else()
        message("Linux, using system freeglut")
        find_package(GLUT REQUIRED)
        include_directories(${GLUT_INCLUDE_DIRS})

endif()

#kiss-icp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kiss-icp/cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/kiss_icp)
message("LASzip library")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(kiss_icp_lidar_odometry main.cpp  RegisteredFrame.cpp lidar_odometry_utils.cpp Params.cpp ${IMGUI_SRCS} ${GIZMO})
set_target_properties(kiss_icp_lidar_odometry PROPERTIES CXX_STANDARD 20)
if (APPLE)
    # freeglut's Cocoa backend needs a real app-bundle identity (CFBundleIdentifier)
    # to keep its window alive -- run as a bare, unbundled Mach-O binary and the
    # window is destroyed right after creation, so glutMainLoop() returns almost
    # immediately with no visible window. MACOSX_BUNDLE makes CMake wrap the
    # build output as a minimal .app bundle automatically.
    set_target_properties(kiss_icp_lidar_odometry PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.michalpelka.kiss-lidarodometry"
        MACOSX_BUNDLE_BUNDLE_NAME "kiss_icp_lidar_odometry")
endif ()
if (WIN32)
    include(3rdparty/kiss-icp/cpp/kiss_icp/cmake/CompilerOptions.cmake)
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip3 freeglut_static ${OPENGL_LIBRARIES} Fusion)
elseif (APPLE)
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip freeglut_static ${OPENGL_LIBRARIES} Fusion)
else ()
    target_link_libraries(kiss_icp_lidar_odometry  kiss_icp_pipeline laszip ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} Fusion)
endif ()

install(TARGETS kiss_icp_lidar_odometry DESTINATION bin)

# CPack configuration
set(CPACK_PACKAGE_NAME "hd_mapping")
set(CPACK_PACKAGE_VERSION "${HDMAPPING_VERSION_MAJOR}.${HDMAPPING_VERSION_MINOR}.${HDMAPPING_VERSION_PATCH}")

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Kiss ICP Lidar Odometry")
set(CPACK_PACKAGE_CONTACT "michalpelka@gmail.com")
set(CPACK_GENERATOR "ZIP")

include(CPack)
