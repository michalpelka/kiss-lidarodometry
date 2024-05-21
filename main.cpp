#include <kiss_icp/pipeline/KissICP.hpp>
#include "lidar_odometry_utils.h"
#include <portable-file-dialogs.h>

#include <imgui.h>
#include <imgui_impl_glut.h>
#include <imgui_impl_opengl2.h>
#include <imgui_internal.h>

#include <GL/freeglut.h>

#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>


#include <nlohmann/json.hpp>
namespace fs = std::filesystem;

struct RegisteredFrame
{
    std::vector<Eigen::Vector3d> points;
    std::vector<float> intensities;
    std::vector<double> timestamps;
    Eigen::Affine3d pose;
};

std::vector<RegisteredFrame> ConcatenateFrames(const std::vector<RegisteredFrame>& frames, int maxNumberOfPoints = 2000000)
{
    std::vector<RegisteredFrame> result;
    result.resize(1);
    result.back().pose = frames.front().pose;
    Eigen::Affine3d currentIncrement = Eigen::Affine3d::Identity();
    for (const auto& currentOriginalFrame : frames)
    {
        currentIncrement = result.back().pose.inverse() * currentOriginalFrame.pose;
        for (size_t i = 0; i < currentOriginalFrame.points.size(); ++i)
        {
            assert(currentOriginalFrame.points.size() == currentOriginalFrame.intensities.size());
            assert(currentOriginalFrame.points.size() == currentOriginalFrame.timestamps.size());
            auto & buildFrame = result.back();
            buildFrame.points.push_back(currentIncrement * currentOriginalFrame.points[i]);
            buildFrame.intensities.push_back(currentOriginalFrame.intensities[i]);
            buildFrame.timestamps.push_back(currentOriginalFrame.timestamps[i]);
            if (buildFrame.points.size() >= maxNumberOfPoints)
            {
                result.resize(result.size() + 1);
                result.back().pose = currentOriginalFrame.pose;
                currentIncrement = result.back().pose.inverse() * currentOriginalFrame.pose;
            }
        }
    }
    std::cout << "Concatenated " << frames.size() << " frames into " << result.size() << " frames" << std::endl;
    return result;
}

std::vector<Point3Di> toPoint3DiV(const std::vector<Eigen::Vector3d>& points, const std::vector<float>& intensities, const std::vector<double>& timestamps)
{
    assert(points.size() == intensities.size());
    assert(points.size() == timestamps.size());
    std::vector<Point3Di> result;
    result.reserve(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        result.push_back({ points[i], timestamps[i], intensities[i], i, 0 });
    }
    return result;
}

namespace globals
{
    float rotate_x = 0.0, rotate_y = 0.0;
    float translate_x, translate_y = 0.0;
    float translate_z = -50.0;
    const unsigned int window_width = 800;
    const unsigned int window_height = 600;
    static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    std::string working_directory = "";
    int mouse_buttons = 0;
    int mouse_old_x, mouse_old_y;
    bool gui_mouse_down{ false };
    float mouse_sensitivity = 1.0;

    struct
    {
        double filter_threshold_xy = 0.0;
        double timestamp_per_icp = 0.05;
        kiss_icp::pipeline::KISSConfig icp_config;
        int decimation = 10;
    } params;

    std::mutex mtx;
    std::vector<RegisteredFrame> registeredFrames;
    std::vector<std::vector<Point3Di>> pointsPerFile;
    std::vector<Eigen::Vector3d> localMap;
    std::thread icpThread;
    std::atomic<bool> icpRunning{ false };
    std::atomic<float> icpProgress{ 0.0 };
} // namespace globals

void LoadDataButton()
{
    static std::shared_ptr<pfd::open_file> open_file;
    std::vector<std::string> input_file_names;
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, (bool)open_file);
    const auto t = [&]()
    {
        std::vector<std::string> filters;
        auto sel = pfd::open_file("Load las files", "C:\\", filters, true).result();
        for (int i = 0; i < sel.size(); i++)
        {
            input_file_names.push_back(sel[i]);
        }
    };
    std::thread t1(t);
    t1.join();

    std::sort(std::begin(input_file_names), std::end(input_file_names));

    std::vector<std::string> laz_files;

    std::for_each(
        std::begin(input_file_names),
        std::end(input_file_names),
        [&](const std::string& fileName)
        {
            if (fileName.ends_with(".laz") || fileName.ends_with(".las"))
            {
                laz_files.push_back(fileName);
            }
        });

    if (input_file_names.size() > 0)
    {
        globals::working_directory = fs::path(input_file_names.front()).parent_path().string();

        const auto calibrationFile = (fs::path(globals::working_directory) / "calibration.json").string();
        const auto preloadedCalibration = MLvxCalib::GetCalibrationFromFile(calibrationFile);
        const std::string imuSnToUse = MLvxCalib::GetImuSnToUse(calibrationFile);

        fs::path wdp = fs::path(input_file_names[0]).parent_path();
        wdp /= "preview";
        if (!fs::exists(wdp))
        {
            fs::create_directory(wdp);
        }
        globals::pointsPerFile.resize(laz_files.size());

        std::transform(
            std::execution::par_unseq,
            std::begin(laz_files),
            std::end(laz_files),
            std::begin(globals::pointsPerFile),
            [&](const std::string& fn)
            {
                // Load mapping from id to sn
                fs::path fnSn(fn);
                fnSn.replace_extension(".sn");

                // GetId of Imu to use
                const auto idToSn = MLvxCalib::GetIdToSnMapping(fnSn.string());
                auto calibration = MLvxCalib::CombineIntoCalibration(idToSn, preloadedCalibration);
                auto data = load_point_cloud(fn.c_str(), true, globals::params.filter_threshold_xy, calibration);

                if (fn == laz_files.front())
                {
                    fs::path calibrationValidtationFile = wdp / "calibrationValidation.asc";

                    std::ofstream testPointcloud{ calibrationValidtationFile.c_str() };
                    for (const auto& p : data)
                    {
                        testPointcloud << p.point.x() << "\t" << p.point.y() << "\t" << p.point.z() << "\t" << p.intensity << "\t"
                                       << (int)p.lidarid << "\n";
                    }
                }

                std::unique_lock lck(globals::mtx);
                for (const auto& [id, calib] : calibration)
                {
                    std::cout << " id : " << id << std::endl;
                    std::cout << calib.matrix() << std::endl;
                }
                return data;
                // std::cout << fn << std::endl;
                //
            });
        std::cout << "std::transform finished" << std::endl;
    }
    else
    {
        pfd::message("Error", "please select files correctly", pfd::choice::ok);
        std::cout << "please select files correctly" << std::endl;
    }
}

void IcpButton()
{
    if (globals::icpRunning)
    {
        return;
    }

    std::thread icpThread(
        []()
        {
            globals::icpRunning.store(true);
            using namespace kiss_icp::pipeline;
            KissICP icp(globals::params.icp_config);
            globals::registeredFrames.resize(1);

            double last_timestamp = 0;

            for (const auto& file : globals::pointsPerFile)
            {
                for (const auto& point : file)
                {
                    double timestamp = point.timestamp;
                    if (timestamp > 0)
                    {
                        if (last_timestamp == 0)
                        {
                            last_timestamp = timestamp;
                        }
                        auto& lastFrame = globals::registeredFrames.back();
                        lastFrame.points.emplace_back(point.point);
                        lastFrame.intensities.emplace_back(point.intensity);
                        double deltaTime = point.timestamp - last_timestamp;
                        lastFrame.timestamps.emplace_back(deltaTime);
                        if (deltaTime > globals::params.timestamp_per_icp)
                        {
                            last_timestamp = timestamp;
                            globals::registeredFrames.emplace_back();
                        }
                    }
                }
            }

            for (size_t i = 0; i < globals::registeredFrames.size(); ++i)
            {
                auto& frame = globals::registeredFrames[i];
                auto [registered_frame, registered_frame_timestamps] = icp.RegisterFrame(frame.points, frame.timestamps);
                std::unique_lock lck(globals::mtx);
                frame.pose = Eigen::Affine3d(icp.pose().matrix());
                globals::localMap = icp.LocalMap();
                globals::icpProgress.store((float)i / globals::registeredFrames.size());
            }
            globals::icpRunning.store(false);
        });
    icpThread.detach();
}

void SaveSession()
{
    std::vector<Eigen::Affine3d> poses;
    std::vector<std::string> lioLazFiles;
    auto concatframes = ConcatenateFrames(globals::registeredFrames);
    for (size_t i = 0; i < concatframes.size(); ++i)
    {
        auto& frame = concatframes[i];
        auto vecPoints = toPoint3DiV(frame.points, frame.intensities, frame.timestamps);
        const auto fn = ("scan_lio_" + std::to_string(i) + ".laz");
        lioLazFiles.push_back(fn);
        poses.push_back(frame.pose);
        const auto filename = fs::path(globals::working_directory) / fn;
        saveLaz(filename.string(), vecPoints);
    }
    const fs::path path(globals::working_directory);
    const fs::path pathReg = path / "lio_initial_poses.reg";
    const fs::path pathSession = path / "session.json";


    save_poses(pathReg.string(), poses, lioLazFiles);

    nlohmann::json jj;
    nlohmann::json j;
    j["offset_x"] = 0.0;
    j["offset_y"] = 0.0;
    j["offset_z"] = 0.0;
    j["folder_name"] = globals::working_directory;
    j["out_folder_name"] = globals::working_directory;
    j["poses_file_name"] = pathReg.string();
    j["initial_poses_file_name"] = path.string();
    j["out_poses_file_name"] = pathReg.string();
    jj["Session Settings"] = j;
    nlohmann::json jlaz_file_names;
    for (const auto& lioFileName : lioLazFiles)
    {
        auto filename = path / lioFileName;
        std::cout << "adding file: " << filename << std::endl;

        nlohmann::json jfn{
            {"file_name", filename.string()}};
        jlaz_file_names.push_back(jfn);
    }
    jj["laz_file_names"] = jlaz_file_names;
    std::ofstream o(pathSession);
    o << std::setw(4) << jj << std::endl;
}

void lidar_odometry_gui()
{
    if (ImGui::Begin("lidar_odometry_step_1-kiss-icp"))
    {
        ImGui::Text("This program is first step in MANDEYE process.");
        ImGui::Text("It results trajectory and point clouds as single session for "
                    "'multi_view_tls_registration_step_2' program.");
        ImGui::Text("It saves session.json file in 'Working directory'.");
        ImGui::Text("Next step will be to load session.json file with "
                    "'multi_view_tls_registration_step_2' program.");
        ImGui::SameLine();
        ImGui::Text("Select all imu *.csv and lidar *.laz files produced by "
                    "MANDEYE saved in 'continousScanning_*' folder");
        ImGui::Separator();
        if (ImGui::Button("load data"))
        {
            LoadDataButton();
        }

        if (ImGui::Button("icp"))
        {
            IcpButton();
        }
        if (ImGui::Button("save session"))
        {
            SaveSession();
        }
        if (globals::icpRunning)
        {
            ImGui::ProgressBar(globals::icpProgress);
        }

        ImGui::Separator();
        ImGui::Text("Parameters");
        ImGui::InputDouble("filter_threshold_xy", &globals::params.filter_threshold_xy);
        ImGui::InputDouble("timestamp_per_icp", &globals::params.timestamp_per_icp);
        ImGui::InputInt("decimation", &globals::params.decimation);
        ImGui::Separator();
        ImGui::Text("ICP Parameters");
        ImGui::InputDouble("voxel_size", &globals::params.icp_config.voxel_size);
        ImGui::InputDouble("max_range", &globals::params.icp_config.max_range);
        ImGui::InputDouble("min_range", &globals::params.icp_config.min_range);
        ImGui::InputInt("max_points_per_voxel", &globals::params.icp_config.max_points_per_voxel);
        ImGui::InputDouble("min_motion_th", &globals::params.icp_config.min_motion_th);
        ImGui::InputDouble("initial_threshold", &globals::params.icp_config.initial_threshold);
        ImGui::InputInt("max_num_iterations", &globals::params.icp_config.max_num_iterations);
        ImGui::InputDouble("convergence_criterion", &globals::params.icp_config.convergence_criterion);
        ImGui::InputInt("max_num_threads", &globals::params.icp_config.max_num_threads);
        ImGui::Checkbox("deskew", &globals::params.icp_config.deskew);
    }

    ImGui::End();
}

void mouse(int glut_button, int state, int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    int button = -1;
    if (glut_button == GLUT_LEFT_BUTTON)
        button = 0;
    if (glut_button == GLUT_RIGHT_BUTTON)
        button = 1;
    if (glut_button == GLUT_MIDDLE_BUTTON)
        button = 2;
    if (button != -1 && state == GLUT_DOWN)
        io.MouseDown[button] = true;
    if (button != -1 && state == GLUT_UP)
        io.MouseDown[button] = false;

    if (!io.WantCaptureMouse)
    {
        if (state == GLUT_DOWN)
        {
            globals::mouse_buttons |= 1 << glut_button;
        }
        else if (state == GLUT_UP)
        {
            globals::mouse_buttons = 0;
        }
        globals::mouse_old_x = x;
        globals::mouse_old_y = y;
    }
}

void wheel(int button, int dir, int x, int y)
{
    if (dir > 0)
    {
        globals::translate_z -= 0.05f * globals::translate_z;
    }
    else
    {
        globals::translate_z += 0.05f * globals::translate_z;
    }
    return;
}

void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.01, 10000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void motion(int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    using namespace globals;

    if (!io.WantCaptureMouse)
    {
        float dx, dy;
        dx = (float)(x - mouse_old_x);
        dy = (float)(y - mouse_old_y);

        gui_mouse_down = mouse_buttons > 0;
        if (mouse_buttons & 1)
        {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        }
        if (mouse_buttons & 4)
        {
            translate_x += dx * 0.5f * mouse_sensitivity;
            translate_y -= dy * 0.5f * mouse_sensitivity;
        }

        mouse_old_x = x;
        mouse_old_y = y;
    }
    glutPostRedisplay();
}

void coordinateSystem(float s)
{
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(s, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, s, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, s);
    glEnd();
}
void display()
{
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float ratio = float(io.DisplaySize.x) / float(io.DisplaySize.y);

    glClearColor(
        globals::clear_color.x * globals::clear_color.w,
        globals::clear_color.y * globals::clear_color.w,
        globals::clear_color.z * globals::clear_color.w,
        globals::clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    reshape((GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);

    {
        using namespace globals;
        glTranslatef(translate_x, translate_y, translate_z);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 0.0, 1.0);
    }

    coordinateSystem(100);

    {
        std::unique_lock lck(globals::mtx);
        for (const auto& frame : globals::registeredFrames)
        {
            glPushMatrix();
            glMultMatrixd(frame.pose.data());
            coordinateSystem(1);
            glPopMatrix();
        }
    }
    glBegin(GL_POINTS);
    std::vector<Eigen::Vector3d> points;
    {
        std::unique_lock lck(globals::mtx);
        points = globals::localMap;
    }
    for (const auto& point : points)
    {
        glVertex3dv(point.data());
    }

    glEnd();
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    lidar_odometry_gui();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    glutSwapBuffers();
    glutPostRedisplay();
}

bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(globals::window_width, globals::window_height);
    glutCreateWindow("lidar_odometry");
    glutDisplayFunc(display);
    glutMotionFunc(motion);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, globals::window_width, globals::window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)globals::window_width / (GLfloat)globals::window_height, 0.01, 10000.0);
    glutReshapeFunc(reshape);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
    // Keyboard Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
    return true;
}

int main(int argc, char* argv[])
{
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(wheel);
    glutMainLoop();

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGLUT_Shutdown();

    ImGui::DestroyContext();
    return 0;
}
