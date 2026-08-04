cmake_minimum_required(VERSION 3.19)
project(RX_H265 LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets SerialPort)

qt_standard_project_setup()

# Remove mainwindow.ui since we build the UI purely in code now.
qt_add_executable(RX_H265
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

target_link_libraries(RX_H265
    PRIVATE
        Qt::Core
        Qt::Gui
        Qt::Widgets
        Qt::SerialPort
)

include(GNUInstallDirs)

install(TARGETS RX_H265
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET RX_H265
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
