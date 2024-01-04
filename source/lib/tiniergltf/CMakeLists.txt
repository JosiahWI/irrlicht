# CMakeLists of tinygltf, adapted for tiniergltf.

cmake_minimum_required(VERSION 3.12)

project(tiniergltf
  VERSION 1.0.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(jsoncpp REQUIRED)

add_library(tiniergltf OBJECT tiniergltf.hpp base64.cpp base64.h)
add_library(tiniergltf::tiniergltf ALIAS tiniergltf)

target_include_directories(tiniergltf
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:./>"
)

target_link_libraries(tiniergltf jsoncpp_lib)

include(GNUInstallDirs)

install(TARGETS tiniergltf
  EXPORT tiniergltf-export
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

export(EXPORT tiniergltf-export
  FILE "${CMAKE_BINARY_DIR}/cmake/tiniergltfTargets.cmake"
  NAMESPACE tiniergltf::
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tiniergltf"
)

install(EXPORT tiniergltf-export
  FILE tiniergltfTargets.cmake
  NAMESPACE tiniergltf::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tiniergltf"
)

include(CMakePackageConfigHelpers)
configure_package_config_file("${PROJECT_SOURCE_DIR}/tiniergltfConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/cmake/tiniergltfConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tiniergltf"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/cmake/tiniergltfConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
)

install(
  FILES
    "${CMAKE_BINARY_DIR}/cmake/tiniergltfConfig.cmake"
    "${CMAKE_BINARY_DIR}/cmake/tiniergltfConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tiniergltf"
)