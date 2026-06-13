cmake_minimum_required(VERSION 3.16)
project(MedicalImageProcessor VERSION 2.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE tooling (clangd, IntelliSense, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Threads ──────────────────────────────────────────────────────────────────
# Required for std::async / std::thread on Linux (pthreads). No-op on Windows/MSVC.
find_package(Threads REQUIRED)

# ─── Common compile options ───────────────────────────────────────────────────

if (MSVC)
    set(MEDIMG_WARN_FLAGS /W4 /WX-)
    set(MEDIMG_OPT_FLAGS  /O2)
else()
    set(MEDIMG_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(MEDIMG_OPT_FLAGS  -O2)
endif()

# ─── Main Executable ──────────────────────────────────────────────────────────

add_executable(medimg
    src/main.cpp
    src/Filter.cpp
    src/ImageIO.cpp
    src/Pipeline.cpp
)

target_include_directories(medimg PRIVATE include)
target_compile_options(medimg PRIVATE ${MEDIMG_WARN_FLAGS} ${MEDIMG_OPT_FLAGS})
target_link_libraries(medimg PRIVATE Threads::Threads)

# ─── Test Executable ──────────────────────────────────────────────────────────

add_executable(medimg_tests
    tests/test_filters.cpp
    src/Filter.cpp
    src/ImageIO.cpp
    src/Pipeline.cpp
)

target_include_directories(medimg_tests PRIVATE include)
target_compile_options(medimg_tests PRIVATE ${MEDIMG_WARN_FLAGS} ${MEDIMG_OPT_FLAGS})
target_compile_definitions(medimg_tests PRIVATE MEDIMG_DEBUG)
target_link_libraries(medimg_tests PRIVATE Threads::Threads)

# ─── Enable CTest ─────────────────────────────────────────────────────────────

enable_testing()
add_test(NAME UnitTests COMMAND medimg_tests)

# ─── Install Rules ────────────────────────────────────────────────────────────

install(TARGETS medimg DESTINATION bin)
install(DIRECTORY samples/ DESTINATION share/medimg/samples)

# ─── Print build info ─────────────────────────────────────────────────────────

message(STATUS "Medical Image Processor v${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
