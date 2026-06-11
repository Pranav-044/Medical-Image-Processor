cmake_minimum_required(VERSION 3.16)
project(MedicalImageProcessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Main Executable ──────────────────────────────────────────────────────────

add_executable(medimg
    src/main.cpp
    src/Filter.cpp
    src/ImageIO.cpp
    src/Pipeline.cpp
)

target_include_directories(medimg PRIVATE include)

if (MSVC)
    target_compile_options(medimg PRIVATE /W4 /O2)
else()
    target_compile_options(medimg PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O2
    )
endif()

# Link filesystem library (required on older GCC/Clang builds)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(medimg PRIVATE stdc++fs)
endif()

# ─── Test Executable ──────────────────────────────────────────────────────────

add_executable(medimg_tests
    tests/test_filters.cpp
    src/Filter.cpp
    src/ImageIO.cpp
    src/Pipeline.cpp
)

target_include_directories(medimg_tests PRIVATE include)

if (MSVC)
    target_compile_options(medimg_tests PRIVATE /W4 /O2 -DMEDIMG_DEBUG)
else()
    target_compile_options(medimg_tests PRIVATE -Wall -Wextra -O2 -DMEDIMG_DEBUG)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(medimg_tests PRIVATE stdc++fs)
endif()

# ─── Enable CTest ─────────────────────────────────────────────────────────────
enable_testing()
add_test(NAME UnitTests COMMAND medimg_tests)

# ─── Print build info ─────────────────────────────────────────────────────────
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
