cmake_minimum_required(VERSION 3.22)
project(HeartbeatMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 0. Toolchain sanity (Windows) ---
if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR "This project requires the MSVC toolchain on Windows (vcpkg binaries are MSVC). Configure CLion to use the Visual Studio toolchain.")
endif()

# If no prefix path is provided (common in CLion), fall back to the repo-local vcpkg install.
if(WIN32 AND NOT CMAKE_PREFIX_PATH)
    set(_vcpkg_debug "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_debug/x64-windows")
    set(_vcpkg_release "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_release/x64-windows")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND EXISTS "${_vcpkg_debug}")
        list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_debug}")
    elseif(EXISTS "${_vcpkg_release}")
        list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_release}")
    elseif(EXISTS "${_vcpkg_debug}")
        list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_debug}")
    endif()
endif()

# Help FindLAPACK/FindBLAS on Windows when using repo-local vcpkg.
if(WIN32)
    set(_openblas_debug "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_debug/x64-windows/lib/openblas.lib")
    set(_openblas_release "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_release/x64-windows/lib/openblas.lib")
    if(NOT DEFINED BLA_VENDOR)
        set(BLA_VENDOR "OpenBLAS" CACHE STRING "" FORCE)
    endif()
    if((NOT BLAS_LIBRARIES OR BLAS_LIBRARIES STREQUAL "BLAS_LIBRARIES-NOTFOUND"))
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND EXISTS "${_openblas_debug}")
            set(BLAS_LIBRARIES "${_openblas_debug}" CACHE FILEPATH "" FORCE)
        elseif(EXISTS "${_openblas_release}")
            set(BLAS_LIBRARIES "${_openblas_release}" CACHE FILEPATH "" FORCE)
        elseif(EXISTS "${_openblas_debug}")
            set(BLAS_LIBRARIES "${_openblas_debug}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
    if((NOT LAPACK_LIBRARIES OR LAPACK_LIBRARIES STREQUAL "LAPACK_LIBRARIES-NOTFOUND"))
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND EXISTS "${_openblas_debug}")
            set(LAPACK_LIBRARIES "${_openblas_debug}" CACHE FILEPATH "" FORCE)
        elseif(EXISTS "${_openblas_release}")
            set(LAPACK_LIBRARIES "${_openblas_release}" CACHE FILEPATH "" FORCE)
        elseif(EXISTS "${_openblas_debug}")
            set(LAPACK_LIBRARIES "${_openblas_debug}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
    if(BLAS_LIBRARIES)
        set(BLAS_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
    if(LAPACK_LIBRARIES)
        set(LAPACK_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
endif()

# --- 1. DLib: Find via vcpkg (Cross-Platform) ---
find_package(dlib CONFIG REQUIRED)
message(STATUS "Found DLib: ${DLib_VERSION}")

# --- 2. OpenCV: Find via vcpkg (Cross-Platform) ---
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# --- 2.5. YAML-CPP: Find via vcpkg ---
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp")

# --- 2.6. SPDLOG: Find via vcpkg ---
find_package(spdlog REQUIRED)
message(STATUS "Found spdlog")

# --- 3. Model Download & Extraction (Universal) ---
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dlib_models")
set(MODEL_FILE "${MODEL_DIR}/shape_predictor_68_face_landmarks.dat")
set(MODEL_URL "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

if(NOT EXISTS "${MODEL_FILE}")
    file(MAKE_DIRECTORY "${MODEL_DIR}")
    message(STATUS "Downloading landmark model...")
    file(DOWNLOAD "${MODEL_URL}" "${MODEL_FILE}.bz2" SHOW_PROGRESS)
    
    message(STATUS "Extracting model...")
    if(WIN32)
        # Windows: Try CMake's tar; fallback to manual if user has no bzip2 lib
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${MODEL_FILE}.bz2" WORKING_DIRECTORY "${MODEL_DIR}")
    else()
        # Linux: Use the native bunzip2 for speed and reliability
        execute_process(COMMAND bunzip2 "${MODEL_FILE}.bz2")
    endif()
endif()

# --- 4. Target Definition ---
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/FaceProcessor.cpp 
    src/HeartbeatAnalyzer.cpp
    src/Config.cpp
    src/Overlay.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS} 
    dlib::dlib
    yaml-cpp::yaml-cpp
    spdlog::spdlog
)
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE gdi32 user32) # For Win32 HUD
endif()
file(TO_NATIVE_PATH "${MODEL_FILE}" NATIVE_PATH)
string(REPLACE "\\" "\\\\" ESCAPED_PATH "${NATIVE_PATH}")
target_compile_definitions(${PROJECT_NAME} PRIVATE MODEL_PATH="${ESCAPED_PATH}")

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3)
endif()

# --- 5. Post-Build: Copy DLLs and config.yml ---
# Determine the output directory based on generator type
if(MSVC)
    # Multi-config generator: use $<CONFIG> for per-config folders
    set(OUTPUT_DIR "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    set(CONFIG_DIR "${OUTPUT_DIR}/..")  # One level up to match structure
else()
    set(OUTPUT_DIR "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
endif()

# Copy DLLs from vcpkg bin directory
if(WIN32)
    set(_vcpkg_bin_dir "$<IF:$<CONFIG:Debug>,${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_debug/x64-windows/bin,${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed_release/x64-windows/bin>")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_vcpkg_bin_dir}"
            "${OUTPUT_DIR}"
        COMMENT "Copying vcpkg DLLs to output directory"
    )
endif()

# Copy config.yml to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
        "${OUTPUT_DIR}/config.yaml"
    COMMENT "Copying config.yaml to output directory"
)
