cmake_minimum_required(VERSION 3.22)
project(HeartbeatMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. DLib: Find via vcpkg (Cross-Platform) ---
find_package(dlib CONFIG REQUIRED)
message(STATUS "Found DLib: ${DLib_VERSION}")

# --- 2. OpenCV: Find via vcpkg (Cross-Platform) ---
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# --- 2.5. YAML-CPP: Find via vcpkg ---
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp")

# --- 3. Model Download & Extraction (Universal) ---
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dlib_models")
set(MODEL_FILE "${MODEL_DIR}/shape_predictor_68_face_landmarks_GTX.dat")
set(MODEL_URL "http://dlib.net/files/shape_predictor_68_face_landmarks_GTX.dat.bz2")

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
    gdi32 user32 # For Win32 HUD
)
file(TO_NATIVE_PATH "${MODEL_FILE}" NATIVE_PATH)
string(REPLACE "\\" "\\\\" ESCAPED_PATH "${NATIVE_PATH}")
target_compile_definitions(${PROJECT_NAME} PRIVATE MODEL_PATH="${ESCAPED_PATH}")

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3)
endif()