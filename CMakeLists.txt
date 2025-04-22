cmake_minimum_required(VERSION 3.5.0)
project(multivox VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

SET(MULTIVOX_INSTALL_DIR $ENV{HOME}/Multivox)
SET(CMAKE_INSTALL_PREFIX ${MULTIVOX_INSTALL_DIR})

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(DRIVER_SRC_DIR ${SRC_DIR}/driver)
set(PLATFORM_SRC_DIR ${SRC_DIR}/platform)
set(MULTIVOX_SRC_DIR ${SRC_DIR}/multivox)

set(BUILD_DIR ${CMAKE_BINARY_DIR})


function(set_gadget GADGET_NAME)
    set(HEADER_PATH "${DRIVER_SRC_DIR}/gadgets/gadget_${GADGET_NAME}.h")
    set(OUTPUT_HEADER "${BUILD_DIR}/generated/gadget.h")
    if(EXISTS ${HEADER_PATH})
        message(STATUS "Using gadget header: ${HEADER_PATH}")
        configure_file(${HEADER_PATH} ${OUTPUT_HEADER} COPYONLY)
    else()
        message(FATAL_ERROR "Gadget header ${HEADER_PATH} does not exist.")
    endif()
endfunction()

set(MULTIVOX_GADGET "vortex" CACHE STRING "Gadget-specific header file")
set_gadget(${MULTIVOX_GADGET})


include_directories(
    ${DRIVER_SRC_DIR}
    ${PLATFORM_SRC_DIR}
    ${BUILD_DIR}/generated
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMATHC_USE_UNIONS -O3 -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DMATHC_USE_UNIONS -Og -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DMATHC_USE_UNIONS -O3 -Wall")

function(add_cart cart_name cart_args)
    install(CODE "
        if(NOT EXISTS \"${MULTIVOX_INSTALL_DIR}/carts/${cart_name}.mct\")
            file(WRITE \"${MULTIVOX_INSTALL_DIR}/carts/${cart_name}.mct\" \"colour=#00EFFF\n\")
            file(APPEND \"${MULTIVOX_INSTALL_DIR}/carts/${cart_name}.mct\" \"command=${CMAKE_INSTALL_PREFIX}/bin/${cart_name}\n\")
            file(APPEND \"${MULTIVOX_INSTALL_DIR}/carts/${cart_name}.mct\" \"arguments=${cart_args}\n\")
        endif()
    ")
endfunction()

file(GLOB DRIVER_SRC ${DRIVER_SRC_DIR}/*.c)
add_executable(vortex
    ${DRIVER_SRC}
    ${PLATFORM_SRC_DIR}/mathc.c
    ${PLATFORM_SRC_DIR}/input.c
)
target_link_libraries(vortex PRIVATE m rt pthread)
install(TARGETS vortex)

file(GLOB PLATFORM_SRC ${PLATFORM_SRC_DIR}/*.c)
add_library(platform STATIC ${PLATFORM_SRC})

file(GLOB MULTIVOX_SRC ${MULTIVOX_SRC_DIR}/*.c)
add_executable(multivox
    ${MULTIVOX_SRC}
)
target_link_libraries(multivox PRIVATE platform m)
install(TARGETS multivox)

function(add_toy toy)
    if(EXISTS ${SRC_DIR}/toys/${toy}.c)
        add_executable(${toy} ${SRC_DIR}/toys/${toy}.c)
    else()
        file(GLOB TOY_SRC "${SRC_DIR}/toys/${toy}/*.c")
        add_executable(${toy} ${TOY_SRC})
    endif()
    target_link_libraries(${toy} PRIVATE platform m)
    install(TARGETS ${toy})
    add_cart(${toy} "")
endfunction()

add_cart(viewer "${MULTIVOX_INSTALL_DIR}/models/*.obj ${MULTIVOX_INSTALL_DIR}/images/*.png")

add_toy(viewer)
add_toy(zander)
add_toy(eighty)
add_toy(tesseract)
add_toy(flight)
add_toy(fireworks)
add_toy(brain)
add_toy(brain2)
add_toy(combined_scene)


install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/ DESTINATION ${MULTIVOX_INSTALL_DIR}/models)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/images/ DESTINATION ${MULTIVOX_INSTALL_DIR}/images FILES_MATCHING PATTERN "*.png")


set(SIMULATOR_SRC_DIR ${SRC_DIR}/simulator)
file(GLOB VIRTEX_SRC ${SIMULATOR_SRC_DIR}/*.c)

set(GLSL_DIRECTORY "${SIMULATOR_SRC_DIR}")
file(GLOB GLSL_FILES "${GLSL_DIRECTORY}/*.glsl")

set(GLSL_HEADERS)
foreach(GLSL_FILE ${GLSL_FILES})
    get_filename_component(BASENAME "${GLSL_FILE}" NAME_WE)
    set(GLSL_HEADER "${BUILD_DIR}/generated/${BASENAME}_glsl.h")

    add_custom_command(
        OUTPUT "${GLSL_HEADER}"
        COMMAND ${CMAKE_COMMAND} -DINPUT_FILE="${GLSL_FILE}" -DOUTPUT_FILE="${GLSL_HEADER}" -P ${SIMULATOR_SRC_DIR}/glsl.cmake
        DEPENDS "${GLSL_FILE}"
        COMMENT "Wrapping ${GLSL_FILE} to ${GLSL_HEADER}"
    )
    
    list(APPEND GLSL_HEADERS ${GLSL_HEADER})
endforeach()

add_library(glsl_headers INTERFACE)
target_sources(glsl_headers INTERFACE ${GLSL_HEADERS})

add_executable(virtex
    ${VIRTEX_SRC}
    ${DRIVER_SRC_DIR}/slicemap.c
)
target_link_libraries(virtex PRIVATE platform glsl_headers X11 EGL GLESv2 m)
