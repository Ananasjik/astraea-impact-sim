find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

get_filename_component(ASTRAEA_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../engine" ABSOLUTE)

add_executable(astraea_tests
    test_eos.cpp
    test_planet_builder.cpp
    test_gravity_sph.cpp
    test_conservation.cpp
)

set_target_properties(astraea_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_features(astraea_tests PRIVATE cxx_std_20)
target_include_directories(astraea_tests PRIVATE "${ASTRAEA_ENGINE_DIR}")
target_compile_definitions(astraea_tests PRIVATE
    ASTRAEA_BIN_DIR="$<TARGET_FILE_DIR:sim_headless>"
)

target_link_libraries(astraea_tests PRIVATE
    sim_engine
    CUDA::cudart
    GTest::gtest_main
)

gtest_discover_tests(astraea_tests
    DISCOVERY_TIMEOUT 30
)

if(TARGET sim_server_lib)
    find_package(ZLIB REQUIRED)
    find_package(Threads REQUIRED)

    get_filename_component(ASTRAEA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

    add_executable(astraea_server_tests
        test_server_protocol.cpp
    )

    set_target_properties(astraea_server_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    target_compile_features(astraea_server_tests PRIVATE cxx_std_20)
    target_include_directories(astraea_server_tests PRIVATE
        "${ASTRAEA_REPO_ROOT}"
        "${ASTRAEA_ENGINE_DIR}"
    )

    if(TARGET sim_server)
        add_dependencies(astraea_server_tests sim_server)
        target_compile_definitions(astraea_server_tests PRIVATE
            ASTRAEA_SERVER_BIN="$<TARGET_FILE:sim_server>"
        )
    else()
        target_compile_definitions(astraea_server_tests PRIVATE
            ASTRAEA_SERVER_BIN=""
        )
    endif()

    target_link_libraries(astraea_server_tests PRIVATE
        sim_server_lib
        sim_engine
        CUDA::cudart
        ZLIB::ZLIB
        GTest::gtest_main
        Threads::Threads
    )

    gtest_discover_tests(astraea_server_tests
        DISCOVERY_TIMEOUT 30
    )
endif()
