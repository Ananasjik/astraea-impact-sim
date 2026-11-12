find_package(ZLIB REQUIRED)

add_library(sim_server_lib STATIC
    ws_crypto.cpp
    json_codec.cpp
    binary_protocol.cpp
    http_ws_server.cpp
)

target_include_directories(sim_server_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(sim_server_lib PRIVATE
    -Wall
    -Wextra
)

target_link_libraries(sim_server_lib PUBLIC
    sim_engine
    ZLIB::ZLIB
    Threads::Threads
)

add_executable(sim_server
    sim_server_main.cpp
)

set_target_properties(sim_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_compile_options(sim_server PRIVATE
    -Wall
    -Wextra
)

target_link_libraries(sim_server PRIVATE
    sim_server_lib
)
