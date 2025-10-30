add_executable(ntt_naive ${CMAKE_CURRENT_SOURCE_DIR}/ntt_naive.cpp)
target_link_libraries(ntt_naive PRIVATE TT::Metalium)
target_include_directories(ntt_naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)