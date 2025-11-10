add_executable(ntt_sfpu ${CMAKE_CURRENT_SOURCE_DIR}/ntt_sfpu.cpp)
target_link_libraries(ntt_sfpu PRIVATE TT::Metalium)
target_include_directories(ntt_sfpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)