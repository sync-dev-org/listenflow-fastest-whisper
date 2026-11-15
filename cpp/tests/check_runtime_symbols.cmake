# The reviewed C ABI is the complete dynamic definition surface.
execute_process(COMMAND "${NM}" -D --defined-only "${LIBRARY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE diagnostic)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect runtime symbols: ${diagnostic}")
endif()
set(expected
    lffw_get_version lffw_model_open lffw_model_get_info lffw_model_close lffw_model_destroy
    lffw_mel_compute lffw_mel_copy lffw_log_mel_block lffw_mel_destroy
    lffw_encode lffw_encode_features lffw_encoded_copy lffw_encoded_destroy
    lffw_decoder_create lffw_prefix_step lffw_greedy_step lffw_sampled_step
    lffw_prefix_destroy lffw_decoder_destroy lffw_error_message lffw_error_destroy)
if(DIAGNOSTICS)
    foreach(name IN ITEMS create begin end get_counts get_operation get_span get_sleep destroy)
        list(APPEND expected "lffw_diagnostic_${name}")
    endforeach()
endif()
string(REPLACE "\n" ";" lines "${symbols}")
set(actual)
foreach(line IN LISTS lines)
    if(line STREQUAL "")
        continue()
    endif()
    if(NOT line MATCHES "^[0-9a-fA-F]+ T (lffw_[a-z_]+)$")
        message(FATAL_ERROR "Unexpected exported runtime definition: ${line}")
    endif()
    list(APPEND actual "${CMAKE_MATCH_1}")
endforeach()
list(SORT actual)
list(SORT expected)
if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "Runtime C ABI differs: actual=${actual}; expected=${expected}")
endif()
