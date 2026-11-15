#include <listenflow/whisper_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* SP-LFFASTESTWHISPER-031, SP-LFFASTESTWHISPER-033: C callers can inspect
 * the ABI and reject malformed calls without initializing a GPU. */
int main(void) {
    lffw_version version = {sizeof(version), 0, 0, 0};
    lffw_error *error = NULL;
    if (lffw_get_version(&version, &error) || error || version.major != LFFW_ABI_MAJOR)
        return 1;
    if (lffw_get_version(NULL, &error) != LFFW_INVALID_ARGUMENT || !error)
        return 2;
    const char *message = lffw_error_message(error);
    if (!message || !strstr(message, "version"))
        return 3;
    /* A later call must not overwrite an independently owned diagnostic. */
    if (lffw_model_close(NULL, NULL) != LFFW_INVALID_ARGUMENT || !strstr(message, "version"))
        return 4;
    lffw_error_destroy(error);
    error = NULL;
    lffw_model *model = (lffw_model *)1;
    lffw_model_options options = {0};
    options.struct_size = sizeof(options);
    options.abi_major = LFFW_ABI_MAJOR + 1;
    if (lffw_model_open(&options, &model, &error) != LFFW_ABI_MISMATCH || model || !error)
        return 5;
    lffw_error_destroy(error);
    if (lffw_model_destroy(NULL, NULL) || lffw_mel_destroy(NULL, NULL) ||
        lffw_encoded_destroy(NULL, NULL) || lffw_decoder_destroy(NULL, NULL) ||
        lffw_prefix_destroy(NULL, NULL))
        return 6;
    return 0;
}
