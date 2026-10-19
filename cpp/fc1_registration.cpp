#include <NvInferRuntime.h>

nvinfer1::IPluginCreatorInterface* fc1_creator() noexcept;

extern "C" __attribute__((visibility("default")))
void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}

extern "C" __attribute__((visibility("default")))
nvinfer1::IPluginCreatorInterface* const* getCreators(std::int32_t& count) noexcept
{
    static nvinfer1::IPluginCreatorInterface* creators[] = {fc1_creator()};
    count = 1;
    return creators;
}
