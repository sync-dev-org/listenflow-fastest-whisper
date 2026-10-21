#include <nvtx3/nvToolsExt.h>

extern "C" void issue40_push(const char* name) { nvtxRangePushA(name); }
extern "C" void issue40_pop() { nvtxRangePop(); }
