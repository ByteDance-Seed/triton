add_triton_library(TritonGPUToCUDA
    TritonGPUToCUDA.cpp
    CUDACodeGen.cpp

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUIR
    TritonNvidiaGPUIR
    NVGPUIR
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRGPUDialect
)
