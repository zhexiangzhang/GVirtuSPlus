//#include <cudnn.h>
//#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "cudarpc.pb.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cublas_v2.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cudnn.h"
#include </usr/local/cuda-11.4/targets/x86_64-linux/include/cuda_runtime.h>
// 11
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUDNN(call) { \
    const cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudnnGetErrorString(status) << std::endl; \
        exit(1); \
    } \
}


__global__ void emptyKernel() {
    // do nothing
}

cudnnHandle_t handle;

// for forward
cudnnTensorDescriptor_t input_descriptor, output_descriptor;
cudnnFilterDescriptor_t filter_descriptor;
cudnnConvolutionDescriptor_t convolution_descriptor;

// for batch normalization forward
cudnnTensorDescriptor_t xDesc_batch, yDesc_batch, bnScaleBiasMeanVarDesc_batch;

cublasHandle_t cublas_handle_;

// for single func
cudnnTensorDescriptor_t t1;
cudnnFilterDescriptor_t f1;
cudnnConvolutionDescriptor_t tc1;
float *space1;

cudaStream_t stream;

float *d_memory;
size_t totalSize = 5L * 1024 * 1024 * 1024; // 10GB
float *d_output;

std::vector<cudnnTensorDescriptor_t> tensorDescriptorsVector;
std::vector<cudnnFilterDescriptor_t> filterDescriptorsVector;
std::vector<cudnnConvolutionDescriptor_t> convolutionDescriptorsVector;

int n, c, h, w;
int batch_size = 1, channels = 3, height = 1024, width = 1024;
int out_channels = 16, kernel_height = 5, kernel_width = 5;

const int dimA[3] = {1, 1, 1};
const int strideA[3] = {3 * 2, 2, 1};

float alpha = 1.0f, beta = 0.0f;
size_t workspace_bytes = 0;

float A[15], B[12], C[20];

/*
only for cublasSgemmStridedBatchedService function
======================================================================
*/   
const int cublasBatch = 5; 

// 在主机上创建临时数组
const float* h_Aarray[cublasBatch];
float* h_Carray[cublasBatch];

// 在设备上为指针数组分配空间
const float** cublasAarry;
float** cublasCarry;

void initForcublasSgemmStridedBatchedService() {    
    cudaMalloc(&cublasAarry, cublasBatch * sizeof(float*));    
    cudaMalloc(&cublasCarry, cublasBatch * sizeof(float*));

    for (int i = 0; i < cublasBatch; ++i) {
        h_Aarray[i] = d_memory;        
        // h_Barray[i] = d_memory + cublasBatch * m * k + i * k * n;
        h_Carray[i] = d_memory;
    }

    cudaMemcpy(cublasAarry, h_Aarray, cublasBatch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(cublasCarry, h_Carray, cublasBatch * sizeof(float*), cudaMemcpyHostToDevice);
}

//======================================================================

void initGlobalVar(){
    try {
        CHECK_CUDNN(cudnnCreate(&handle));

    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }

    // memory
    CHECK_CUDA(cudaMalloc(&d_memory, totalSize));
    CHECK_CUDA(cudaMalloc(&d_output, totalSize));
    cudaDeviceSynchronize();

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&tc1));
    cudaStreamCreate(&stream);  
}

void destoryinitGlobalVar(){
    cudnnDestroy(handle);
    CHECK_CUDA(cudaFree(d_memory));
}

void initDescriptor(){
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, channels, kernel_height, kernel_width));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &n, &c, &h, &w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    //==========================
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc_batch));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc_batch));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc_batch));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc_batch, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 32, 32));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc_batch, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 32, 32));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc_batch, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 1, 1));
}

void destoryDescriptor(){
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

// ===========================================================fv
//
void cublasSetStreamService() { cublasSetStream(cublas_handle_, 0);}
void cublasSetMathModeService() { cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);}
void cublasSgemmService() {
    cublasSgemm(cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                500, 400, 300,
                &alpha, d_memory, 500, d_memory, 300, &beta, d_memory, 500);}

void cublasSgemmStridedBatchedService() {
    cublasSgemmBatched(cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        5, 4, 3,
                        &alpha,
                        cublasAarry, 5,
                        cublasAarry, 3,
                        &beta,
                        cublasCarry, 5,
                        cublasBatch);                  
}
void cudaStreamSynchronizeService() { cudaStreamSynchronize(0);}

// void cudaStreamIsCapturingService() {
//     cudaStreamCaptureStatus captureStatus;
//     // cudaStreamIsCapturing(stream, &captureStatus);
//     cudaStreamIsCapturing(0, &captureStatus);
// }

void cudaMemcpyAsyncService() {
    // from src --copy-> count bbyte to dst。
    // cudaStream_t 类型的参数来指定操作关联的流。
    const int copy_size = 4096;
    float* h_src = new float[copy_size]; // host
    for (int i = 0; i < copy_size; i++) {
        h_src[i] = static_cast<float>(i);
    }

    size_t count = copy_size * sizeof(float);  
    cudaMemcpyAsync(d_output, h_src, count, cudaMemcpyHostToDevice, 0); // 最后是流
}

void cublasCreateService(){
    cublasCreate(&cublas_handle_);
}

// --------------------------------------------------------------
void cudaGetLastErrorService() {cudaGetLastError();}  // need to add cudacheck
void cudaDeviceSynchronizeService() {cudaDeviceSynchronize();}
void cudaMallocService() { CHECK_CUDA(cudaMalloc(&space1, 0));}
// cudnn --------------------------
void cudaLaunchKernelService() {
    dim3 blockDim(256);
    dim3 gridDim((1024 + blockDim.x - 1) / blockDim.x);
    void *args[] = {};
    cudaLaunchKernel((const void*)emptyKernel, gridDim, blockDim, args, 0, 0);
}
void cudnnCreateService() { CHECK_CUDNN(cudnnCreate(&handle));}
void cudnnSetConvolutionGroupCountService() { CHECK_CUDNN(cudnnSetConvolutionGroupCount(tc1, 1));}
void cudnnSetConvolutionMathTypeService() { CHECK_CUDNN(cudnnSetConvolutionMathType(tc1, CUDNN_TENSOR_OP_MATH)); }
void cudnnSetFilterNdDescriptorService() { CHECK_CUDNN(cudnnSetFilterNdDescriptor(f1, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimA));}
void cudnnSetTensorNdDescriptorService() { CHECK_CUDNN(cudnnSetTensorNdDescriptor(t1, CUDNN_DATA_FLOAT, 3, dimA, strideA));}
void cudnnSetStreamService() { CHECK_CUDNN(cudnnSetStream(handle, 0));}
void cudnnCreateTensorDescriptorService(){
    cudnnTensorDescriptor_t tempTensor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tempTensor));
    tensorDescriptorsVector.push_back(tempTensor);
}
void cudnnCreateFilterDescriptorService(){
    cudnnFilterDescriptor_t tempFilter;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&tempFilter));
    filterDescriptorsVector.push_back(tempFilter);
}
void cudnnCreateConvolutionDescriptorService() {
    cudnnConvolutionDescriptor_t tempConvolution;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&tempConvolution));
    convolutionDescriptorsVector.push_back(tempConvolution);
}
void cudnnSetConvolutionNdDescriptorService() { CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(tc1, 3, dimA, dimA, dimA, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));}
void cudnnDestroyTensorDescriptorService() {
    if (!tensorDescriptorsVector.empty()) {
        cudnnTensorDescriptor_t tensorDescriptor = tensorDescriptorsVector.back();
        tensorDescriptorsVector.pop_back();
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDescriptor));
    } else {
        std::cerr << "Error: No tensor descriptor to destroy." << std::endl;
        //    CHECK_CUDNN(cudnnDestroyTensorDescriptor(t1));}
    }
}
void cudnnDestroyFilterDescriptorService() {
    if (!filterDescriptorsVector.empty()) {
        cudnnFilterDescriptor_t filterDescriptor = filterDescriptorsVector.back();
        filterDescriptorsVector.pop_back();
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));
    } else {
        std::cerr << "Error: No filter descriptor to destroy." << std::endl;
        //    CHECK_CUDNN(cudnnDestroyTensorDescriptor(t1));}
    }
}
void cudnnDestroyConvolutionDescriptorService() {
    if (!convolutionDescriptorsVector.empty()) {
        cudnnConvolutionDescriptor_t convolutionDescriptor = convolutionDescriptorsVector.back();
        convolutionDescriptorsVector.pop_back();
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
    } else {
        std::cerr << "Error: No convolution descriptor to destroy." << std::endl;
        //    CHECK_CUDNN(cudnnDestroyTensorDescriptor(t1));}
    }
}
void cudnnConvolutionForwardService(){
    CHECK_CUDNN(cudnnConvolutionForward(
            handle, &alpha,
            input_descriptor, d_memory,
            filter_descriptor, d_memory,
            convolution_descriptor,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            d_memory, workspace_bytes,
            &beta,
            output_descriptor, d_output));
}
void cudnnBatchNormalizationForwardInferenceService() {
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            handle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            xDesc_batch, d_memory,
            yDesc_batch, d_memory,
            bnScaleBiasMeanVarDesc_batch,
            d_memory, d_memory,
            d_memory, d_memory, 1));
}
void cudnnGetConvolutionForwardAlgorithm_v7Service() {
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                                       input_descriptor, filter_descriptor,
                                                       convolution_descriptor,output_descriptor,
                                                       1, &returnedAlgoCount, &algoPerf));
}

int inValid = 0;
void parse(const cudarpc::QueryType &type) {
    switch (type) {
//        case cudarpc::QueryType::cudnnSetConvolutionNdDescriptor:
//            cudnnSetConvolutionNdDescriptorService();
//            break;
        case cudarpc::QueryType::cudnnCreate:
            cudnnCreateService();
            break;
//        case cudarpc::QueryType::cudnnSetTensorNdDescriptor:
//            cudnnSetTensorNdDescriptorService();
//            break;
//        case cudarpc::QueryType::cudnnSetFilterNdDescriptor:
//            cudnnSetFilterNdDescriptorService();
//            break;
        case cudarpc::QueryType::cudnnSetStream:
            cudnnSetStreamService();
            break;
        case cudarpc::QueryType::cudnnConvolutionForward:
            cudnnConvolutionForwardService();
            break;
        case cudarpc::QueryType::cudnnCreateTensorDescriptor:
            cudnnCreateTensorDescriptorService();
            break;
        case cudarpc::QueryType::cudnnCreateFilterDescriptor:
            cudnnCreateFilterDescriptorService();
            break;
        case cudarpc::QueryType::cudnnCreateConvolutionDescriptor:
            cudnnCreateConvolutionDescriptorService();
            break;
        case cudarpc::QueryType::cudnnGetConvolutionForwardAlgorithm_v7:
            cudnnGetConvolutionForwardAlgorithm_v7Service();
            break;
        case cudarpc::QueryType::cudnnDestroyTensorDescriptor:
            cudnnDestroyTensorDescriptorService();
            break;
        case cudarpc::QueryType::cudnnDestroyFilterDescriptor:
            cudnnDestroyFilterDescriptorService();
            break;
        case cudarpc::QueryType::cudnnDestroyConvolutionDescriptor:
            cudnnDestroyConvolutionDescriptorService();
            break;
//        case cudarpc::QueryType::cudnnBatchNormalizationForwardInference:
//            cudnnBatchNormalizationForwardInferenceService();
//            break;
        case cudarpc::QueryType::cudnnSetConvolutionMathType:
            cudnnSetConvolutionMathTypeService();
            break;
        case cudarpc::QueryType::cudnnSetConvolutionGroupCount:
            cudnnSetConvolutionGroupCountService();
            break;
        case cudarpc::cudaGetLastError:
            cudaGetLastErrorService();
            break;
        case cudarpc::QueryType::cudaMalloc:
            cudaMallocService();
            break;
        case cudarpc::QueryType::cudaDeviceSynchronize:
            cudaDeviceSynchronizeService();
            break;
//        case cudarpc::QueryType::cudaLaunchKernel:
//            cudaLaunchKernelService();
//            break;
// =========================================================
        case cudarpc::QueryType::cuBLAS_cublasSetStream:
            cublasSetStreamService(); // 带一个参数
            break;
        case cudarpc::QueryType::cuBLAS_cublasSetMathMode: // 带一个参数
            cublasSetMathModeService();
            break;
        case cudarpc::QueryType::cuBLAS_cublasSgemm:
            cublasSgemmService(); // 计算密集型
            break;
        case cudarpc::QueryType::cuBLAS_cublasSgemmStridedBatched:
            cublasSgemmStridedBatchedService(); // 计算密集型
            break;
        case cudarpc::QueryType::cudaStreamSynchronize:
            cudaStreamSynchronizeService(); // 带一个参数
            break;
        // case cudarpc::QueryType::cudaStreamIsCapturing:  // 编译不存在，等会应该要解决
        //     cudaStreamIsCapturingService(); // 忽略参数
        //     break;
        case cudarpc::QueryType::cudaMemcpyAsync: // 复制的size，以及复制的流，0是默认阻塞流，用户创建的是异步，允许并发
            cudaMemcpyAsyncService();
            break;
        case cudarpc::QueryType::cuBLAS_cublasCreate:
            cublasCreateService();
            break;    
        default:
            inValid++;
            cublasSgemmService();
            break;
    }
}
// Execution time: 539.816 ms
int main() {
//    cudnnHandle_t cudnn_handle_;
//    cudnnCreateFilterDescriptorService();
//    cublasCreate(&cublas_handle_);
//    cudnnCreate(&cudnn_handle_);
//    cudnnCreate(&handle);
    initGlobalVar();
    initDescriptor();
    initForcublasSgemmStridedBatchedService();

    std::vector<int> commands;
    std::ifstream commandFile("../bert_cudaLog.txt");
    std::string line;

    if (commandFile.is_open()) {
        while (getline(commandFile, line)) {
            int command = std::stoi(line);
            commands.push_back(command);
        }
        commandFile.close();
    } else {
        std::cout << "Unable to open the command file." << std::endl;
        return 1;
    }


    int cnt = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int command : commands) {
        cudarpc::QueryType queryType = static_cast<cudarpc::QueryType>(command);
    //    std::cout << cnt++ << " " << queryType << std::endl;
        parse(queryType);
    }

    cudaDeviceSynchronize();
    
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "=======================================================" << std::endl;
    std::cout << "Total commands: " << commands.size() << "  inValid: " << inValid << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "=======================================================" << std::endl;
//    for (int i = 0; i < 10; ++i) {
//        cudaMallocService();
//        cudnnCreateService();
//        cudnnSetStreamService();
//        cudnnDestroyTensorDescriptorService();
//        cudnnDestroyFilterDescriptorService();
//        cudnnDestroyConvolutionDescriptorService();
//        cudnnConvolutionForwardService();
//        cudnnCreateTensorDescriptorService();
//        cudnnCreateFilterDescriptorService();
//        cudnnCreateConvolutionDescriptorService();
//        cudnnGetConvolutionForwardAlgorithm_v7Service();
//        cudaLaunchKernelService();
////        cudnnBatchNormalizationForwardInferenceService();
////        cudnnSetTensorNdDescriptorService();
////        cudnnSetFilterNdDescriptorService();
////        cudnnSetConvolutionNdDescriptorService();
//        cudnnSetConvolutionMathTypeService();
//        cudnnSetConvolutionGroupCountService();
//
//        cudaGetLastErrorService();
//        cudaDeviceSynchronize();
//    }

    destoryinitGlobalVar();
    destoryDescriptor();

    return 0;
}