//#include <cudnn.h>
//#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "proto/cudarpc.pb.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cublas_v2.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cudnn.h"
#include </usr/local/cuda-11.4/targets/x86_64-linux/include/cuda_runtime.h>

#include <sstream>
#include <array>
using namespace std;

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







/*
现在我有一个txt文件，每一行为一个cuda api和参数

包含以下几种类型：

[cudnnCreateConvolutionDescriptor] [{convDesc:2304475264}]
[cudnnCreateFilterDescriptor] [{filterDesc:140731202895632}]
[cudnnCreateTensorDescriptor] [{tensorDesc:30}]
[cudnnDestroyConvolutionDescriptor] [{convDesc:1048681}]
[cudnnDestroyFilterDescriptor] [{filterDesc:1048697}]
[cudnnDestroyTensorDescriptor] [{tensorDesc:1048869}]

[cudnnSetConvolutionNdDescriptor] [{convDesc:1048681}, {padA:(0, 0)}, {filterStrideA:(1, 1)}, {dilationA:(1, 1)}, {dataType:0}]
[cudnnSetFilterNdDescriptor] [{filterDesc:1048697}, {filterDimA:(512, 128, 1, 1)}]
[cudnnSetTensorNdDescriptor] [{tensorDesc:1048604}, {dimA:(1, 256, 56, 56)}, {strideA:(802816, 3136, 56, 1)}]

[cudnnBatchNormalizationForwardInference] [{mode:1}, {xDesc:1048658}, {yDesc:1048658}, {bnScaleBiasMeanVarDesc:1048659}, {epsilon:0.000010}]
[cudnnConvolutionForward] [{alpha:1.000000}, {xDesc:1048660}, {wDesc:1048661}, {convDesc:1048663}, {algo:1}, {workSpaceSizeInBytes:1024}, {yDesc:1048662}, {beta:0.000000}]
*/

//=====================================================================================================

// 每种create和destroy API对应一个vector，存储对应的id
vector<string> cudnnCreConvolutionDCmd;
vector<string> cudnnCreFilterDCmd;
vector<string> cudnnCreTensorDCmd;
vector<string> cudnnDesConvolutionDCmd;
vector<string> cudnnDesFilterDCmd;
vector<string> cudnnDesTensorDCmd;

// 定义一个集合，key为11种API，value为对应的访问数量
map<string, int> apiAccessCount;

// 向map中插入11种API
void resetApiAccessCount() {
    apiAccessCount["cudnnCreateConvolutionDescriptor"] = 0;
    apiAccessCount["cudnnCreateFilterDescriptor"] = 0;
    apiAccessCount["cudnnCreateTensorDescriptor"] = 0;
    apiAccessCount["cudnnDestroyConvolutionDescriptor"] = 0;
    apiAccessCount["cudnnDestroyFilterDescriptor"] = 0;
    apiAccessCount["cudnnDestroyTensorDescriptor"] = 0;
    apiAccessCount["cudnnSetConvolutionNdDescriptor"] = 0;
    apiAccessCount["cudnnSetFilterNdDescriptor"] = 0;
    apiAccessCount["cudnnSetTensorNdDescriptor"] = 0;
    apiAccessCount["cudnnBatchNormalizationForwardInference"] = 0;
    apiAccessCount["cudnnConvolutionForward"] = 0;
}

struct cudnnSetConvolutionNd { string convDesc; };
struct cudnnSetFilterNd { string filterDesc; array<int, 4> filterDimA; };
struct cudnnSetTensorNd { string tensorDesc; array<int, 4> dimA, strideA; };
struct cudnnBatchNormaForwardInf { string xDesc, yDesc, bnScaleBiasMeanVarDesc; };
struct cudnnConvForward { string xDesc, wDesc, convDesc, yDesc; int algo, workSpaceSizeInBytes; };

// 定义cmdVector
vector<cudnnSetConvolutionNd> cudnnSetConvolutionNdDCmd;
vector<cudnnSetFilterNd> cudnnSetFilterNdDCmd;
vector<cudnnSetTensorNd> cudnnSetTensorNdDCmd;
vector<cudnnBatchNormaForwardInf> cudnnBatchNormaForwardInfCmd;
vector<cudnnConvForward> cudnnConvForwardCmd;

//=====================================================================================================
struct Parameter {
    string name;
    string value;
};

struct ApiCall {
    string name;
    vector<Parameter> parameters;
};

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

ApiCall parseLine(const string& line) {
    ApiCall apiCall;
    auto parts = split(line, ']');
    if (!parts.empty()) {
        apiCall.name = parts[0].substr(1); // Remove leading '['
        
        // Now process the parameters, if any
        if (parts.size() > 1) {
            auto paramsWithBraces = split(parts[1], '{');
            for (auto& param : paramsWithBraces) {
                if (!param.empty()) {
                    auto paramParts = split(param, ':');
                    if (paramParts.size() == 2) {
                        Parameter p;
                        p.name = paramParts[0];
                        p.value = paramParts[1].substr(0, paramParts[1].find('}')); // Remove trailing '}'
                        apiCall.parameters.push_back(p);
                    }
                }
            }
        }
    }
    return apiCall;
}



















int cntAsyncMecopy = 0;

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

float *d_memory;
size_t totalSize = 1L * 1024 * 1024 * 1024; // 10GB
float *d_output;

// 改为map
map<string, cudnnTensorDescriptor_t> tensorDescriptorsMap;
map<string, cudnnFilterDescriptor_t> filterDescriptorsMap;
map<string, cudnnConvolutionDescriptor_t> convolutionDescriptorsMap;

int n, c, h, w;
int batch_size = 1, channels = 3, height = 1024, width = 1024;
int out_channels = 16, kernel_height = 5, kernel_width = 5;

const int dimA[4] = {1, 64, 56, 56};
const int strideA[4] = {200704, 3136, 56, 1};

float alpha = 1.0f, beta = 0.0f;
size_t workspace_bytes = 0;

float A[15], B[12], C[20];

int batchCount = 10;
    
float *A_mem = new float[15 * batchCount];
float *B_mem = new float[12 * batchCount];
float *C_mem = new float[20 * batchCount];

void initGlobalVar(){
    cublasCreate(&cublas_handle_);

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
}

void destoryinitGlobalVar(){
    cudnnDestroy(handle);
    CHECK_CUDA(cudaFree(d_memory));
}

void initDescriptor(){
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&t1));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&f1));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&tc1));


    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 56, 56));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 256, 1, 1));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));    
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &n, &c, &h, &w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));


    // CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 244, 244));
    // CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 64, 3, 7, 7));
    // CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 3, 3, 2, 2, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));    
    // CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &n, &c, &h, &w));
    // CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));


    // CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));
    // CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, channels, kernel_height, kernel_width));
    // CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));


    // CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &n, &c, &h, &w));
    // CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

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
void cudaStreamSynchronizeService() { CHECK_CUDA(cudaStreamSynchronize(0));}
void cublasSetStreamService() { cublasSetStream(cublas_handle_, 0);}
void cublasSetMathModeService() { cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);}
void cublasSgemmService() {
    cublasSgemm(cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                1000, 1, 2048,
                &alpha, d_memory, 1000, d_memory, 2048, &beta, d_memory, 1000);}
void cudaMemcpyAsyncService() {
    // from src --copy-> count bbyte to dst。
    // cudaStream_t 类型的参数来指定操作关联的流

    float* h_src = new float[200528]; // host
    if (cntAsyncMecopy == 0) {
        CHECK_CUDA(cudaMemcpyAsync(d_output, h_src, 602112, cudaMemcpyHostToDevice, 0)); // 最后是流
    }
    if (cntAsyncMecopy == 1) {
        CHECK_CUDA(cudaMemcpyAsync(d_output, h_src, 4000, cudaMemcpyHostToDevice, 0)); // 最后是流
    }
    if (cntAsyncMecopy == 2) { 
        CHECK_CUDA(cudaMemcpyAsync(d_output, h_src, 4, cudaMemcpyHostToDevice, 0)); // 最后是流
    }
    cntAsyncMecopy = cntAsyncMecopy + 1;
}                
// --------------------------------------------------------------
void cudaGetLastErrorService() {cudaGetLastError();}  // need to add cudacheck
void cudaDeviceSynchronizeService() {cudaDeviceSynchronize();}
void cudaMallocService() { CHECK_CUDA(cudaMalloc(&space1, 20971520));}
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
void cudnnSetFilterNdDescriptorService() { 
    // 访问对应cudnnSetFilterNdDCmd的第apiAccessCount["cudnnSetFilterNdDescriptor"]个元素
    cudnnSetFilterNd tmp = cudnnSetFilterNdDCmd[apiAccessCount["cudnnSetFilterNdDescriptor"]];
    apiAccessCount["cudnnSetFilterNdDescriptor"]++;

    // 获取filterDimA
    const int tmpfilterDimA[4] = {tmp.filterDimA[0], tmp.filterDimA[1], tmp.filterDimA[2], tmp.filterDimA[3]};

    // 检查是否存在对应的 filterDesc, 如果不存在则创建一个
    if (filterDescriptorsMap.find(tmp.filterDesc) == filterDescriptorsMap.end()) {
        cudnnFilterDescriptor_t tempFilter;
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&tempFilter));
        filterDescriptorsMap[tmp.filterDesc] = tempFilter;
    }
    // 从map中获取对应的filterDescriptor
    cudnnFilterDescriptor_t tmpfilterDescriptor = filterDescriptorsMap[tmp.filterDesc];

    CHECK_CUDNN(cudnnSetFilterNdDescriptor(tmpfilterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, tmpfilterDimA));
}
void cudnnSetTensorNdDescriptorService() { 
    // 访问对应cudnnSetTensorNdDCmd的第apiAccessCount["cudnnSetTensorNdDescriptor"]个元素
    cudnnSetTensorNd tmp = cudnnSetTensorNdDCmd[apiAccessCount["cudnnSetTensorNdDescriptor"]];
    apiAccessCount["cudnnSetTensorNdDescriptor"]++;

    // 获取dimA和strideA
    const int dimA[4] = {tmp.dimA[0], tmp.dimA[1], tmp.dimA[2], tmp.dimA[3]};
    const int strideA[4] = {tmp.strideA[0], tmp.strideA[1], tmp.strideA[2], tmp.strideA[3]};


    // 检查是否存在对应的 tensorDesc, 如果不存在则创建一个
    if (tensorDescriptorsMap.find(tmp.tensorDesc) == tensorDescriptorsMap.end()) {
        cudnnTensorDescriptor_t tempTensor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&tempTensor));
        tensorDescriptorsMap[tmp.tensorDesc] = tempTensor;
    }
    // 从map中获取对应的tensorDescriptor
    cudnnTensorDescriptor_t t1 = tensorDescriptorsMap[tmp.tensorDesc];

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(t1, CUDNN_DATA_FLOAT, 4, dimA, strideA));
}
void cudnnSetStreamService() { CHECK_CUDNN(cudnnSetStream(handle, 0));}

void cudnnCreateTensorDescriptorService(){

    // 访问对应cudnnCreConvolutionDCmd的第apiAccessCount["cudnnCreateTensorDescriptor"]个元素
    string convDesc = cudnnCreTensorDCmd[apiAccessCount["cudnnCreateTensorDescriptor"]];
    apiAccessCount["cudnnCreateTensorDescriptor"]++;

    // 查询tensorDescriptorsMap中是否存在对应的convDesc，如果不存在继续处理    
    if (tensorDescriptorsMap.find(convDesc) == tensorDescriptorsMap.end()) {                
        cudnnTensorDescriptor_t tempTensor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&tempTensor));
    
        // tensorDescriptorsMap 插入
        tensorDescriptorsMap[convDesc] = tempTensor;
    }

}
void cudnnCreateFilterDescriptorService(){

     // 访问对应cudaCreFilterDCmd的第apiAccessCount["cudnnCreateFilterDescriptor"]个元素
    string filterDesc = cudnnCreFilterDCmd[apiAccessCount["cudnnCreateFilterDescriptor"]];
    apiAccessCount["cudnnCreateFilterDescriptor"]++;

    // 查询filterDescriptorsMap中是否存在对应的filterDesc，如果不存在继续处理
    if (filterDescriptorsMap.find(filterDesc) == filterDescriptorsMap.end()) {
        cudnnFilterDescriptor_t tempFilter;
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&tempFilter));
    
        // filterDescriptorsMap 插入
        filterDescriptorsMap[filterDesc] = tempFilter;
    }
}
void cudnnCreateConvolutionDescriptorService() {
    
    // 访问对应cudnnCreConvolutionDCmd的第apiAccessCount["cudnnCreateConvolutionDescriptor"]个元素
    string convDesc = cudnnCreConvolutionDCmd[apiAccessCount["cudnnCreateConvolutionDescriptor"]];
    apiAccessCount["cudnnCreateConvolutionDescriptor"]++;

    // 查询convolutionDescriptorsMap中是否存在对应的convDesc，如果不存在继续处理
    if (convolutionDescriptorsMap.find(convDesc) == convolutionDescriptorsMap.end()) {
        cudnnConvolutionDescriptor_t tempConvolution;
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&tempConvolution));
        // convolutionDescriptorsVector.push_back(tempConvolution);
        
        // convolutionDescriptorsMap 插入
        convolutionDescriptorsMap[convDesc] = tempConvolution;
    }
}

void cudnnSetConvolutionNdDescriptorService() { 
    // 访问对应cudnnSetConvolutionNdDCmd的第apiAccessCount["cudnnSetConvolutionNdDescriptor"]个元素     
    cudnnSetConvolutionNd tmp = cudnnSetConvolutionNdDCmd[apiAccessCount["cudnnSetConvolutionNdDescriptor"]];
    apiAccessCount["cudnnSetConvolutionNdDescriptor"]++;

    // 检查是否存在对应的 convDesc, 如果不存在则创建一个
    if (convolutionDescriptorsMap.find(tmp.convDesc) == convolutionDescriptorsMap.end()) {
        cudnnConvolutionDescriptor_t tempConvolution;
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&tempConvolution));
        convolutionDescriptorsMap[tmp.convDesc] = tempConvolution;
    }
    // 从map中获取对应的convolutionDescriptor
    cudnnConvolutionDescriptor_t tmpconvolutionDescriptor = convolutionDescriptorsMap[tmp.convDesc];
    const int tmpM[2] = {1, 1};

    // CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 3, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    // CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(tmpconvolutionDescriptor, 2, tmpM, tmpM, tmpM, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
}

void cudnnDestroyTensorDescriptorService() {
    // 首先获取apiAccessCount中cudnnDestroyTensorDescriptor的值
    string tensorDesc = cudnnDesTensorDCmd[apiAccessCount["cudnnDestroyTensorDescriptor"]];
    apiAccessCount["cudnnDestroyTensorDescriptor"]++;
    // 通过tensorDescriptorsMap获取对应的tensorDescriptor
    cudnnTensorDescriptor_t tensorDescriptor = tensorDescriptorsMap[tensorDesc];
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDescriptor));
    // 从tensorDescriptorsMap中删除
    tensorDescriptorsMap.erase(tensorDesc);
}
void cudnnDestroyFilterDescriptorService() {
    // 首先获取apiAccessCount中cudnnDestroyFilterDescriptor的值
    string filterDesc = cudnnDesFilterDCmd[apiAccessCount["cudnnDestroyFilterDescriptor"]];
    apiAccessCount["cudnnDestroyFilterDescriptor"]++;
    // 通过filterDescriptorsMap获取对应的filterDescriptor
    cudnnFilterDescriptor_t filterDescriptor = filterDescriptorsMap[filterDesc];
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));
    // 从filterDescriptorsMap中删除
    filterDescriptorsMap.erase(filterDesc);    
}
void cudnnDestroyConvolutionDescriptorService() {
    // 首先获取apiAccessCount中cudnnDestroyConvolutionDescriptor的值
    string convDesc = cudnnDesConvolutionDCmd[apiAccessCount["cudnnDestroyConvolutionDescriptor"]];
    apiAccessCount["cudnnDestroyConvolutionDescriptor"]++;
    // 通过convolutionDescriptorsMap获取对应的convolutionDescriptor
    cudnnConvolutionDescriptor_t convolutionDescriptor = convolutionDescriptorsMap[convDesc];
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
    // 从convolutionDescriptorsMap中删除
    convolutionDescriptorsMap.erase(convDesc);
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
    // 访问对应cudnnBatchNormaForwardInfCmd的第apiAccessCount["cudnnBatchNormalizationForwardInference"]个元素
    cudnnBatchNormaForwardInf tmp = cudnnBatchNormaForwardInfCmd[apiAccessCount["cudnnBatchNormalizationForwardInference"]];
    apiAccessCount["cudnnBatchNormalizationForwardInference"]++;
    // 获取对应的xDesc, yDesc, bnScaleBiasMeanVarDesc
    cudnnTensorDescriptor_t xDesc_batch = tensorDescriptorsMap[tmp.xDesc];
    cudnnTensorDescriptor_t yDesc_batch = tensorDescriptorsMap[tmp.yDesc];
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_batch = tensorDescriptorsMap[tmp.bnScaleBiasMeanVarDesc];    

    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            handle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            xDesc_batch, d_memory,
            yDesc_batch, d_memory,
            bnScaleBiasMeanVarDesc_batch,
            d_memory, d_memory,
            d_memory, d_memory, 0.000010));
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
       case cudarpc::QueryType::cudnnSetConvolutionNdDescriptor:
           cudnnSetConvolutionNdDescriptorService();
           break;
        case cudarpc::QueryType::cudnnCreate:
            cudnnCreateService();
            break;
       case cudarpc::QueryType::cudnnSetTensorNdDescriptor:
           cudnnSetTensorNdDescriptorService();
           break;
        case cudarpc::QueryType::cudnnSetFilterNdDescriptor:
           cudnnSetFilterNdDescriptorService();
           break;
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
       case cudarpc::QueryType::cudnnBatchNormalizationForwardInference:
           cudnnBatchNormalizationForwardInferenceService();
           break;
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
       case cudarpc::QueryType::cudaLaunchKernel:
           cudaLaunchKernelService();
           break;
// =========================================================
        case cudarpc::QueryType::cuBLAS_cublasSetStream:
            cublasSetStreamService();
            break;
        case cudarpc::QueryType::cudaStreamSynchronize:
            cudaStreamSynchronizeService(); // 带一个参数
            break;
        case cudarpc::QueryType::cuBLAS_cublasSetMathMode:
            cublasSetMathModeService();
            break;
        case cudarpc::QueryType::cuBLAS_cublasSgemm:
            cublasSgemmService();
            break;
        case cudarpc::QueryType::cudaMemcpyAsync: // 复制的size，以及复制的流，0是默认阻塞流，用户创建的是异步，允许并发
            cudaMemcpyAsyncService();
            break;
        default:
            inValid++;
            cudnnSetConvolutionMathTypeService();
            break;
    }
}
// Execution time: 539.816 ms


void readFromWholeCVPara(){
    ifstream file("/root/zzx/GVirtuSPlus/Faaswap_2_GVirtus/LogTransforTool/cv_each_para_value/whole.txt"); 
    string line;

    if (!file.is_open()) {
        cerr << "Unable to open file whole.txt" << endl;
        exit(0);
    }

    while (getline(file, line)) {
        ApiCall apiCall = parseLine(line);
        // 只有apiCall.name为cudnnCreateFilterDescriptor时，才输出
        
        
        // 存储到对应的vector中
        if (apiCall.name == "cudnnCreateFilterDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnCreFilterDCmd.push_back(param.value);}
        } else if (apiCall.name == "cudnnCreateConvolutionDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnCreConvolutionDCmd.push_back(param.value);}
        } else if (apiCall.name == "cudnnCreateTensorDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnCreTensorDCmd.push_back(param.value);}
        } else if (apiCall.name == "cudnnDestroyConvolutionDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnDesConvolutionDCmd.push_back(param.value);}
        } else if (apiCall.name == "cudnnDestroyFilterDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnDesFilterDCmd.push_back(param.value);}
        } else if (apiCall.name == "cudnnDestroyTensorDescriptor") {
            for (const auto& param : apiCall.parameters) {cudnnDesTensorDCmd.push_back(param.value);}
        } 
        //================另外的先存到结构体，再放到向量中
        else if ( apiCall.name == "cudnnSetConvolutionNdDescriptor") {
            cudnnSetConvolutionNd std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "convDesc") std.convDesc = param.value;
            }
            cudnnSetConvolutionNdDCmd.push_back(std);
        } else if ( apiCall.name == "cudnnSetFilterNdDescriptor") {
            cudnnSetFilterNd std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "filterDesc") std.filterDesc = param.value;
                if (param.name == "filterDimA") {
                    std.filterDimA[0] = stoi(param.value.substr(1, param.value.find(',')));
                    std.filterDimA[1] = stoi(param.value.substr(param.value.find(',')+1, param.value.find(',', param.value.find(',')+1)));
                    std.filterDimA[2] = stoi(param.value.substr(param.value.find(',', param.value.find(',')+1)+1, param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)));
                    std.filterDimA[3] = stoi(param.value.substr(param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)+1, param.value.find(')')));
                    // 输出四个维度
                    // cout << "filterDimA: " << std.filterDimA[0] << " " << std.filterDimA[1] << " " << std.filterDimA[2] << " " << std.filterDimA[3] << endl;
                }
            }
            cudnnSetFilterNdDCmd.push_back(std);
        } else if ( apiCall.name == "cudnnSetTensorNdDescriptor") {
            cudnnSetTensorNd std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "tensorDesc") std.tensorDesc = param.value;
                if (param.name == "dimA") {
                    std.dimA[0] = stoi(param.value.substr(1, param.value.find(',')));
                    std.dimA[1] = stoi(param.value.substr(param.value.find(',')+1, param.value.find(',', param.value.find(',')+1)));
                    std.dimA[2] = stoi(param.value.substr(param.value.find(',', param.value.find(',')+1)+1, param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)));
                    std.dimA[3] = stoi(param.value.substr(param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)+1, param.value.find(')')));
                }
                if (param.name == "strideA") {  
                    std.strideA[0] = stoi(param.value.substr(1, param.value.find(',')));
                    std.strideA[1] = stoi(param.value.substr(param.value.find(',')+1, param.value.find(',', param.value.find(',')+1)));
                    std.strideA[2] = stoi(param.value.substr(param.value.find(',', param.value.find(',')+1)+1, param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)));
                    std.strideA[3] = stoi(param.value.substr(param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)+1, param.value.find(')')));
                }
            }
            cudnnSetTensorNdDCmd.push_back(std);
        } else if ( apiCall.name == "cudnnBatchNormalizationForwardInference") {
            cudnnBatchNormaForwardInf std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "xDesc") std.xDesc = param.value;
                if (param.name == "yDesc") std.yDesc = param.value;
                if (param.name == "bnScaleBiasMeanVarDesc") std.bnScaleBiasMeanVarDesc = param.value;
            }
            cudnnBatchNormaForwardInfCmd.push_back(std);
        } else if ( apiCall.name == "cudnnConvolutionForward") {
            cudnnConvForward std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "xDesc") std.xDesc = param.value;
                if (param.name == "wDesc") std.wDesc = param.value;
                if (param.name == "convDesc") std.convDesc = param.value;
                if (param.name == "yDesc") std.yDesc = param.value;
                if (param.name == "algo") std.algo = stoi(param.value);
                if (param.name == "workSpaceSizeInBytes") std.workSpaceSizeInBytes = stoi(param.value);
            }
            cudnnConvForwardCmd.push_back(std);
        }
        
    }         

    file.close();
}

int main() {
    initGlobalVar();
    initDescriptor();

    std::vector<int> commands;
    std::ifstream commandFile("/root/zzx/GVirtuSPlus/Faaswap_2_GVirtus/cvCUDA/cv_cudaLog.txt");
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

    readFromWholeCVPara();

    // 输出每个vector的维度    
    cout << "cudnnCreFilterDCmd: " << cudnnCreFilterDCmd.size() << endl;
    cout << "cudnnCreConvolutionDCmd: " << cudnnCreConvolutionDCmd.size() << endl;
    cout << "cudnnCreTensorDCmd: " << cudnnCreTensorDCmd.size() << endl;
    cout << "cudnnDesConvolutionDCmd: " << cudnnDesConvolutionDCmd.size() << endl;
    cout << "cudnnDesFilterDCmd: " << cudnnDesFilterDCmd.size() << endl;
    cout << "cudnnDesTensorDCmd: " << cudnnDesTensorDCmd.size() << endl;
    cout << "cudnnSetConvolutionNdDCmd: " << cudnnSetConvolutionNdDCmd.size() << endl;
    cout << "cudnnSetFilterNdDCmd: " << cudnnSetFilterNdDCmd.size() << endl;
    cout << "cudnnSetTensorNdDCmd: " << cudnnSetTensorNdDCmd.size() << endl;
    // cout << "cudnnBatchNormalizationForwardInferenceCmd: " << cudnnBatchNormalizationForwardInferenceCmd.size() << endl;
    // cout << "cudnnConvolutionForwardCmd: " << cudnnConvolutionForwardCmd.size() << endl;

    resetApiAccessCount();  //很重要


    int cnt = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int command : commands) {
        cudarpc::QueryType queryType = static_cast<cudarpc::QueryType>(command);
//        std::cout << cnt++ << " " << queryType << std::endl;
        parse(queryType);
    }

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "=======================================================" << std::endl;
    std::cout << "Total commands: " << commands.size() << "  inValid: " << inValid << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "=======================================================" << std::endl;

    for (int i = 0; i < 10; i++) {
        resetApiAccessCount();
        auto start = std::chrono::high_resolution_clock::now();
        // cublasSgemmStridedBatchedCount = 0;
        // cublasSgemmCount = 0;
        for (int command : commands) {
            cudarpc::QueryType queryType = static_cast<cudarpc::QueryType>(command);
        //    std::cout << cnt++ << " " << queryType << std::endl;
            parse(queryType);
        }
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "=======================================================" << std::endl;
        // std::cout << "Total commands: " << commands.size() << "  inValid: " << inValid << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "=======================================================" << std::endl;
    }

    destoryinitGlobalVar();
    destoryDescriptor();

    return 0;
}