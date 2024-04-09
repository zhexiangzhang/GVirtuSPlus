#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
using namespace std;
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
   
struct cudnnSetConvolutionNdDescriptor { string convDesc; };
struct cudnnSetFilterNdDescriptor { string filterDesc; array<int, 4> filterDimA; };
struct cudnnSetTensorNdDescriptor { string tensorDesc; array<int, 4> dimA, strideA; };
struct cudnnBatchNormalizationForwardInference { string xDesc, yDesc, bnScaleBiasMeanVarDesc; };
struct cudnnConvolutionForward { string xDesc, wDesc, convDesc, yDesc; int algo, workSpaceSizeInBytes; };

// 定义cmdVector
vector<cudnnSetConvolutionNdDescriptor> cudnnSetConvolutionNdDCmd;
vector<cudnnSetFilterNdDescriptor> cudnnSetFilterNdDCmd;
vector<cudnnSetTensorNdDescriptor> cudnnSetTensorNdDCmd;
vector<cudnnBatchNormalizationForwardInference> cudnnBatchNormalizationForwardInferenceCmd;
vector<cudnnConvolutionForward> cudnnConvolutionForwardCmd;


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


int main() {
    ifstream file("whole.txt"); 
    string line;

    if (!file.is_open()) {
        cerr << "Unable to open file." << endl;
        return 1;
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
            cudnnSetConvolutionNdDescriptor std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "convDesc") std.convDesc = param.value;
            }
            cudnnSetConvolutionNdDCmd.push_back(std);
        } else if ( apiCall.name == "cudnnSetFilterNdDescriptor") {
            cudnnSetFilterNdDescriptor std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "filterDesc") std.filterDesc = param.value;
                if (param.name == "filterDimA") {
                    std.filterDimA[0] = stoi(param.value.substr(1, param.value.find(',')));
                    std.filterDimA[1] = stoi(param.value.substr(param.value.find(',')+1, param.value.find(',', param.value.find(',')+1)));
                    std.filterDimA[2] = stoi(param.value.substr(param.value.find(',', param.value.find(',')+1)+1, param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)));
                    std.filterDimA[3] = stoi(param.value.substr(param.value.find(',', param.value.find(',', param.value.find(',')+1)+1)+1, param.value.find(')')));
                    // 输出四个维度
                    cout << "filterDimA: " << std.filterDimA[0] << " " << std.filterDimA[1] << " " << std.filterDimA[2] << " " << std.filterDimA[3] << endl;
                }
            }
            cudnnSetFilterNdDCmd.push_back(std);
        } else if ( apiCall.name == "cudnnSetTensorNdDescriptor") {
            cudnnSetTensorNdDescriptor std;
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
            cudnnBatchNormalizationForwardInference std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "xDesc") std.xDesc = param.value;
                if (param.name == "yDesc") std.yDesc = param.value;
                if (param.name == "bnScaleBiasMeanVarDesc") std.bnScaleBiasMeanVarDesc = param.value;
            }
            cudnnBatchNormalizationForwardInferenceCmd.push_back(std);
        } else if ( apiCall.name == "cudnnConvolutionForward") {
            cudnnConvolutionForward std;
            for (const auto& param : apiCall.parameters) {
                if (param.name == "xDesc") std.xDesc = param.value;
                if (param.name == "wDesc") std.wDesc = param.value;
                if (param.name == "convDesc") std.convDesc = param.value;
                if (param.name == "yDesc") std.yDesc = param.value;
                if (param.name == "algo") std.algo = stoi(param.value);
                if (param.name == "workSpaceSizeInBytes") std.workSpaceSizeInBytes = stoi(param.value);
            }
            cudnnConvolutionForwardCmd.push_back(std);
        }
    }
        

    file.close();
    return 0;
}
