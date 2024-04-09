#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <array>

struct TensorDescriptor {
    uint64_t tensorDesc;
};

struct SetTensorNdDescriptor {
    uint64_t tensorDesc;
    int dataType, nbDims;
    std::array<int, 4> dimA;
    std::array<int, 4> strideA;
};

struct FilterDescriptor {
    uint64_t filterDesc;
};

struct SetFilterNdDescriptor {
    uint64_t filterDesc;
    int dataType, format, nbDims;
    std::array<int, 4> filterDimA;
};

// 其他结构体定义类似...

// 存储每个API的结构体
std::vector<TensorDescriptor> tensorDescs;
std::vector<SetTensorNdDescriptor> setTensorNdDescs;
std::vector<FilterDescriptor> filterDescs;
std::vector<SetFilterNdDescriptor> setFilterNdDescs;

// 解析辅助函数
uint64_t extractUInt64(const std::string& str) {
    size_t pos1 = str.find(':') + 1;
    size_t pos2 = str.find('}');
    return std::stoull(str.substr(pos1, pos2 - pos1));
}

int extractInt(const std::string& str) {
    size_t pos1 = str.find(':') + 1;
    size_t pos2 = str.find('}');
    return std::stoi(str.substr(pos1, pos2 - pos1));
}

void parseDims(const std::string& str, std::array<int, 4>& dims) {
    size_t pos1 = str.find('(') + 1;
    size_t pos2 = str.find(')');
    std::string dimsStr = str.substr(pos1, pos2 - pos1);

    // Replace commas with spaces
    for (char& ch : dimsStr) {
        if (ch == ',') {
            ch = ' ';
        }
    }
    std::istringstream iss(dimsStr);
    int value, index = 0;
    while (iss >> value) {
        dims[index++] = value;
    }
}



// 解析每行并创建对象
void parseLine(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    iss >> token; // 读取 API 名称

    if (token == "[cudnnCreateTensorDescriptor]") {
        TensorDescriptor td;
        std::string param;
        iss >> param;
        td.tensorDesc = extractUInt64(param);
        tensorDescs.push_back(td);
    } // 解析[cudnnSetTensorNdDescriptor]行
   if (token == "[cudnnSetTensorNdDescriptor]") {
        SetTensorNdDescriptor std;
        std::string param;
        while (iss >> param) {
            if (param.find("tensorDesc") != std::string::npos) std.tensorDesc = extractUInt64(param);
            else if (param.find("dataType") != std::string::npos) std.dataType = extractInt(param);
            else if (param.find("nbDims") != std::string::npos) std.nbDims = extractInt(param);
            else if (param.find("dimA") != std::string::npos) {
                parseDims(param, std.dimA);
                std::cout << "Parsed dimA: ";
                for (int dim : std.dimA) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }
            else if (param.find("strideA") != std::string::npos) {
                parseDims(param, std.strideA);
                // 在这里打印 strideA 的解析结果
                std::cout << "Parsed strideA: ";
                for (int stride : std.strideA) {
                    std::cout << stride << " ";
                }
                std::cout << std::endl;
            }
        }
        setTensorNdDescs.push_back(std);
    }

    
    // 其他API解析类似...
}

int main() {
    std::string line;
    std::ifstream file("output_cv_par.txt");

    while (getline(file, line)) {
        parseLine(line);
    }

    // // 输出结果来验证
    // for (const auto& td : tensorDescs) {
    //     std::cout << "TensorDescriptor: " << td.tensorDesc << std::endl;
    // }
    // for (const auto& std : setTensorNdDescs) {
    //     std::cout << "SetTensorNdDescriptor: " << std.tensorDesc << " " << std.dataType << " " << std.nbDims << " " << std.dimA[0] << " " << std.dimA[1] << " " << std.dimA[2] << " " << std.dimA[3] << " " << std.strideA[0] << " " << std.strideA[1] << " " << std.strideA[2] << " " << std.strideA[3] << std::endl;
    // }

    return 0;
}
