#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <regex>

struct SgemmParam {
    int m, n, k, lda, ldb, ldc;
    double alpha, beta;
};

struct SgemmStridedBatchedParam {
    int m, n, k, lda, ldb, ldc;
    double alpha, beta;
    int strideA, strideB, strideC, batchCount;
};

std::vector<SgemmParam> cublasSgemmVector;
std::vector<SgemmStridedBatchedParam> cublasSSBVector;

void parseAndStoreData(const std::string& line) {
    std::regex rgx("\\{(\\w+):([\\d\\.]+)\\}"); // 匹配 {key:value} 格式，value可以是整数或浮点数
    std::sregex_iterator next(line.begin(), line.end(), rgx);
    std::sregex_iterator end;
    
    if (line.find("[cublasSgemm]") != std::string::npos) {
        SgemmParam param = {};
        while (next != end) {
            std::smatch match = *next;
            std::string key = match[1];
            std::string value = match[2];
            if (key == "m") param.m = std::stoi(value);
            else if (key == "n") param.n = std::stoi(value);
            else if (key == "k") param.k = std::stoi(value);
            else if (key == "lda") param.lda = std::stoi(value);
            else if (key == "ldb") param.ldb = std::stoi(value);
            else if (key == "ldc") param.ldc = std::stoi(value);
            else if (key == "alpha") param.alpha = std::stod(value);
            else if (key == "beta") param.beta = std::stod(value);
            next++;
        }
        cublasSgemmVector.push_back(param);
    } else if (line.find("[cublasSgemmStridedBatched]") != std::string::npos) {
        SgemmStridedBatchedParam param = {};
        while (next != end) {
            std::smatch match = *next;
            std::string key = match[1];
            std::string value = match[2];
            if (key == "m") param.m = std::stoi(value);
            else if (key == "n") param.n = std::stoi(value);
            else if (key == "k") param.k = std::stoi(value);
            else if (key == "lda") param.lda = std::stoi(value);
            else if (key == "ldb") param.ldb = std::stoi(value);
            else if (key == "ldc") param.ldc = std::stoi(value);
            else if (key == "alpha") param.alpha = std::stod(value);
            else if (key == "beta") param.beta = std::stod(value);
            else if (key == "strideA") param.strideA = std::stoi(value);
            else if (key == "strideB") param.strideB = std::stoi(value);
            else if (key == "strideC") param.strideC = std::stoi(value);
            else if (key == "batchCount") param.batchCount = std::stoi(value);
            next++;
        }
        cublasSSBVector.push_back(param);
    }
}

int main() {
    std::ifstream file("bert_parameter_well.txt"); // 替换为你的文件名
    std::string line;

    if (file.is_open()) {
        while (getline(file, line)) {
            parseAndStoreData(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    std::cout << "cublasSgemmVector contains:" << std::endl;
    for (const auto& elem : cublasSgemmVector) {
        std::cout << "m: " << elem.m << ", n: " << elem.n << ", k: " << elem.k
                  << ", lda: " << elem.lda << ", ldb: " << elem.ldb << ", ldc: " << elem.ldc
                  << ", alpha: " << elem.alpha << ", beta: " << elem.beta << std::endl;
    }

    std::cout << "cublasSSBVector contains:" << std::endl;
    for (const auto& elem : cublasSSBVector) {
        std::cout << "m: " << elem.m << ", n: " << elem.n << ", k: " << elem.k
                  << ", lda: " << elem.lda << ", ldb: " << elem.ldb << ", ldc: " << elem.ldc
                  << ", alpha: " << elem.alpha << ", beta: " << elem.beta
                  << ", strideA: " << elem.strideA << ", strideB: " << elem.strideB
                  << ", strideC: " << elem.strideC << ", batchCount: " << elem.batchCount << std::endl;
    }

    return 0;
}
