import sys
import re

def filter_lines(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        pattern = re.compile(r'\[.*?\] \[info\] (.*)')
        filtered_lines = [pattern.search(line).group(1) + '\n' for line in lines if '[info]' in line 
                          and 'cudaGetLastError' not in line 
                          and 'recv sync response' not in line 
                          and 'cublasSetStream' not in line
                          and 'cudaStreamSynchronize' not in line
                          and 'cublasCreate' not in line
                          and 'cublasSetMathMode' not in line
                          and 'cudaMemcpyAsync' not in line
                          and 'cudaMalloc' not in line 
                          and 'cudaStreamIsCapturing' not in line]

        with open(output_file, 'w') as f:
            f.writelines(filtered_lines)
    except FileNotFoundError:
        pass

if __name__ == "__main__":    
    input_file = "../bertCUDA/bert_parameter_pre.txt"
    output_file = "output_bert_par.txt"
    filter_lines(input_file, output_file)
