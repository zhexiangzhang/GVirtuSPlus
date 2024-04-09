import re

def filter_lines(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # 正则表达式用于移除行首的时间戳和 [info]
        pattern = re.compile(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\] \[info\] (.*)')

        filtered_lines = []
        for line in lines:
            match = pattern.search(line)
            if match:
                content = match.group(1)
                # 检查行内容是否不含指定的字符串
                if not any(word in content for word in ['cublasSgemm', 'cudaMemcpyAsync', 'cublasSetStream', 'cublasSetMathMode', 'cudaStreamSynchronize', 'cudaGetLastError', 'cudaStreamIsCapturing', 'recv sync response', 'cublasCreate', 'cudaMalloc']):
                    filtered_lines.append(content + '\n')


        with open(output_file, 'w') as f:
            f.writelines(filtered_lines)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":    
    input_file = "../cvCUDA/cv_parameter_pre.txt"
    output_file = "output_cv_par.txt"
    filter_lines(input_file, output_file)
