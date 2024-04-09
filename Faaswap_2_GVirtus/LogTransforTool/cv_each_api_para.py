# def filter_lines(input_file, output_file):
#     try:
#         with open(input_file, 'r') as file:
#             with open(output_file, 'w') as output:
#                 for line in file:
#                     if 'cudnnDestroyTensorDescriptor' in line:
#                         # # # # 移除 dataType 和 nbDims
#                         # line_parts = line.split(', ')   
#                         # filtered_parts = [part for part in line_parts if 'alpha' not in part and 'beta' not in part]
#                         # filtered_line = ', '.join(filtered_parts)
#                         output.write(line)                        
#                         # output.write(filtered_line)
#         print(f"Filtered lines have been saved to {output_file}")
#     except FileNotFoundError:
#         print(f"The file {input_file} was not found.")

# filter_lines('output_cv_par.txt', 'cv_each_para_value/cudnnDestroyTensorDescriptor.txt')


# def remove_alpha_beta(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()

#     output_lines = []
#     for line in lines:
#         line = line.strip()
#         # 利用字符串处理方法去除alpha和beta信息
#         line = line.replace('{alpha:1.000000},', '')
#         line = line.replace(', {beta:0.000000}', '')
#         output_lines.append(line)

#     with open(output_file, 'w') as f:
#         f.write('\n'.join(output_lines))


# input_file = "cv_each_para_value/cudnnConvolutionForward.txt"
# output_file = "cv_each_para_value/cudnnConvolutionForward2.txt"
# remove_alpha_beta(input_file, output_file)

import os

def merge_txt_files(input_folder, output_file):
    txt_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.txt')]

    # 逐个读取txt文件内容并写入新文件
    with open(output_file, 'w') as output_f:
        for txt_file in txt_files:
            with open(txt_file, 'r') as input_f:
                output_f.write(input_f.read())
                # 在每个文件的内容之后添加换行符
                # output_f.write('\n')
                # sleep(10)

# 输入文件夹路径和输出文件名
input_folder = "cv_each_para_value/"
output_file = "cv_each_para_value/whole.txt"

# 调用函数进行合并
merge_txt_files(input_folder, output_file)
