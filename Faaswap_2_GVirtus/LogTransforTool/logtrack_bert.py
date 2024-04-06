def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    line_counts = {}
    total_lines = 0

    for line in lines:
        end_index = line.find(']', line.find(']') + 1) + 1
        processed_line = line[end_index:].strip()
        
        if processed_line and processed_line != "Receive API requests 1" and not processed_line.startswith("Skip"):
            # 删除每一行的前缀
            prefixes = ["Async: ", "Response: "]
            for prefix in prefixes:
                if processed_line.startswith(prefix):
                    processed_line = processed_line[len(prefix):].strip()
                    break
            if processed_line.startswith("[cublas] "):
                processed_line = processed_line[len("[cublas] "):].strip()
            
            if processed_line:  # 是否为空字符串
                processed_lines.append(processed_line + '\n')
                # 统计每行内容出现的次数
                if processed_line in line_counts:
                    line_counts[processed_line] += 1
                else:
                    line_counts[processed_line] = 1
                
                total_lines += 1

    with open(output_file, 'w') as file:
        file.writelines(processed_lines)
    
    # 输出每行内容及其出现的次数
    print("Line counts:")
    for line, count in line_counts.items():
        print(f"{line}: {count}")
    
    # 输出总行数
    print(f"Total lines: {total_lines}")

process_file('bert_log.txt', 'bert_log_tmp.txt')
