import re
from collections import Counter

filename = 'log.txt'
new_filename = 'cv_cudaLog_filter.txt'

with open(filename, 'r') as file:
    content = file.readlines()

# 打开新文件以便写入
with open(new_filename, 'w') as new_file:
    pattern = r'CUDA API (\d+)'

    matches = []
    for line in content:
        match = re.findall(pattern, line)
        if match:
            # 将找到的数字添加到matches列表中
            matches.extend(match)
            # 同时，将数字写入新文件，后面加上换行符
            new_file.write(f"{match[0]}\n")

    counter = Counter(matches)
    sorted_matches = sorted(counter.items(), key=lambda x: int(x[0]))

    num_matches = len(matches)

    print("num_matches:", num_matches)

    for match, count in sorted_matches:
        print(f"Match: {match}, Count: {count}")
