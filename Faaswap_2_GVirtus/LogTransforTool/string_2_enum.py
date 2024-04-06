# from string service name to enum number
from cudarpc_pb2 import QueryType

def service_to_enum(service_name):
    if service_name.startswith('cublas'):
        enum_name = 'cuBLAS_' + service_name
    else:
        enum_name = service_name
    return getattr(QueryType, enum_name, None)

def process_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip().replace('Service', '') 
            if line: 
                enum_value = service_to_enum(line)
                if enum_value is not None:
                    f_out.write(f"{enum_value}\n")
                else:
                    f_out.write("Unknown Service\n")


input_file = 'bert_log_tmp.txt'  
output_file = 'bert_log_no.txt'  

process_file(input_file, output_file)
