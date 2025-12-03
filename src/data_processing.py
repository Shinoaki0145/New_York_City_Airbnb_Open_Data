import numpy as np

def read_csv(file_path):
    parsed_rows = []
    current_row_str = ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        current_row_str += line
        quote_count = current_row_str.count('"')
        
        if quote_count % 2 == 0:
            row_text = current_row_str.rstrip('\n')
            row = []
            in_quote = False
            current_val = ""
            
            for char in row_text:
                if char == '"':
                    in_quote = not in_quote 
                elif char == ',' and not in_quote:
                    row.append(current_val)
                    current_val = ""
                else:
                    current_val += char

            row.append(current_val)
            parsed_rows.append(row)
            current_row_str = ""
    
    return np.array(parsed_rows)


def write_csv(file_path, data, column_names):
    data_with_header = np.vstack([column_names, data])
    
    np.savetxt(file_path, data_with_header, fmt='%s', delimiter=',', encoding='utf-8')
    
    print(f"Saved data to: {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Number of columns: {len(column_names)}")