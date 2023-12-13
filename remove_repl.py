def remove_duplicates(input_file_path, output_file_path):
    unique_lines = set()

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            unique_lines.add(line.strip())  # Strip to remove any leading/trailing whitespace

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in unique_lines:
            file.write(line + '\n')

# Example usage
input_file_path = 'train_data/moveListTrain.txt'  # Replace with your input file path
output_file_path = 'train_data/moveListTrain_remove_duplicate.txt'  # The output file with duplicates removed

remove_duplicates(input_file_path, output_file_path)