# Define input and output file paths
input_file = "/Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/optitrack/recordings/NEW_JUMP_ACTUAL.txt"  # Replace with your actual input file path
output_file = "/Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/optitrack/recordings/NEW_JUMP_2DANCERS.txt"  # Replace with your desired output file path

# Read the input file
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Open the output file for writing
with open(output_file, 'w') as outfile:
    # Iterate through each line
    for i, line in enumerate(lines):
        if i == 0:
            # Header line: duplicate the columns with '2::x::y'
            columns = line.strip().split('\t')
            new_columns = []

            # Loop through the columns
            for col in columns:
                # Add original column
                new_columns.append(col)
                if "::" in col:
                    # Duplicate column with '2::x::y' instead of '1::x::y'
                    duplicated_col = col.replace('1::', '2::')
                    new_columns.append(duplicated_col)

            # Write the modified header to the output file
            outfile.write('\t'.join(new_columns) + '\n')

        else:
            # Data line: process each entry and duplicate the columns accordingly
            data = line.strip().split('\t')
            new_data = []

            for idx, entry in enumerate(data):
                # Add original data
                new_data.append(entry)
                
                # Check if the current column is a position or orientation column
                if idx > 0 and "::" in columns[idx]:  # Ignore timestamp at index 0
                    # Duplicate the entry
                    new_data.append(entry)

            # Write the modified data line to the output file
            outfile.write('\t'.join(new_data) + '\n')

print(f"New file generated: {output_file}")
