import csv

def process_row(row, sample_number):
    # Extract EEG data; ignore other data like 'Label'
    eeg_data = row[1:9]  # EEG_1 to EEG_8

    # Format each EEG data point as a string
    formatted_eeg_data = ' '.join(map(str, eeg_data))

    # Combine the sample number and EEG data
    formatted_row = f"{sample_number}, {formatted_eeg_data}"

    return formatted_row





input_file = 'data/collected_data.csv'
output_file = 'output_file.txt'

# OpenBCI file header
header_lines = [
    "%OpenBCI Raw EEG Data",
    "%Number of channels = 8",
    "%Sample Rate = 250 Hz",
    "%Board = OpenBCI_GUI$BoardCytonSerial"
]

# Read the CSV file
with open(input_file, mode='r') as infile, open(output_file, mode='w') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header to the output file
    for line in header_lines:
        outfile.write(line + '\n')

    # Skip the header of the CSV file
    next(reader)

    sample_number = 0
    for row in reader:
        if sample_number != 0:  # Skip the header line of the CSV
            formatted_row = process_row(row, sample_number%255)
            outfile.write(formatted_row + '\n')
        sample_number += 1



