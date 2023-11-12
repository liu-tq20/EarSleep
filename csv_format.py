import csv


# Define the function to process the CSV file
def process_csv(input_file_path, output_file_path):
    # Open the input CSV file
    with open(input_file_path, mode='r') as csvfile:
        # Read the CSV file
        csvreader = csv.reader(csvfile)
        # Open the output CSV file
        with open(output_file_path, mode='w') as outfile:
            csvwriter = csv.writer(outfile)
            # Iterate over each row in the input CSV file
            for row in csvreader:
                # Check if the row has at least one element to process
                if len(row) > 0:
                    # Remove the date from the last element of the row
                    first_element = row[0]
                    # Split the last element by space and remove the date part
                    first_element_without_date = ' '.join(first_element.split()[:-1])
                    # Update the last element of the row
                    row[0] = first_element_without_date
                # Write the updated row to the output CSV file
                csvwriter.writerow(row)

# Paths for the input and output CSV files
input_csv_path = './data/BPM-18-1.csv'
output_csv_path = './data/BPM-18.csv'

# Call the function with the file paths
process_csv(input_csv_path, output_csv_path)
