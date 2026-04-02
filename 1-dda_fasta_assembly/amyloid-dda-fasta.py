import os
import re

# Define the input file names
HUMAN_CANONICAL_FASTA = "UP000005640_9606.fasta"
ISOFORM_FASTA = "uniprot_sprot_varsplic.fasta"
MISSENSE_VARIANTS_LIST = "humsavar.txt"
PROTEIN_IDS_LIST = "protein_ids.txt"
SUPP_TARGETS_FASTA = "supp_targets.fasta"

# Function to read protein IDs from a file and return them as a list
def read_protein_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Function to create a directory if it does not exist
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Function to create output folder and subdirectories for protein IDs
def create_output_directories(protein_ids, output_dir='output'):
    create_directory(output_dir)  # Create the main output directory
    for protein_id in protein_ids:
        create_directory(os.path.join(output_dir, protein_id))  # Create subdirectory for each protein ID

# Function to copy the canonical sequence for a given protein ID to a new file
def copy_canonical_sequence(protein_id, fasta_file, output_dir):
    found = False
    with open(fasta_file, 'r') as file:
        sequence_lines = []
        for line in file:
            if line.startswith('>') and found:
                break  # Stop if the next protein entry is found
            if found:
                sequence_lines.append(line)
            if line.startswith(f'>{protein_id}|') or f'|{protein_id}|' in line:
                sequence_lines.append(line)
                found = True

        if found:
            output_file_path = os.path.join(output_dir, protein_id, f'{protein_id}_canonical.fasta')
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(sequence_lines)
        else:
            print(f'Warning: {protein_id} was not found in canonical fasta.')

# Function to copy isoform sequences for a given protein ID to a new file
def copy_isoform_sequences(protein_id, fasta_file, output_dir):
    with open(fasta_file, 'r') as file:
        sequence_lines = []
        collect_lines = False
        for line in file:
            if line.startswith('>'):
                if f'|{protein_id}-' in line:
                    collect_lines = True  # Start collecting lines for this isoform
                else:
                    if collect_lines:
                        # We've reached a new protein entry, so stop collecting
                        break
            if collect_lines:
                sequence_lines.append(line)

        if sequence_lines:
            output_file_path = os.path.join(output_dir, protein_id, f'{protein_id}_isoforms.fasta')
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(sequence_lines)
        else:
            print(f'Warning: {protein_id} is missing isoforms.')

# Function to copy protein variants for a given protein ID to a new file
def copy_protein_variants(protein_id, variants_file, output_dir):
    with open(variants_file, 'r') as file:
        variant_lines = [line for line in file if protein_id in line]

    if variant_lines:
        output_file_path = os.path.join(output_dir, protein_id, f'{protein_id}_variants.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(variant_lines)
    else:
        print(f'Warning: No variants found for {protein_id}.')

# Mapping from three-letter to one-letter amino acid codes
amino_acid_mapping = {
    'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E',
    'Phe': 'F', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Lys': 'K', 'Leu': 'L', 'Met': 'M', 'Asn': 'N',
    'Pro': 'P', 'Gln': 'Q', 'Arg': 'R', 'Ser': 'S',
    'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y'
}

# Function to determine the highest isoform number or default to 1
def get_highest_isoform_number(protein_id, output_dir):
    isoforms_file_path = os.path.join(output_dir, protein_id, f'{protein_id}_isoforms.fasta')
    highest_number = 1  # Default to 1 if no isoforms file is found
    isoform_numbers = []  # List to store all isoform numbers found

    if os.path.exists(isoforms_file_path):
        with open(isoforms_file_path, 'r') as isoforms_file:
            for line in isoforms_file:
                if line.startswith('>'):
                    match = re.search(r'\|' + re.escape(protein_id) + r'-(\d+)\|', line)
                    if match:
                        isoform_number = int(match.group(1))
                        isoform_numbers.append(isoform_number)

        if isoform_numbers:  # If we found any isoform numbers
            highest_number = max(isoform_numbers)
        else:  # No isoform numbers found, default to "-1"
            highest_number = 1

    # If the only isoform number is "-1", we start numbering from "-3"
    if highest_number == 1 and isoform_numbers:
        return 2  # Start from "-2" after "-1"
    else:
        return highest_number  # Continue numbering from the highest isoform number

# Function to convert space-separated variants file to tab-delimited format
def convert_to_tab_delimited(protein_id, variants_file, output_dir):
    input_file_path = os.path.join(output_dir, protein_id, variants_file)
    output_file_path = os.path.join(output_dir, protein_id, f'{protein_id}_variants_tab.txt')
    
    # Get the highest isoform number or default to 1
    next_isoform_number = get_highest_isoform_number(protein_id, output_dir)
    
    # Check if the input file exists before proceeding
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                # Split the line into columns based on whitespace
                columns = line.split()
                # Only include the first four columns
                first_four_columns = columns[:4]
                # Convert the amino acid substitution in the fourth column
                aa_substitution = first_four_columns[3][2:]  # Remove the 'p.' prefix
                original_aa_3 = aa_substitution[:3]
                new_aa_3 = aa_substitution[-3:]
                position = aa_substitution[3:-3]
                original_aa_1 = amino_acid_mapping[original_aa_3]
                new_aa_1 = amino_acid_mapping[new_aa_3]
                first_four_columns[3] = original_aa_1 + position + new_aa_1
                # Append the position, original amino acid, and new amino acid as new columns
                first_four_columns.append(position)
                first_four_columns.append(original_aa_1)
                first_four_columns.append(new_aa_1)
                # Create the 8th column by concatenating the 3rd and 4th columns with an underscore
                eighth_column = first_four_columns[2] + '_' + first_four_columns[3]
                first_four_columns.append(eighth_column)
                # Create the 9th column with the incremented isoform number
                ninth_column = f'{protein_id}-{next_isoform_number + 1}'
                first_four_columns.append(ninth_column)
                next_isoform_number += 1  # Increment for the next variant
                # Join the columns using tabs to create a tab-delimited line
                tab_delimited_line = '\t'.join(first_four_columns) + '\n'
                # Write the tab-delimited line to the output file
                output_file.write(tab_delimited_line)
    else:
        print(f'Warning: No variants file found for {protein_id}. Skipping tab-delimited conversion.')

def create_variants_fasta(protein_id, output_dir):
    canonical_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_canonical.fasta')
    variants_tab_path = os.path.join(output_dir, protein_id, f'{protein_id}_variants_tab.txt')
    variants_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_variants.fasta')
    
    # Check if the variants tab file exists before proceeding
    if not os.path.exists(variants_tab_path):
        print(f'No variants tab file found for {protein_id}. Skipping creation of variants fasta.')
        return
    
    with open(canonical_fasta_path, 'r') as canonical_file:
        # Read the header line
        header_line = canonical_file.readline().strip()
        # Read the sequence lines and concatenate them into a single string
        sequence = ''.join(line.strip() for line in canonical_file)
    
    with open(variants_fasta_path, 'w') as variants_fasta_file:
        with open(variants_tab_path, 'r') as variants_tab_file:
            for line in variants_tab_file:
                columns = line.strip().split('\t')
                # Ensure there are enough columns
                if len(columns) == 9:
                    position = int(columns[4])  # Column 5 (index 4) contains the amino acid position
                    original_aa = columns[5]  # Column 6 (index 5) contains the original amino acid
                    new_aa = columns[6]  # Column 7 (index 6) contains the new amino acid
                    variant_id = columns[8]  # Column 9 (index 8) contains the text for the new header
                    variant_info = columns[7]  # Column 8 (index 7) contains the text to append to the header
                    
                    # Check if the original amino acid matches the sequence at the specified position
                    if sequence[position - 1] != original_aa:
                        raise ValueError(f"Error: Original amino acid {original_aa} does not match the sequence at position {position} for {protein_id}")
                    
                    # Replace the original amino acid with the new amino acid
                    modified_sequence = sequence[:position - 1] + new_aa + sequence[position:]
                    
                    # Re-wrap the sequence to 60 characters per line
                    wrapped_sequence = '\n'.join(modified_sequence[i:i+60] for i in range(0, len(modified_sequence), 60))
                    
                    # Adjust the header
                    header_line_modified = re.sub(r'\|.*?\|', f'|{variant_id}|', header_line)
                    header_line_modified += f' {variant_info}\n'
                    
                    # Write the adjusted header and the modified sequence to the variants fasta file
                    variants_fasta_file.write(header_line_modified)
                    variants_fasta_file.write(wrapped_sequence + '\n')
                else:
                    print(f'Warning: Line in {variants_tab_path} does not have 9 columns.')

def create_isoform_variant_fasta(protein_id, output_dir):
    isoforms_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_isoforms.fasta')
    variants_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_variants.fasta')
    isoform_variant_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_isoform_variant.fasta')
    
    # Check if both the isoforms fasta file and the variants fasta file are missing
    if not os.path.exists(isoforms_fasta_path) and not os.path.exists(variants_fasta_path):
        print(f'No isoforms or variants fasta files found for {protein_id}. Skipping creation of isoform variant fasta.')
        return
    
    with open(isoform_variant_fasta_path, 'w') as isoform_variant_fasta_file:
        # If the isoforms fasta file exists, write its content to the isoform variant fasta file
        if os.path.exists(isoforms_fasta_path):
            with open(isoforms_fasta_path, 'r') as isoforms_fasta_file:
                isoform_variant_fasta_file.write(isoforms_fasta_file.read())
        
        # If the variants fasta file exists, write its content to the isoform variant fasta file
        if os.path.exists(variants_fasta_path):
            with open(variants_fasta_path, 'r') as variants_fasta_file:
                isoform_variant_fasta_file.write(variants_fasta_file.read())

def create_amyloid_dda_fasta(output_dir, human_canonical_fasta, supp_targets_fasta, protein_ids_list):
    amyloid_dda_fasta_path = os.path.join(output_dir, 'amyloid_dda.fasta')
    
    with open(amyloid_dda_fasta_path, 'w') as amyloid_dda_fasta_file:
        # Write the content of HUMAN_CANONICAL_FASTA to the amyloid_dda fasta file
        with open(human_canonical_fasta, 'r') as canonical_fasta_file:
            amyloid_dda_fasta_file.write(canonical_fasta_file.read())
        
        # Write the content of SUPP_TARGETS_FASTA to the amyloid_dda fasta file
        with open(supp_targets_fasta, 'r') as supp_targets_fasta_file:
            amyloid_dda_fasta_file.write(supp_targets_fasta_file.read())
        
        # Write the content of each <protein_id>_isoform_variant.fasta to the amyloid_dda fasta file
        for protein_id in protein_ids_list:
            isoform_variant_fasta_path = os.path.join(output_dir, protein_id, f'{protein_id}_isoform_variant.fasta')
            if os.path.exists(isoform_variant_fasta_path):
                with open(isoform_variant_fasta_path, 'r') as isoform_variant_fasta_file:
                    amyloid_dda_fasta_file.write(isoform_variant_fasta_file.read())

# Main execution
if __name__ == "__main__":
    # Read the list of protein IDs
    protein_ids = read_protein_ids(PROTEIN_IDS_LIST)
    
    # Assume 'output' is the output directory in the working directory
    output_dir = 'output'
    
    # Create output directories
    create_output_directories(protein_ids)
    
    # Process each protein ID
    for protein_id in protein_ids:
        copy_canonical_sequence(protein_id, HUMAN_CANONICAL_FASTA, 'output')
        copy_isoform_sequences(protein_id, ISOFORM_FASTA, 'output')
        copy_protein_variants(protein_id, MISSENSE_VARIANTS_LIST, 'output')
        convert_to_tab_delimited(protein_id, f'{protein_id}_variants.txt', 'output')
        create_variants_fasta(protein_id, 'output')
        create_isoform_variant_fasta(protein_id, 'output')
        create_amyloid_dda_fasta(output_dir, HUMAN_CANONICAL_FASTA, SUPP_TARGETS_FASTA, protein_ids)
