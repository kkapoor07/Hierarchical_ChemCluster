# scripts/utils/split_smiles_file.py
import argparse
import os
import sys
import math

def split_file(input_file, num_parts, output_pattern="{basename}_part{part_num}.smi"):
    """
    Splits a large text file (like SMILES) into a specified number of smaller parts.

    Args:
        input_file (str): Path to the input file.
        num_parts (int): Number of parts to split the file into.
        output_pattern (str): F-string pattern for output filenames.
                              Available placeholders: {basename}, {part_num}.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return False
    if num_parts <= 0:
        print(f"Error: Number of parts must be positive.", file=sys.stderr)
        return False

    try:
        print(f"Calculating lines in {input_file}...")
        # Count lines efficiently
        line_count = 0
        with open(input_file, 'rb') as f: # Open in binary for faster counting
            while True:
                buf = f.read(65536) # Read in 64k chunks
                if not buf:
                    break
                line_count += buf.count(b'\n')
        print(f"Found {line_count} lines.")

        if line_count == 0:
            print("Input file is empty. No splitting needed.")
            return True

        lines_per_part = math.ceil(line_count / num_parts) # Use ceil to ensure all lines are covered
        print(f"Splitting into {num_parts} parts with approx. {lines_per_part} lines per part.")

        part_num = 1
        lines_written_current_part = 0
        outfile = None
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                # Open new output file if needed
                if outfile is None or (lines_written_current_part >= lines_per_part and part_num < num_parts):
                    if outfile:
                        outfile.close()
                        print(f"Finished writing part {part_num}")
                        part_num += 1
                        lines_written_current_part = 0

                    output_filename = output_pattern.format(basename=base_name, part_num=part_num)
                    print(f"Opening part {part_num}: {output_filename}")
                    outfile = open(output_filename, 'w', encoding='utf-8')

                outfile.write(line)
                lines_written_current_part += 1

        if outfile:
            outfile.close()
            print(f"Finished writing part {part_num}")

        print("File splitting complete.")
        return True

    except Exception as e:
        print(f"An error occurred during splitting: {e}", file=sys.stderr)
        # Clean up potentially partially written files? Might be complex.
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large text file into multiple smaller parts.")
    parser.add_argument("input_file", help="Path to the large input file.")
    parser.add_argument("num_parts", type=int, help="Number of parts to split the file into.")
    parser.add_argument("-o", "--output_pattern", default="{basename}_part{part_num}.smi",
                        help="Output filename pattern. Use {basename} and {part_num}. Default: '{basename}_part{part_num}.smi'")

    args = parser.parse_args()

    if not split_file(args.input_file, args.num_parts, args.output_pattern):
        sys.exit(1)
