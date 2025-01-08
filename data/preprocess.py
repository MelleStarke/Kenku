import os
import re

from localspelling import convert_spelling

from tqdm import tqdm

from __init__ import *

def standardize_text(text: str):
  """
  Standardize the syntax of a given text.
  - Trim extra spaces
  - Ensure consistent punctuation (e.g., no spaces before punctuation)
  - Remove trailing parentheses or extra symbols
  - Fix inconsistent quotes
  """
  # Remove leading/trailing whitespace and normalize internal spaces
  text = re.sub(r"\s+", " ", text.strip())

  # Remove trailing parenthesis or extra symbols
  text = re.sub(r"\s*\)+$", "", text)

  # Ensure no spaces before punctuation (e.g., ". ", "? ")
  text = re.sub(r"\s+([.,!?])", r"\1", text)

  # Remove all quotation marks (single or double)
  text = re.sub(r"[‘’`“”'\"]", "", text)

  # Final removal of leading/trailing whitespace
  text = text.strip()
  
  # Convert British to American spelling
  text = convert_spelling(text, 'us')
  
  # Capitalize leading character
  text = text[0].capitalize() + text[1:] if text else text

  return text

def process_transcripts(input_dir, output_dir):
  """
  Process all transcript .txt files in the input directory, normalize their contents,
  and optionally save to an output directory.
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  all_files = list(os.walk(input_dir))

  for root, _, files in tqdm(all_files, total = len(all_files)):
    for file in files:
      if file.endswith(".txt"):
        input_path = os.path.join(root, file)
        output_path = os.path.join(root.replace(input_dir, output_dir), file)
        
        if not os.path.exists(os.path.dirname(output_path)):
          os.makedirs(os.path.dirname(output_path))

        # Read, normalize, and write the transcript
        with open(input_path, "r", encoding="utf-8") as f:
          text = f.read()

        standardized_text = standardize_text(text)

        with open(output_path, "w", encoding="utf-8") as f:
          f.write(standardized_text)

        print(f"Processed: {text} -> {standardized_text}")

if __name__ == "__main__":
  # Define the input directory containing transcript .txt files
  input_directory = os.path.join(VCTK_PATH, "transcript")  # Replace with the actual path to your transcripts

  # Optionally define an output directory for saving the standardized files
  # Leave as None to overwrite the original files
  output_directory = os.path.join(VCTK_PATH, "transcript_standardized")  # Replace with your desired output directory or None

  process_transcripts(input_directory, output_directory)