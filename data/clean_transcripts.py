import os
import re
import argparse

from localspelling import convert_spelling

from tqdm import tqdm


def standardize_text(text: str):
  """
  Standardize the syntax of a given string.
  - Trim extra spaces
  - Ensure consistent punctuation (e.g., no spaces before punctuation)
  - Remove trailing parentheses
  - Remove quotation marks not part of contractions
  - Change British to American spelling
  - Set all characters to lowercase
  """
  # Remove leading/trailing whitespace and normalize internal spaces
  text = re.sub(r"\s+", " ", text.strip())

  # Remove trailing parenthesis or extra symbols
  text = re.sub(r"\s*\)+$", "", text)

  # Ensure no spaces before punctuation (e.g., " .", " ?")
  text = re.sub(r"\s+([.,!?])", r"\1", text)

  # Remove double quotes and fancy quotes
  text = re.sub(r"[‘’“”\"]", "", text)
  
  # Remove single quotes not part of contractions
  text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)

  # Final removal of leading/trailing whitespace
  text = text.strip()
  
  # Convert British to American spelling
  text = convert_spelling(text, 'us')
  
  # # Capitalize leading character
  # text = text[0].capitalize() + text[1:] if text else text

  # All lowercase
  text = text.lower()

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('src', type=str, metavar="STR",
                      help='Source directory containing input text files.')
  parser.add_argument('dst', type=str, metavar="STR",
                      help='Destination directory for cleaned-up text files.')
  
  args = parser.parse_args()
  input_directory = args.src
  output_directory = args.dst

  process_transcripts(input_directory, output_directory)