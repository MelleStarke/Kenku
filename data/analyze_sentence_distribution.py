import os
import difflib

from tqdm import tqdm

from itertools import islice

from __init__ import *

def analyze_distribution(transcript_dir_root):
    # Example structure:
    # transcript/
    #   p225/
    #     p225_001.txt
    #     p225_002.txt
    #   p226/
    #     p226_001.txt
    #     ...
    
    # Dictionary mapping sentence text -> count of how many times it appears
    sentence_counts = {}
    
    # Traverse each speaker folder
    for speaker_dir in os.listdir(transcript_root):
        speaker_path = os.path.join(transcript_root, speaker_dir)
        
        # Only process if it's really a directory
        if not os.path.isdir(speaker_path):
            continue
        
        # Read each .txt transcript file
        for filename in os.listdir(speaker_path):
            if filename.endswith(".txt"):
                transcript_path = os.path.join(speaker_path, filename)
                text = None
                
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    
                # Increment count for this exact sentence
                if text not in sentence_counts:
                    sentence_counts[text] = 0
                sentence_counts[text] += 1
    
    # Number of unique sentences
    num_unique_sentences = len(sentence_counts)
    print(f"Total unique sentences: {num_unique_sentences}")
    
    # Sort sentences by their frequency in descending order
    sorted_by_count = sorted(sentence_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print out how many samples each sentence has
    # (You can limit how many you print, in case there are many)
    print("\nSentence counts:")
    for i, (sentence, count) in enumerate(sorted_by_count):
        print(f"{i:2}. Count = {count} | {sentence}")
        
        
def compare_sentences(log_filepath):
  # Load and analyze the file content to identify syntactically different but semantically equivalent sentences

  with open(log_filepath, "r") as file:
      lines = file.readlines()

  # Extract sentences and counts from the lines
  sentence_counts = {}
  for line in lines:
      if "|" in line:
          try:
              parts = line.strip().split("|")
              count_part = parts[0].strip()
              sentence_part = parts[1].strip()
              count = int(count_part.split("=")[1].strip())
              sentence_counts[sentence_part] = count
          except (IndexError, ValueError):
              print(f"Error found in `{line}`")
              continue

  # Create a normalized version of sentences for comparison
  potential_matches = {}

  # Check for closely matching sentences using difflib
  for i, (sentence, count) in tqdm(enumerate(sentence_counts.items()), total = len(sentence_counts)):
      for other_sentence, other_count in islice(sentence_counts.items(), i, None):
          if sentence != other_sentence:
              similarity = difflib.SequenceMatcher(None, sentence, other_sentence).ratio()
              if similarity > 0.9:  # High similarity threshold
                  potential_matches[(sentence, other_sentence)] = similarity
                  print()
                  print(f"Count: {count}\t| {sentence}\nCount: {other_count}\t| {other_sentence}\n---")

  return potential_matches


if __name__ == "__main__":
  standardized_transcript_dir = os.path.join(VCTK_PATH, "transcript_standardized")
  sentence_distribution_log_path = "./logs/standardized_sentence_distribution.log"
  
  # compare_sentences(sentence_distribution_log_path)
  # analyze_distribution(standardized_transcript_dir)
