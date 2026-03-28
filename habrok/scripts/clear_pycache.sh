#!/bin/bash

# Description: Recursively find and remove all __pycache__ directories.

echo "Searching for __pycache__ directories..."
find $HOME/Kenku -type d -name "__pycache__" -print -exec rm -r {} +

echo "All __pycache__ directories have been removed."
