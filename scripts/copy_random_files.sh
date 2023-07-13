#!/bin/bash

# Defaults to 50 if not specified
num_files=${3:-50}

# Check if source and destination are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <source_dir> <dest_dir> [num_files]"
    exit 1
fi

# Check if source directory exists
if [ ! -d "$1" ]; then
    echo "Source directory $1 does not exist."
    exit 1
fi

# Check if destination directory exists, if not create it
if [ ! -d "$2" ]; then
    mkdir -p "$2"
fi

# Randomly copy files
find "$1" -type f | shuf -n "$num_files" | while read file; do cp "$file" "$2"; done

echo "$num_files random files have been copied from $1 to $2."
