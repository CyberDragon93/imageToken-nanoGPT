#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

def filter_jsonl(input_files, output_file="filtered_data.txt", target_chars=1000000, min_score=0.9):
    """
    Filter multiple JSONL files based on score and character count, then save to a text file.
    
    Args:
        input_files: list of input JSONL file paths
        output_file: output text file path
        target_chars: target number of characters (default 1,000,000)
        min_score: minimum score threshold (default 0.9)
    """
    
    total_chars = 0
    processed_lines = 0
    filtered_lines = 0
    
    print(f"begin processing {len(input_files)} file(s)")
    print(f"filter condition: score >= {min_score}")
    print(f"target characters: {target_chars:,}")
    print("-" * 50)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            
            for file_idx, input_file in enumerate(input_files, 1):
                print(f"\nProcessing file {file_idx}/{len(input_files)}: {input_file}")
                
                if not Path(input_file).exists():
                    print(f"warning: file {input_file} not found, skipping.")
                    continue
                
                with open(input_file, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        processed_lines += 1
                        
                        try:
                            data = json.loads(line.strip())
                            
                            score = data.get('score', 0)
                            if score < min_score:
                                continue
                            
                            text = data.get('text', '').strip()
                            if not text:
                                continue
                            
                            text_len = len(text)
                            if total_chars + text_len + 1 > target_chars:
                                remaining_chars = target_chars - total_chars - 1
                                if remaining_chars > 0:
                                    text = text[:remaining_chars]
                                    outfile.write(text + '\n\n')
                                    total_chars += len(text) + 1
                                    filtered_lines += 1
                                print(f"Reached target character count, stopping.")
                                return True
                            
                            outfile.write(text + '\n\n')
                            total_chars += text_len + 1
                            filtered_lines += 1
                            
                        except json.JSONDecodeError:
                            print(f"warning: line {line_num} in {input_file} is not valid JSON, skipping.")
                            continue
                        except Exception as e:
                            print(f"warning: error processing line {line_num} in {input_file}: {e}, skipping.")
                            continue
    
    except Exception as e:
        print(f"error: unexpected error: {e}")
        return False
    
    print("-" * 50)
    print("All done!")
    print(f"Processed lines: {processed_lines:,}")
    print(f"Filtered lines: {filtered_lines:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Output file: {output_file}")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter JSONL file(s) based on score and character count.")
    parser.add_argument('-i', '--input_files', nargs='+', default=["part-0000.jsonl", "part-0001.jsonl", "part-0002.jsonl"], 
                        help="Input JSONL file path(s), can specify multiple files")
    parser.add_argument('-o', '--output_file', default="filtered_data.txt", 
                        help="Output text file path")
    parser.add_argument('-c', '--target_chars', type=int, default=3000000000, 
                        help="Target number of characters (default: 100,000,000)")
    parser.add_argument('-s', '--min_score', type=float, default=0.5, 
                        help="Minimum score threshold (default: 0.9)")

    args = parser.parse_args()
    
    # 执行过滤
    success = filter_jsonl(args.input_files, args.output_file, args.target_chars, args.min_score)
    
    if success:
        print(f"\n✅ The filtered data has been saved to {args.output_file}.")
    else:
        print("\n❌ Filtering failed.")

if __name__ == "__main__":
    main()