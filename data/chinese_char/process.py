"""
Prepare the Chinese dataset for character-level language modeling.
Map Chinese characters to integers and generate character images.
"""
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def generate_character_images(chars, stoi, output_dir='char_images', 
                              font_path=None, img_size=64, font_size=48):
    """
    Generate images for each character
    
    Args:
        chars: List of characters
        stoi: Character to ID mapping
        output_dir: Image output directory
        font_path: Font file path, None to use system default font
        img_size: Image size (square)
        font_size: Font size
    """
    # Create output directory
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"\n📸 Generating character images...")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Font size: {font_size}")
    
    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            print(f"   Using font: {font_path}")
        else:
            font = ImageFont.load_default()
            print("   ⚠️  Using default font (may not support Chinese, recommend specifying font path)")
    except Exception as e:
        print(f"   ⚠️  Font loading failed: {e}")
        font = ImageFont.load_default()
    
    # Generate image mapping information
    char_to_image = {}
    id_to_image = {}
    
    # Generate image for each character
    for i, char in enumerate(chars):
        char_id = stoi[char]
        
        # Create white background image
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate centered text position
        # Use textbbox to get text bounding box
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img_size - text_width) // 2 - bbox[0]
        y = (img_size - text_height) // 2 - bbox[1]
        
        # Draw black text
        draw.text((x, y), char, fill='black', font=font)
        
        # Save image, filename is character ID
        img_filename = f"{char_id:05d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img.save(img_path)
        
        # Record mapping relationship
        char_to_image[char] = img_filename
        id_to_image[char_id] = img_filename
        
        # Display progress
        if (i + 1) % 500 == 0 or i == len(chars) - 1:
            print(f"   Progress: {i + 1}/{len(chars)} ({(i + 1) / len(chars) * 100:.1f}%)")
    
    # Save mapping relationship to JSON file
    mapping_file = os.path.join(output_dir, 'char_image_mapping.json')
    mapping_data = {
        'char_to_image': char_to_image,
        'id_to_image': {str(k): v for k, v in id_to_image.items()},  # JSON keys must be strings
        'image_size': img_size,
        'font_size': font_size,
        'total_chars': len(chars)
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Image generation complete!")
    print(f"   Image directory: {img_dir}")
    print(f"   Mapping file: {mapping_file}")
    print(f"   Total images: {len(chars)}")
    
    return char_to_image, id_to_image


def prepare_chinese_dataset(input_file='filtered_data.txt', output_dir='.',
                           generate_images=True, font_path=None, 
                           img_size=64, font_size=48):
    """
    Process Chinese dataset, convert to nanoGPT-compatible format, and generate character images
    
    Args:
        input_file: Input text file path
        output_dir: Output directory
        generate_images: Whether to generate character images
        font_path: Font file path
        img_size: Image size
        font_size: Font size
    """
    
    input_file_path = os.path.join(output_dir, input_file)
    
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found {input_file_path}")
        print("Please prepare the input text file first")
        return False
    
    # Read data
    print(f"Reading file: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Total characters in dataset: {len(data):,}")
    
    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size:,}")
    print("First 50 characters:", ''.join(chars[:50]))
    print("Last 50 characters:", ''.join(chars[-50:]))
    
    # Create character to integer mapping
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        """Encoder: string -> list of integers"""
        return [stoi[c] for c in s]
    
    def decode(l):
        """Decoder: list of integers -> string"""
        return ''.join([itos[i] for i in l])
    
    # Generate character images
    char_to_image = None
    id_to_image = None
    if generate_images:
        char_to_image, id_to_image = generate_character_images(
            chars, stoi, output_dir, font_path, img_size, font_size
        )
    
    # Create train and validation split
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    print(f"\nTraining set characters: {len(train_data):,}")
    print(f"Validation set characters: {len(val_data):,}")
    
    # Encode to integers
    print("Encoding training set...")
    train_ids = encode(train_data)
    print("Encoding validation set...")
    val_ids = encode(val_data)
    
    print(f"Training set tokens: {len(train_ids):,}")
    print(f"Validation set tokens: {len(val_ids):,}")
    
    # Convert to numpy array and save
    print("Saving binary files...")
    
    # Choose appropriate data type based on vocabulary size
    if vocab_size < 65536:  # 2^16
        dtype = np.uint16
        print("Using uint16 data type")
    else:
        dtype = np.uint32
        print("Using uint32 data type (large vocabulary)")
    
    train_ids = np.array(train_ids, dtype=dtype)
    val_ids = np.array(val_ids, dtype=dtype)
    
    # Save binary files
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    # Save metadata (including image mapping)
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars,
        'char_to_image': char_to_image,
        'id_to_image': id_to_image,
    }
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n" + "=" * 60)
    print("✅ Data preprocessing complete!")
    print(f"📁 Output files:")
    print(f"   - train.bin: {len(train_ids):,} tokens")
    print(f"   - val.bin: {len(val_ids):,} tokens") 
    print(f"   - meta.pkl: vocabulary size {vocab_size:,}")
    if generate_images:
        print(f"   - images/: {vocab_size:,} character images")
        print(f"   - char_image_mapping.json: character-image mapping")
    print("=" * 60)
    
    # Verify encoding/decoding functionality
    print("\n🔍 Verifying encoding/decoding:")
    test_text = data[:100]
    encoded = encode(test_text)
    decoded = decode(encoded)
    
    print(f"Original text: {test_text[:50]}...")
    print(f"Encoded result: {encoded[:10]}...")
    print(f"Decode verification: {'✅ Passed' if test_text == decoded else '❌ Failed'}")
    
    # Display sample mapping
    if generate_images and len(chars) > 0:
        print(f"\n📋 Character mapping examples:")
        for i in range(min(5, len(chars))):
            char = chars[i]
            char_id = stoi[char]
            img_name = id_to_image[char_id] if id_to_image else "N/A"
            print(f"   '{char}' -> ID: {char_id} -> Image: {img_name}")
    
    return True


def main():
    """Main function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Chinese dataset preprocessing and character image generation')
    parser.add_argument('--input_file', default='filtered_data.txt',
                       help='Input text file path')
    parser.add_argument('--output_dir', default='.',
                       help='Output directory')
    parser.add_argument('--no-images', action='store_true',
                       help='Do not generate character images')
    parser.add_argument('--font', type=str, default='OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf',
                       help='Font file path')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Image size (default: 64)')
    parser.add_argument('--font-size', type=int, default=48,
                       help='Font size (default: 48)')
    
    args = parser.parse_args()
    
    print("🇨🇳 Chinese Dataset Preprocessing and Image Generation Tool")
    print("=" * 60)
    
    # Process dataset
    success = prepare_chinese_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        generate_images=not args.no_images,
        font_path=args.font,
        img_size=args.img_size,
        font_size=args.font_size
    )
    
    if success:
        print("\n🚀 Processing complete!")
        print("\n📦 Generated files:")
        print("   - train.bin, val.bin: training and validation data")
        print("   - meta.pkl: contains vocabulary and image mapping")
        print("   - images/: character image folder")
        print("   - char_image_mapping.json: character-image mapping JSON")
    else:
        print("\n❌ Preprocessing failed")


if __name__ == "__main__":
    main()