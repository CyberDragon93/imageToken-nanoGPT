"""
Prepare the Chinese dataset for character-level language modeling.
Map Chinese characters to integers and generate character images.
Unsupported characters are mapped to space character image.
"""
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def check_char_support(char, font, reference_char='?'):
    """
    Check if font supports a character by comparing with a known replacement character
    
    Strategy: If font doesn't support a char, it often renders the same replacement 
    glyph (like □ or ?). We compare the target char with a definitely unsupported char.
    
    Args:
        char: Character to check
        font: PIL ImageFont object
        reference_char: A character we know is unsupported (used as comparison)
        
    Returns:
        bool: True if supported, False otherwise
    """
    try:
        img_size = 32  # Small size for faster comparison
        
        # Render the character we're testing
        test_img = Image.new('L', (img_size, img_size), color=255)  # Grayscale
        draw = ImageDraw.Draw(test_img)
        
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # If no dimension, definitely not supported
        if text_width <= 0 or text_height <= 0:
            return False
        
        x = (img_size - text_width) // 2 - bbox[0]
        y = (img_size - text_height) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
        
        # Render a definitely unsupported character (use a rare Unicode)
        # Use a character from Private Use Area that won't be in any font
        unsupported_char = '\uE000'  # Private Use Area
        ref_img = Image.new('L', (img_size, img_size), color=255)
        ref_draw = ImageDraw.Draw(ref_img)
        
        ref_bbox = ref_draw.textbbox((0, 0), unsupported_char, font=font)
        ref_width = ref_bbox[2] - ref_bbox[0]
        ref_height = ref_bbox[3] - ref_bbox[1]
        
        ref_x = (img_size - ref_width) // 2 - ref_bbox[0]
        ref_y = (img_size - ref_height) // 2 - ref_bbox[1]
        ref_draw.text((ref_x, ref_y), unsupported_char, fill=0, font=font)
        
        # Compare the two images
        test_array = np.array(test_img)
        ref_array = np.array(ref_img)
        
        # Calculate similarity (normalized cross-correlation would be better, but this is simpler)
        # If images are very similar, the test char is probably also unsupported
        diff = np.sum(np.abs(test_array.astype(int) - ref_array.astype(int)))
        total_pixels = img_size * img_size * 255  # Max possible diff
        similarity = 1 - (diff / total_pixels)
        
        # If similarity > 0.9, they're rendering the same replacement glyph
        if similarity > 0.999:
            return False
        
        # Additional check: very low pixel coverage suggests not rendered
        black_pixels = np.sum(test_array < 200)
        coverage = black_pixels / (img_size * img_size)
        
        if coverage < 0.01:  # Less than 1% coverage
            return False
        
        return True
        
    except Exception as e:
        # If any error, assume unsupported
        return False


def generate_character_images(chars, stoi, output_dir='char_images', 
                              font_path=None, img_size=64, font_size=48):
    """
    Generate images for each character
    Unsupported characters are mapped to space character image
    
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
    
    # Statistics
    unsupported_chars = []
    supported_count = 0
    space_char_id = stoi.get(' ', None)
    space_img_filename = None
    
    # Generate image for each character
    for i, char in enumerate(chars):
        char_id = stoi[char]
        
        # Check if font supports this character
        is_supported = check_char_support(char, font)
        
        if is_supported:
            # Supported character: generate image normally
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
            
            # If this is space character, save filename for later use
            if char == ' ':
                space_img_filename = img_filename
            
            supported_count += 1
        else:
            # Unsupported character: will map to space image
            # We'll set the filename after space image is generated
            img_filename = None  # Placeholder
            unsupported_chars.append((char, char_id))
        
        # Record mapping relationship (will update unsupported ones later)
        char_to_image[char] = img_filename
        id_to_image[char_id] = img_filename
        
        # Display progress
        if (i + 1) % 500 == 0 or i == len(chars) - 1:
            print(f"   Progress: {i + 1}/{len(chars)} ({(i + 1) / len(chars) * 100:.1f}%)")
    
    # Update unsupported characters to map to space image
    if space_img_filename:
        for char, char_id in unsupported_chars:
            char_to_image[char] = space_img_filename
            id_to_image[char_id] = space_img_filename
        print(f"\n   ✅ Mapped {len(unsupported_chars)} unsupported chars to space image: {space_img_filename}")
    elif unsupported_chars:
        print(f"\n   ⚠️  WARNING: {len(unsupported_chars)} unsupported chars, but no space character in vocabulary!")
        print(f"   Creating a blank image as fallback...")
        
        # Create a blank fallback image
        blank_img = Image.new('RGB', (img_size, img_size), color='white')
        fallback_filename = "fallback_blank.png"
        fallback_path = os.path.join(img_dir, fallback_filename)
        blank_img.save(fallback_path)
        
        for char, char_id in unsupported_chars:
            char_to_image[char] = fallback_filename
            id_to_image[char_id] = fallback_filename
    
    # Save mapping relationship to JSON file
    mapping_file = os.path.join(output_dir, 'char_image_mapping.json')
    mapping_data = {
        'char_to_image': char_to_image,
        'id_to_image': {str(k): v for k, v in id_to_image.items()},  # JSON keys must be strings
        'image_size': img_size,
        'font_size': font_size,
        'total_chars': len(chars),
        'supported_chars': supported_count,
        'unsupported_chars': len(unsupported_chars),
        'space_image': space_img_filename,
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    # Save unsupported characters list
    if unsupported_chars:
        unsupported_file = os.path.join(output_dir, 'unsupported_chars.txt')
        with open(unsupported_file, 'w', encoding='utf-8') as f:
            f.write(f"Unsupported characters total: {len(unsupported_chars)}\n")
            f.write(f"All mapped to: {space_img_filename or 'fallback_blank.png'}\n")
            f.write("=" * 60 + "\n\n")
            for char, char_id in unsupported_chars:
                unicode_code = f"U+{ord(char):04X}"
                f.write(f"'{char}' -> {unicode_code} (ID: {char_id})\n")
    
    print(f"\n✅ Image generation complete!")
    print(f"   Image directory: {img_dir}")
    print(f"   Mapping file: {mapping_file}")
    print(f"   Total chars: {len(chars)}")
    print(f"   ✅ Supported: {supported_count} ({supported_count/len(chars)*100:.1f}%)")
    print(f"   ⚠️  Unsupported (mapped to space): {len(unsupported_chars)} ({len(unsupported_chars)/len(chars)*100:.1f}%)")
    
    # Show some examples of unsupported characters
    if unsupported_chars:
        print(f"\n📋 Unsupported characters (first 20):")
        for char, char_id in unsupported_chars[:20]:
            print(f"   '{char}' (U+{ord(char):04X}, ID:{char_id}) -> {space_img_filename or 'fallback_blank.png'}")
        if len(unsupported_chars) > 20:
            print(f"   ... and {len(unsupported_chars) - 20} more")
        print(f"\n   💾 Full list saved to: unsupported_chars.txt")
    
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
        print(f"   - images/: character images")
        print(f"   - char_image_mapping.json: character-image mapping")
        print(f"   - unsupported_chars.txt: list of unsupported characters")
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
        print("   - unsupported_chars.txt: unsupported characters list")
    else:
        print("\n❌ Preprocessing failed")


if __name__ == "__main__":
    main()