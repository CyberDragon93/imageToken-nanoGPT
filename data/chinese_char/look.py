import pickle
import json

def inspect_meta(meta_path='meta.pkl'):
    """完整查看 meta.pkl 文件内容"""
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    print("=" * 60)
    print("📋 Meta 文件内容:")
    print("=" * 60)
    
    # 基本信息
    print(f"\n📊 基本信息:")
    print(f"  词汇表大小: {meta['vocab_size']:,}")
    print(f"  字符总数: {len(meta['chars']):,}")
    
    # 字符示例
    print(f"\n🔤 字符示例:")
    print(f"  前 50 个: {''.join(meta['chars'][:50])}")
    print(f"  后 50 个: {''.join(meta['chars'][-50:])}")
    
    # 映射示例
    print(f"\n🔗 映射示例 (前 10 个):")
    for i in range(min(20, len(meta['chars']))):
        char = meta['chars'][i]
        char_id = meta['stoi'][char]
        print(f"  {i+1}. '{char}' -> ID: {char_id}")
    
    # 图像信息
    if 'char_to_image' in meta and meta['char_to_image']:
        print(f"\n🖼️  图像映射:")
        print(f"  已生成图像数量: {len(meta['char_to_image']):,}")
        print(f"\n  示例:")
        for i in range(min(20, len(meta['chars']))):
            char = meta['chars'][i]
            img = meta['char_to_image'].get(char, 'N/A')
            print(f"    '{char}' -> {img}")
    
    print("\n" + "=" * 60)
    
    return meta

# 使用
if __name__ == "__main__":
    meta = inspect_meta('meta.pkl')