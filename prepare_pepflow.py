import os
import json
from tqdm import tqdm

def read_fasta(file_path):
    """读取 fasta 文件并提取纯氨基酸序列"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 过滤掉注释行，拼接序列
    seq = "".join([line.strip() for line in lines if not line.startswith(">")])
    return seq

def main():
    base_dir = "/workspace/guest/cyh/Dataset/PepFlow"
    
    # 获取所有的复合体文件夹
    folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # 准备输出目录
    output_dir = "./dataset/Fine_Diffusion"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pepflow_fine_sequence.jsonl")
    
    valid_count = 0
    
    # 遍历提取数据并直接写入 jsonl
    print(f"🔍 正在扫描 {base_dir} ...")
    with open(output_file, "w") as f:
        for folder_name in tqdm(folder_names, desc="打包 PepFlow 序列"):
            folder_path = os.path.join(base_dir, folder_name)
            rec_fasta = os.path.join(folder_path, "receptor.fasta")
            pep_fasta = os.path.join(folder_path, "peptide.fasta")
            
            prot_seq = read_fasta(rec_fasta)
            pep_seq = read_fasta(pep_fasta)
            
            # 只要靶点和多肽都存在，就写入文件
            if prot_seq and pep_seq:
                json_line = {"src": prot_seq, "trg": pep_seq}
                f.write(json.dumps(json_line) + "\n")
                valid_count += 1
                
    print(f"\n🎉 大功告成！共成功打包 {valid_count} 条数据。")
    print(f"💾 文件已完美对齐并保存至: {output_file}")

if __name__ == '__main__':
    main()