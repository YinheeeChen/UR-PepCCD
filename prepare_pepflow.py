import os
import json
from tqdm import tqdm

def read_fasta(file_path):
    """Read a FASTA file and return the concatenated amino acid sequence."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Drop header lines and join residues.
    seq = "".join([line.strip() for line in lines if not line.startswith(">")])
    return seq

def main():
    base_dir = "/workspace/guest/cyh/Dataset/PepFlow"
    
    # Collect all complex folders.
    folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # Prepare output directory and output file.
    output_dir = "./dataset/Fine_Diffusion"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pepflow_fine_sequence.jsonl")
    
    valid_count = 0
    
    # Extract receptor/peptide sequence pairs and write JSONL records.
    print(f"Scanning {base_dir} ...")
    with open(output_file, "w") as f:
        for folder_name in tqdm(folder_names, desc="Packing PepFlow sequences"):
            folder_path = os.path.join(base_dir, folder_name)
            rec_fasta = os.path.join(folder_path, "receptor.fasta")
            pep_fasta = os.path.join(folder_path, "peptide.fasta")
            
            prot_seq = read_fasta(rec_fasta)
            pep_seq = read_fasta(pep_fasta)
            
            # Write the pair only when both sequences are available.
            if prot_seq and pep_seq:
                json_line = {"src": prot_seq, "trg": pep_seq}
                f.write(json.dumps(json_line) + "\n")
                valid_count += 1
                
    print(f"\nDone. Packed {valid_count} valid pairs.")
    print(f"Saved to: {output_file}")

if __name__ == '__main__':
    main()
