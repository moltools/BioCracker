import argparse
import glob

import joblib
from tqdm import tqdm

from biocracker.antismash import parse_region_gbk_file
from biocracker.paras import find_a_domains, featurize_signature


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbks", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def main():
    args = cli()

    model = joblib.load(args.model)

    gbk_iter = glob.iglob(f"{args.gbks}/*.gbk")
    for gbk_file in tqdm(gbk_iter):
        for region in parse_region_gbk_file(gbk_file, top_level="region"):
            for gene in region.genes:
                name, protein_seq = gene.name, gene.protein_seq
                a_domains = find_a_domains(seq_id=name, protein_seq=protein_seq, evalue_cutoff=1e-5)
                for a_domain in a_domains:
                    protein_name = a_domain.protein
                    signature = a_domain.extended_signature
                    if signature is None:
                        print(protein_name, "N/A", "0.0000", "N/A", sep="\t")
                    else:
                        if model is not None:
                            features = featurize_signature(signature)
                            features_reshaped = features.reshape(1, -1)  # reshape for single sample
                            prediction = model.predict_proba(features_reshaped)
                            pred_names = model.classes_
                            prediction = {name: prob for name, prob in zip(pred_names, prediction[0])}
                            top_pred = max(prediction.items(), key=lambda x: x[1])
                            print(protein_name, top_pred[0], f"{top_pred[1]:.4f}", signature, sep="\t")
                        else:
                            print(protein_name, "N/A", "0.0000", signature, sep="\t")


if __name__ == "__main__":
    main()
