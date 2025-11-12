from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def main():
    df_raw = pd.read_csv('data/raw/b3db.tsv', sep='\t')
    df_features = compute_features(df_raw)
    df_features.to_csv('data/preprocess/b3db.csv', index=False)

def compute_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = []
    Y = []
    
    for _, row in df_raw.iterrows():
        fingerprint = smiles_to_fingerprints(row.SMILES)
        X.append(fingerprint)
        Y.append(1 if row['BBB+/BBB-'] == 'BBB+' else 0
        )
    
    X = np.array(X)
    Y = np.array(Y)

    df_features = pd.DataFrame(X)
    df_features['label'] = Y

    return df_features

def smiles_to_fingerprints(smiles: str) -> np.array:
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, useChirality=True, radius=2, nBits=1024, bitInfo={}
    )
    return np.array(fingerprint)

if __name__ == '__main__':
    main()