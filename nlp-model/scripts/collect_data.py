import os
import pandas as pd
from datasets import load_dataset


# Create data directory
DATA_DIR="C:\\Code\\Fake News Detection System\\nlp-model\\data"
os.makedirs(DATA_DIR,exist_ok=True)


def download_liar_dataset():
    # Download and save dataset as csv file
    print("Downloading LIAR dataset...")
    dataset=load_dataset("liar",trust_remote_code=True)

    # convert into pandas dataframe
    df_train=pd.DataFrame(dataset["train"])
    df_valid=pd.DataFrame(dataset["validation"])
    df_test=pd.DataFrame(dataset["test"])


    # save as csv
    df_train.to_csv(os.path.join(DATA_DIR,"liar_train.csv"),index=False)
    df_valid.to_csv(os.path.join(DATA_DIR,"liar_valid.csv"),index=False)
    df_test.to_csv(os.path.join(DATA_DIR,"liar_test.csv"),index=False)


    print(f"Dataset downloaded and saved in {DATA_DIR}")
    print(df_train.head())  # Show sample data



if __name__=="__main__":
    download_liar_dataset()