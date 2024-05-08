import pandas as pd
import spacy
from typing import List
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = strip_quotes(df)
    mt = df["Maltese"].to_list()
    en = df["English"].to_list()
    mt_tok = tokenize(mt)
    en_tok = tokenize(en)
    df["mt_tokens"] = mt_tok
    df["en_tokens"] = en_tok
    return df


def tokenize(untok_data: List[str]) -> List[List[str]]:
    tok_data = []
    for sent in tqdm(untok_data):
        doc = nlp(sent)
        tok_sent = [x.text for x in doc]
        tok_data.append(tok_sent)
    return tok_data


def strip_quotes(df: pd.DataFrame) -> pd.DataFrame:
    df["Maltese"] = df["Maltese"].apply(lambda x: x.strip('"'))
    df["English"] = df["English"].apply(lambda x: x.strip('"'))
    return df


def main():
    print("Preprocessing train file...")
    train = pd.read_csv("data/new_split/train.csv")
    p_train = preprocess(train)
    p_train.to_csv("data/new_split/p_train.csv", index=False)

    print("Preprocessing test file...")
    test = pd.read_csv("data/new_split/test.csv")
    p_test = preprocess(test)
    p_test.to_csv("data/new_split/p_test", index=False)

    print("Preprocessing dev file...")
    dev = pd.read_csv("data/new_split/dev.csv")
    p_dev = preprocess(dev)
    p_dev.to_csv("data/new_split/p_dev.csv", index=False)


if __name__ == "__main__":
    main()
