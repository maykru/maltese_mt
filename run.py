import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch import optim
import torch.nn as nn
from rnn_model import EncoderRNN, DecoderRNN
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

EMBED_DIM = 200
BATCH_SIZE = 32
HIDDEN_DIM = 128
MAX_SEQ_LEN = 25
EPOCHS = 10
LR = 0.001


def make_vocab(tokenized_data: list[list[str]]) -> tuple[dict[str: int], dict[int: str]]:
    tok_data = [x for sent in tokenized_data for x in sent]
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for i, tok in enumerate(set(tok_data)):
        vocab[tok] = i+4
    rev_voc = {v: k for k, v in vocab.items()}
    return vocab, rev_voc


def pad_truncate(tok_data: list[list[str]]) -> list[list[str]]:
    new_tok_data = []
    max_seq_len = MAX_SEQ_LEN
    pad_tok = "<PAD>"
    for sent in tok_data:
        if len(sent) > max_seq_len:
            sent = sent[:max_seq_len]
        elif len(sent) < max_seq_len:
            padding = [pad_tok] * (max_seq_len - len(sent))
            sent.extend(padding)
        new_tok_data.append(sent)
    return new_tok_data


def tok2id(tokens: list[list[str]], vocab: dict[str, int]) -> list[torch.Tensor]:
    indices = []
    for sent in tokens:
        sent_ids = []
        for tok in sent:
            if tok in vocab:
                sent_ids.append(vocab[tok])
            else:
                sent_ids.append(vocab["<UNK>"])
        indices.append(torch.tensor(sent_ids, dtype=torch.int64))
    return indices


def prepare_data(df: pd.DataFrame):
    df["mt_tokens"] = df["mt_tokens"].apply(lambda x: x.strip('][').replace("'", "").split(', '))
    df["en_tokens"] = df["en_tokens"].apply(lambda x: x.strip('][').replace("'", "").split(', '))
    mt = df["mt_tokens"].to_list()
    en = df["en_tokens"].to_list()
    mt_voc, mt_rev_voc = make_vocab(mt)
    en_voc, en_rev_voc = make_vocab(en)
    mt = pad_truncate(mt)
    en = pad_truncate(en)
    mt_token_ids = tok2id(mt, mt_voc)
    en_token_ids = tok2id(en, en_voc)
    return mt_token_ids, en_token_ids, mt_voc, en_voc, mt_rev_voc, en_rev_voc


class DataMapper(Dataset):
    def __init__(self, src, tgt):
        self.x = src
        self.y = tgt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(train_loader, encoder, decoder, rev_voc, epochs=EPOCHS, lr=LR):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_bleu = 0.0
        for src, tgt in tqdm(train_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_output, encoder_hidden = encoder(src)
            decoder_output, _, _ = decoder(encoder_output, encoder_hidden, tgt)
            loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), tgt.view(-1))
            loss.backward()

            _, topi = decoder_output.topk(1)
            decoded_ids = topi.squeeze()
            batch_bleu = 0.0
            # looping over each sentence in the batch
            for i, sent_ids in enumerate(decoded_ids):
                decoded_words = []
                for word_id in sent_ids:
                    if (key := int(word_id.item())) < len(rev_voc):
                        decoded_words.append(rev_voc[key])
                    else:
                        decoded_words.append("<UNK>")
                src_sent = [rev_voc[int(x.item())] for x in tgt[i]]
                batch_bleu += sentence_bleu([src_sent], decoded_words)
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_bleu += (batch_bleu / 32)
            total_loss += loss.item()
        print(f"\nEpoch {epoch + 1} Loss: {total_loss/len(train_loader)} BLEU: {epoch_bleu/len(train_loader)}")


def evaluate(test_loader, encoder, decoder, rev_voc):
    total_bleu = 0.0
    for src, tgt in test_loader:
        encoder_output, encoder_hidden = encoder(src)
        decoder_output, _, _ = decoder(encoder_output, encoder_hidden, tgt)
        _, topi = decoder_output.topk(1)
        decoded_ids = topi.squeeze()
        batch_bleu = 0.0
        for i, sent_ids in enumerate(decoded_ids):
            # calculate BLEU score with predicted sequence
            decoded_words = []
            for word_id in sent_ids:
                if (key := int(word_id.item())) < len(rev_voc):
                    decoded_words.append(rev_voc[key])
                elif (key := int(word_id.item())) == 4:
                    decoded_words.append(rev_voc[key])
                    break
                else:
                    decoded_words.append("<UNK>")
            tgt_sent = [rev_voc[int(x.item())] for x in tgt[i]]
            batch_bleu += sentence_bleu([tgt_sent], decoded_words)
        total_bleu += (batch_bleu / 32)
    print(f"Evaluation BLEU: {total_bleu / len(test_loader)}")


def main():
    train_df = pd.read_csv("p_train.csv")
    # train_df = pd.read_csv("p_eu_train.csv")
    train_mt_ids, train_en_ids, train_mt_tok2id, train_en_tok2id, train_mt_id2tok, train_en_id2tok = prepare_data(train_df)
    train_set = DataMapper(src=train_mt_ids, tgt=train_en_ids)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

    eval_df = pd.read_csv("p_dev.csv")
    # eval_df = pd.read_csv("p_test.csv")
    # eval_df = pd.read_csv("p_eu_test.csv")
    test_mt_ids, test_en_ids, test_mt_tok2id, test_en_tok2id, test_mt_id2tok, test_en_id2tok = prepare_data(eval_df)
    test_set = DataMapper(src=test_mt_ids, tgt=test_en_ids)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    encoder = EncoderRNN(
        vocab_size=len(train_mt_tok2id),
        hidden_size=HIDDEN_DIM
    )

    decoder = DecoderRNN(
        hidden_size=HIDDEN_DIM,
        vocab_size=len(train_mt_tok2id)
    )

    train(train_loader, encoder, decoder, train_en_id2tok)
    evaluate(test_loader, encoder, decoder, test_en_id2tok)


if __name__ == "__main__":
    main()
