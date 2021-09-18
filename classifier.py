from collections import defaultdict as ddict
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    accuracy_score,
)

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from torch.optim import AdamW


# fix the random seed
def seed_everything(seed=11747):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()

    def forward(self, input_ids, token_type_ids, attention_mask):
        # todo
        raise NotImplementedError()


# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        token_type_ids = torch.LongTensor(encoding["token_type_ids"])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx : start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(
                data
            )
            batches.append(
                {
                    "token_ids": token_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "sents": sents,
                }
            )

        return batches


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, flag="train"):
    # how to specify the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer_our	= BertTokenizerOur.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    with open(filename, "r") as fp:
        for line in fp:
            label, org_sent = line.split(" ||| ")
            sent = org_sent.lower().strip()
            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            label = int(label.strip())
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, label, tokens))

    if flag == "train":
        return data, len(num_labels)
    else:
        return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/sst-train.txt")
    parser.add_argument("--dev", type=str, default="data/sst-dev.txt")
    parser.add_argument("--test", type=str, default="data/sst-test.txt")
    parser.add_argument("--seed", type=int, default=11747)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)
    parser.add_argument("--option", type=str, default="pretrain")
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--dev_out", type=str, default="sst-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="sst-test-output.txt")

    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args


# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, args, save_file=None):
    model.eval()
    y_true = []
    y_pred = []
    sents = []
    use_cuda = int(args.cuda) >= 0

    for step, batch in enumerate(dataloader):
        b_ids, b_type_ids, b_mask, b_labels, b_sents = (
            batch[0]["token_ids"],
            batch[0]["token_type_ids"],
            batch[0]["attention_mask"],
            batch[0]["labels"],
            batch[0]["sents"],
        )

        if use_cuda:
            b_ids = b_ids.cuda()
            b_type_ids = b_type_ids.cuda()
            b_mask = b_mask.cuda()

        with torch.no_grad():
            logits = model(b_ids, b_type_ids, b_mask)
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            b_labels = b_labels.flatten()
            y_true.extend(b_labels)
            y_pred.extend(preds)
            sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    if save_file is not None:
        out_fp = open(save_file, "w")
        for sent, pred in zip(sents, preds):
            out_fp.write(f"{pred} ||| {sent}\n")
        out_fp.close()

    return acc, f1


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)  # fix the seed for reproducibility

    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train, "train")
    dev_data = create_data(args.dev, "valid")
    test_data = create_data(args.test, "test")

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)
    test_dataset = BertDataset(test_data, args)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn,
    )

    # you can customize the config file that you want to provide to the Sentence classifier model
    config = {
        "hidden_dropout_prob": 0.3,
        "num_labels": num_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "option": args.option,
    }
    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model
    model = BertSentClassifier(config)

    print("Loading Done")

    use_cuda = False

    if int(args.cuda) >= 0:
        use_cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        model.cuda()

    ## Specify the option for pretraining or finetuning
    lr = 1e-3
    if args.option == "pretrain":
        lr = args.lr_pretrain
    elif args.option == "finetune":
        lr = args.lr_finetune

    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    best_model = None
    best_dev_acc = 0
    filepath = f"{args.option}-{args.epochs}-{lr}.pt"
    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        # print(epoch)

        train_loss = 0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            b_ids, b_type_ids, b_mask, b_labels, b_sents = (
                batch[0]["token_ids"],
                batch[0]["token_type_ids"],
                batch[0]["attention_mask"],
                batch[0]["labels"],
                batch[0]["sents"],
            )

            if use_cuda:
                b_ids = b_ids.cuda()
                b_type_ids = b_type_ids.cuda()
                b_mask = b_mask.cuda()
                b_labels = b_labels.cuda()

            optimizer.zero_grad()
            logits = model(b_ids, b_type_ids, b_mask)
            loss = (
                F.nll_loss(logits, b_labels.view(-1), reduction="sum") / args.batch_size
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        model.eval()

        train_acc, train_f1 = model_eval(train_dataloader, model, args)
        dev_acc, dev_f1 = model_eval(dev_dataloader, model, args)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model = model
            torch.save(best_model, filepath)

        print(
            f"Epoch {epoch} \t Train loss :: {round(train_loss, 3)} \t Train Acc :: {round(train_acc,3)} \t Dev Acc :: {round(dev_acc, 3)}"
        )

    best_model = torch.load(filepath)
    dev_acc, dev_f1 = model_eval(
        dev_dataloader, best_model, args, save_file=args.dev_out
    )
    test_acc, test_f1 = model_eval(
        test_dataloader, best_model, args, save_file=args.test_out
    )
