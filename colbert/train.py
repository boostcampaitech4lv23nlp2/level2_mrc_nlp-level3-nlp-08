# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from datasets import load_from_disk
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizer import *
from model import *
import json
import pickle

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

"""
Traceback (most recent call last):
  File "train.py", line 163, in <module>
    main()
  File "train.py", line 68, in main
    trained_model = train(args, train_dataset, model)
  File "train.py", line 135, in train
    outputs = model(p_inputs, q_inputs)
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/opt/ml/input/level2_mrc_nlp-level3-nlp-08/colbert/model.py", line 36, in forward
    return self.get_score(Q, D)
  File "/opt/ml/input/level2_mrc_nlp-level3-nlp-08/colbert/model.py", line 75, in get_score
    q_sequence_output = Q.view(
RuntimeError: shape '[16, 1, -1, 128]' is invalid for input of size 118272
"""


def main():

    set_seed(42)
    batch_size = 16
    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=12,
        weight_decay=0.01,
    )

    MODEL_NAME = "klue/bert-base"
    tokenizer = load_tokenizer(MODEL_NAME)

    datasets = load_from_disk("../data/wiki_korQuAD_aug_dataset")
    train_dataset = pd.DataFrame(datasets["train"])
    train_dataset = train_dataset.reset_index(drop=True)
    train_dataset = set_columns(train_dataset)

    print("dataset tokenizing.......")
    # 토크나이저
    train_context, train_query = tokenize_colbert(train_dataset, tokenizer, corpus="both")

    train_dataset = TensorDataset(
        train_context["input_ids"],
        train_context["attention_mask"],
        train_context["token_type_ids"],
        train_query["input_ids"],
        train_query["attention_mask"],
        train_query["token_type_ids"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    model.to(device)

    print("model train...")
    trained_model = train(args, train_dataset, model)
    torch.save(trained_model.state_dict(), "best_model/colbert.pth")


def train(args, dataset, model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )

    # Optimizer 세팅
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training 시작
    global_step = 0

    model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        print(f"epoch {epoch}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss = 0

        for step, batch in enumerate(epoch_iterator):
            model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            # outputs with similiarity score
            outputs = model(p_inputs, q_inputs)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, outputs.shape[0]).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()
        final_loss = total_loss / len(dataset)
        print("total_loss :", final_loss)
        torch.save(model.state_dict(), f"best_model/colbert_epoch{epoch+1}.pth")

    return model


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
