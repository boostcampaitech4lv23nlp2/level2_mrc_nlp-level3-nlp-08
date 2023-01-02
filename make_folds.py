import warnings
import argparse
import pandas as pd

from datasets import (
    load_from_disk,
    concatenate_datasets,
)

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(action='ignore')

def main(args):
    org_dataset = load_from_disk('../data/train_dataset/')
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )

    _id = []
    doc_id = []
    title = []
    context = []
    question = []
    answers = []
    context_len = []

    for train_data in full_ds:
        _id.append(train_data['id'])
        doc_id.append(train_data['document_id'])
        title.append(train_data['title'])
        context.append(train_data['context'])
        question.append(train_data['question'])
        answers.append(train_data['answers'])
        context_len.append(len(train_data['context']))
        
    train_dict = {
        "id":_id,
        "doc_id":doc_id,
        "title":title,
        "context":context,
        "question":question,
        "answers":answers,
        "context_len":context_len
    }

    train_df = pd.DataFrame(train_dict)

    kfold= StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    folds = kfold.split(train_df, train_df['context_len'].values)

    for fold, (train_idx, val_idx) in enumerate(folds):
        val_df= train_df.iloc[val_idx]
        val_df.to_csv(args.output_dir+'/fold'+str(fold+1)+'_test.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
    )

    args = parser.parse_args()
    main(args)