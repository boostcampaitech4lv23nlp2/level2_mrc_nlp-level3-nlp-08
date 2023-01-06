from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="monologg/koelectra-base-v3-discriminator",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    retrieval_ColBERT_path: str = field(
        # ColBERT rank pth파일이 설정된 경우 무시됨.
        # ColBERT rank pth파일이 없는 경우, 기학습된 모델 pth파일 경로를 아래에 설정
        default="colbert/best_model/nobm25_colbert_epoch10.pth",
        metadata={"help": "choice retrieval model ColBERT_path"},
    )
    ColBERT_rank_path: str = field(
        # _eval, _predict는 뺴고 작성
        # ColBERT rank pth파일이 없는 경우, None으로 설정
        default="colbert/inference_colbert_rank",
        metadata={"help": "ColBERT로 생성한 rank 파일이 있을 경우 load할 경로"},
    )
    save_ColBERT_rank_path: str = field(
        # _eval, _predict는 뺴고 작성
        # retrieval_ColBERT_path이 반드시 설정되어 있어야 함.
        default="colbert/inference_colbert_rank_new",
        metadata={"help": "retrieval_ColBERT_path로 기학습된 모델로 rank pth를 생성할 경로"},
    )
    is_roberta: bool = field(
        default=True,
        metadata={"help": "roberta를 쓰면 True로 사용"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=5,
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(default=False, metadata={"help": "Whether to build with faiss"})
    retrieval_choice: str = field(
        default="ColBERT",  # ColBERT, bm25, tfidf
        metadata={"help": "choice retrieval algorithms"},
    )
    bm25_sample_reader: bool = field(
        default=True,
        metadata={"help": "Whether to add bm25 hard negative passage"},
    )
