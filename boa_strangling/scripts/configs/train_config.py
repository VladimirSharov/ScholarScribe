from dataclasses import dataclass, field
from datetime import datetime

def default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"boa_strangling/outputs/run-{timestamp}"

@dataclass
class TrainConfig:
    tag_vocab_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40/tag_vocab.json"
    dataset_path: str = "boa_strangling/data/data_split_fi_eng_min5_abs40"
    model_name: str = "xlm-roberta-base"
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    project: str = "thesis-tagger"
    output_dir: str = field(default_factory=default_output_dir)
    ...
