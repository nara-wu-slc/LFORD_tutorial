import os
from pathlib import Path
from typing import List

import torch
from tap import Tap
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import PromptFormatter, load_dataset


class Args(Tap):
    LP: str = "en-ja"

    src_file_path: Path = ""
    tgt_file_path: Path = ""

    output_file_dir: Path = "./results"

    model_name: str = "meta-llama/Meta-Llama-3-8B-instruct"


class Experiment:
    def __init__(self, args):
        self.args = args

        self.src_data: List[str] = []
        self.tgt_data: List[str] = []
        self.formatted_src_data: List[str] = []

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self.all_scores: List[float] = []

        self.submit_id: str = str(self.args.tgt_file_path).split("/")[-1][:-3]

    def execute(self):
        # --- Load dataset ---
        self.src_data, self.tgt_data = load_dataset(
            src_file_path=self.args.src_file_path, tgt_file_path=self.args.tgt_file_path
        )

        # --- Format prompt ---
        self.formatted_src_data = PromptFormatter().format_prompt(
            lp=self.args.LP,
            model_name=self.args.model_name,
            tokenizer=self.tokenizer,
            src_data=self.src_data,
        )

        # --- Evaluate QE score ---
        for src, tgt in zip(tqdm(self.formatted_src_data), self.tgt_data):
            input_ids = self.tokenizer.encode(src, return_tensors="pt")
            force_ids = self.tokenizer.encode(tgt, return_tensors="pt")

            force_ids_len = force_ids.size(-1)
            score = 0.0

            with torch.no_grad():
                for i in range(force_ids_len):
                    output = self.model(input_ids)

                    next_token_logits = output.logits[:, -1, :]

                    next_tokens_scores = torch.nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    )

                    next_tokens = force_ids[:, i]

                    force_logits = next_tokens_scores[
                        torch.arange(next_tokens_scores.size(0)), next_tokens
                    ]

                    force_logits = force_logits.item()
                    score += force_logits

                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            avg_score = score / force_ids_len
            self.all_scores.append(avg_score)

        # --- Write result ---
        output_dir: Path = Path(
            self.args.output_file_dir,
        )
        os.makedirs(output_dir, exist_ok=True)

        with Path(output_dir, f"{self.submit_id}.result").open(mode="w") as f:
            for score in self.all_scores:
                f.write(f"{score}\n")

        avg_score: float = sum(self.all_scores) / len(self.all_scores)
        with Path(output_dir, f"{self.submit_id}.avg_result").open(mode="w") as f:
            f.write(f"{avg_score}\n")

        print(f"{self.submit_id} score: {avg_score}")


def main(args):
    experiment = Experiment(args)
    experiment.execute()


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
