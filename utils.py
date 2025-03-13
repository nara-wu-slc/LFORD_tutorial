from pathlib import Path
from typing import Dict, List, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def load_dataset(
    src_file_path: Path, tgt_file_path: Path
) -> Tuple[List[str], List[str]]:
    src_data = []
    tgt_data = []
    with src_file_path.open(mode="r") as f:
        for line in f:
            src_data.append(line.strip())
    with tgt_file_path.open(mode="r") as f:
        for line in f:
            tgt_data.append(line.strip())

    return src_data, tgt_data


class PromptFormatter:
    def __init__(self):
        self.formatted_src_data: List[str] = []
        self.encodeds: str = ""
        self.lang_convert: Dict[str, str] = {
            "en": "English",
            "ja": "Japanese",
        }

    def formatting_prompt_func(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        src: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """
        zero-shotのpromptを作成する関数

        == Example ==
        system prompt:
            Translate English to Japanese.
        user ptompt:
            English: I am a student.
            Japanese:
        ====

        """
        if (
            model_name == "meta-llama/Meta-Llama-3-8B-instruct"
            or model_name == "meta-llama/Meta-Llama-3-70B-instruct"
        ):
            input = [
                {
                    "role": "system",
                    "content": f"Translate {self.lang_convert[src_lang]} to {self.lang_convert[tgt_lang]}.",
                },
                {
                    "role": "user",
                    "content": f"{self.lang_convert[src_lang]}: {src}\n{self.lang_convert[tgt_lang]}: ",
                },
            ]
            self.encodeds = tokenizer.apply_chat_template(input, tokenize=False)
            self.encodeds += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            self.encodeds = self.encodeds.replace("<|begin_of_text|>", "", 1)
        elif (
            model_name == "meta-llama/Meta-Llama-3-8B"
            or model_name == "meta-llama/Meta-Llama-3-70B"
        ):
            self.encodeds = f"Translate {self.lang_convert[src_lang]} to {self.lang_convert[tgt_lang]}. {self.lang_convert[src_lang]}: {src} {self.lang_convert[tgt_lang]}: "
        else:
            raise ValueError("model_name is invalid.")

        return self.encodeds

    def format_prompt(
        self,
        lp: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        src_data: List[str],
    ) -> List[str]:
        src_lang, tgt_lang = lp.split("-")

        for src in src_data:
            self.formatted_src_data.append(
                self.formatting_prompt_func(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    src=src,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
            )

        return self.formatted_src_data
