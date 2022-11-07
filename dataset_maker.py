import logging
import pdb
import torch
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from args import DataTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from datasets import load_dataset, DownloadConfig
import random

logger = logging.getLogger(__name__)


class DatasetMaker:
    def __init__(self, dataset_saved_path: str, data_args: DataTrainingArguments,
                 training_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizerBase):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.dataset_saved_path = dataset_saved_path

    def make_dataset(self):
        logger.info('******* Making Dataset **********')
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file
            extension = self.data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))
        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0:
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for model. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples: Dict):
            """
            如果是json，examples就是json对应的dict。如果是纯文本，examples["text"]就是全部文本,每个item就是文本文件中的一行
            """
            if isinstance(examples["src"][0], str):
                inputs = [ex.replace(' ', '') if self.data_args.chinese_data else ex for ex in examples["src"]]
            elif isinstance(examples["src"][0], list):
                inputs = [' '.join(ex).replace(' ', '') if self.data_args.chinese_data else ' '.join(ex) for ex in
                          examples["src"]]
            else:
                raise ValueError(f'only support str/list in content, now {type(examples["src"][0])}')

            if isinstance(examples["tgt"][0], str):
                targets = [ex.replace(' ',
                                      '') + self.tokenizer.eos_token if self.data_args.chinese_data else ex + self.tokenizer.eos_token
                           for ex in examples["tgt"]]
            elif isinstance(examples["tgt"][0], list):
                targets = [' '.join(ex).replace(' ',
                                                '') + self.tokenizer.eos_token if self.data_args.chinese_data else ' '.join
                                                                                                                   (ex) + self.tokenizer.eos_token
                           for ex in examples["tgt"]]
            else:
                raise ValueError(f'only support str/list in summary, now {type(examples["tgt"][0])}')

            questions = [ex for ex in examples["questions"]]
            answer_index = [ex for ex in examples["answer_index"]]
            entity_spans = [ex for ex in examples["entity_spans"]]
            sent_spans = [ex for ex in examples["sent_spans"]]
            sent_entity_edge = [ex for ex in examples["sent_entity_edges"]]
            answers = [ex for ex in examples["answers"]]

            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                          truncation=True,
                                          add_special_tokens=False, return_offsets_mapping=True)
            offset_mappings = model_inputs['offset_mapping']

            map_entity_spans = []
            for case_index, case in enumerate(entity_spans):
                map_entity_span = []
                for entity_span in case:
                    offset_mapping = offset_mappings[case_index]
                    start, end = entity_span
                    map_starts = [person_id for (age, person_id) in offset_mapping if age <= start]
                    map_ends = [person_id for (age, person_id) in offset_mapping if age <= end]
                    map_start = map_starts[-1]
                    map_end = map_ends[-1]
                    off_start = \
                    [index for index, (age, person_id) in enumerate(offset_mapping) if person_id == map_start][0]
                    off_end = [index for index, (age, person_id) in enumerate(offset_mapping) if person_id == map_end][
                                  0] + 1
                    if off_end == len(offset_mapping):
                        off_end = off_end - 1
                    map_entity_span.append([off_start, off_end])
                map_entity_spans.append(map_entity_span)

            map_sent_spans = []
            for case_index, case in enumerate(sent_spans):
                map_sent_span = []
                for sent_span in case:
                    offset_mapping = offset_mappings[case_index]
                    start, end = sent_span
                    map_starts = [person_id for (age, person_id) in offset_mapping if age > start]
                    map_ends = [person_id for (age, person_id) in offset_mapping if age <= end]
                    map_start = map_starts[0]
                    map_end = map_ends[-1]
                    off_start = \
                        [index for index, (age, person_id) in enumerate(offset_mapping) if person_id == map_start][0]
                    if off_start == 1:
                        off_start = 0
                    off_end = [index for index, (age, person_id) in enumerate(offset_mapping) if person_id == map_end][
                                  0] + 1
                    if off_end == len(offset_mapping):
                        off_end = off_end - 1
                    map_sent_span.append([off_start, off_end])
                    # self.tokenizer.decode(model_inputs['input_ids'][0][63:84])
                map_sent_spans.append(map_sent_span)

            questions_inputs = []
            questions_inputs_mask = []
            for case in questions:
                case_input = []
                case_input_mask = []
                for question in case:
                    questions_input = self.tokenizer(question, max_length=15, padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=False)
                    case_input.append(questions_input['input_ids'])
                    case_input_mask.append(questions_input['attention_mask'])
                questions_inputs.append(case_input)
                questions_inputs_mask.append(case_input_mask)

            new_answer_index = []
            for case_index,answer in enumerate(answer_index):
                # answer = [each[0] for each in answer]
                if all(i >= len(entity_spans[case_index]) for i in answer) == True and len(answer)!=0:
                new_answer_index.append(answer)


            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,
                                        add_special_tokens=False)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            model_inputs["questions_inputs"] = questions_inputs
            model_inputs["questions_inputs_mask"] = questions_inputs_mask
            model_inputs["map_entity_spans"] = map_entity_spans
            model_inputs["map_sent_spans"] = map_sent_spans
            model_inputs["sent_entity_edge"] = sent_entity_edge
            model_inputs["answer_index"] = new_answer_index
            model_inputs['answers'] = answers
            return model_inputs

        datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=False,
        )

        logger.info('saving dataset')
        dataset_saved_path = self.dataset_saved_path
        datasets.save_to_disk(dataset_saved_path)
        logger.info(f'******* Dataset Finish {dataset_saved_path} **********')
        return datasets
