""" GLUE processors and helpers """
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

from transformers import is_tf_available
from transformers import PreTrainedTokenizer
from transformers import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import simple_accuracy

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if "mnli" in task_name:
        return {"acc": simple_accuracy(preds, labels)}
    elif 'snli' in task_name:
        return {"acc": simple_accuracy(preds, labels)}
    elif 'anli' in task_name:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b if example.text_b else ' ') for example in examples], max_length=max_length, pad_to_max_length=True,
    )  ## RoBERTa needs to ensure text_b is not empty string


    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"

class AnliAllProcessor(DataProcessor):
    """Processor for the ANLI data set (GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples("train_r1") + self._create_examples("train_r2") + self._create_examples("train_r3")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples("dev_r1") + self._create_examples("dev_r2") + self._create_examples("dev_r3")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples("test_r1") + self._create_examples("test_r2") + self._create_examples("test_r3")
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        from nlp import load_dataset
        dataset = load_dataset('anli')
        for (i, data) in enumerate(dataset[set_type]):
            guid = "%s-%s" % (set_type, data['uid'])
            text_a = data['premise']
            text_b = data['hypothesis']
            # label = None if set_type.startswith("test") else label_map[data['label']]
            label = label_map[data['label']]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AnliR1Processor(AnliAllProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples("dev_r1")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples("test_r1")


class AnliR2Processor(AnliAllProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples("dev_r2")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples("test_r2")


class AnliR3Processor(AnliAllProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples("dev_r3")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples("test_r3")


class AnliFullProcessor(AnliAllProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        # examples = SnliProcessor().get_train_examples(os.path.join(data_dir, 'SNLI'))
        # examples += FeverProcessor().get_train_examples(os.path.join(data_dir, 'nli_fever'))
        # examples += MnliProcessor().get_train_examples(os.path.join(data_dir, 'MNLI'))
        examples = AnliAllProcessor().get_train_examples(data_dir)
        return examples


class AnliPartProcessor(AnliFullProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        examples = SnliProcessor().get_train_examples(os.path.join(data_dir, 'SNLI'))
        examples += MnliProcessor().get_train_examples(os.path.join(data_dir, 'MNLI'))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

glue_tasks_num_labels = {
    "mnli": 3,
    "anli-r1": 3,
    "anli-r2": 3,
    "anli-r3": 3,
    "anli-all": 3,
    "anli-full": 3,
    "anli-part": 3,
    "snli": 3,
}

glue_processors = {
    "anli-r1": AnliR1Processor,
    "anli-r2": AnliR2Processor,
    "anli-r3": AnliR3Processor,
    "anli-all": AnliAllProcessor,
    "anli-full": AnliFullProcessor,
    "anli-part": AnliPartProcessor,
    "snli": SnliProcessor,
    "mnli": MnliProcessor,
}

glue_output_modes = {
    "anli-r1": "classification",
    "anli-r2": "classification",
    "anli-r3": "classification",
    "anli-all": "classification",
    "anli-full": "classification",
    "anli-part": "classification",
    "snli": "classification",
    "mnli": "classification",
}