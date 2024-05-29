# 음성 인식 실습 자료 - 학습
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import torch

def preprocess_function(examples):
    max_duration = 1.0
    audio_arrays = [x['array'] for x in examples['audio']]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True
    )
    return inputs

import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

if __name__ == "__main__":

    model_checkpoint = "facebook/wav2vec2-base"

    dataset = load_dataset("superb", "ks")
    metric = load_metric('accuracy')

    labels = dataset['train'].features['label'].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels) :
        label2id[label] = str(i)
        id2label[str(i)] = label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    encoded_dataset = dataset.map(preprocess_function, remove_columns=['audio', 'file'], batched=True)

    # Training the model
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels = num_labels,
        label2id = label2id,
        id2label = id2label
    ).to(device)

    training_args = TrainingArguments(
        output_dir="my_awesome_mind_model",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_on_each_node=1,
        learning_rate=3e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir="my_awesome_mind_model"
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.21.0`:
    # Please run `pip install transformers[torch]` or `pip install accelerate -U`
    # pip install transformers[torch] accelerate -U
