# 음성 Classification 평가 코드

import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer

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

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


if __name__ == "__main__" :

    dataset = load_dataset("superb", "ks")
    metric = load_metric('accuracy')

    labels = dataset['train'].features['label'].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir="my_awesome_mind_model",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_steps=500,
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

    # last check point get
    last_checkpoint = get_last_checkpoint('./my_awesome_mind_model/')

    # Load the best model
    best_model = AutoModelForAudioClassification.from_pretrained(last_checkpoint).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(last_checkpoint)
    encoded_dataset = dataset.map(preprocess_function, remove_columns=['audio', 'file'], batched=True)

    # Create a new trainer for evaluation
    eval_trainer = Trainer(
        model=best_model,
        args=training_args,
        eval_dataset=encoded_dataset['test'],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    # Evaluate the model on the test dataset
    test_results = eval_trainer.evaluate()
    print(f"Test evaluation results: {test_results}")

    # Make predictions on the test dataset
    test_predictions = eval_trainer.predict(encoded_dataset['test'])
    predicted_labels = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids

    # Print predictions and true labels
    for i in range(len(predicted_labels)):
        print(f"Prediction: {id2label[str(predicted_labels[i])]}, True Label: {id2label[str(true_labels[i])]}")

