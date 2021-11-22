"""
BERT pre-training set up
"""
import time
import numpy as np
from transformers import BatchEncoding
from datasets import load_dataset, load_from_disk#, save_to_disk
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaModel,
    PreTrainedTokenizerBase,
    FlaxAutoModelForMaskedLM
)
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForMaskedLM,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
import json
#from preprocess import get_data
from dataclasses import dataclass, field
import flax
import jax
from flax import jax_utils, traverse_util
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
import optax
from flax.training import train_state

@flax.struct.dataclass
class FlaxDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples: List[Dict[str, np.ndarray]], pad_to_multiple_of: int) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=pad_to_multiple_of, return_tensors=TensorType.NUMPY)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# Constants
mlm_probability = 0.15
seed = 42
dtype = "float32"
num_epochs = 10
train_batch_size = 1000
eval_batch_size = 1000
warmup_steps = 1000
learning_rate = "5e-3"
dataset_amount = 10000
num_train_steps = dataset_amount // train_batch_size * num_epochs
adam_beta1 = "0.9"
adam_beta2 = "0.98"
adam_epsilon = "0.1"
weight_decay = "0.01"

# Initialize tokenizer and data collator
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

# Initialize training
rng = jax.random.PRNGKey(seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

# Initialize model
config = RobertaConfig.from_pretrained("roberta-base", vocab_size=50265)
tiny = {"num_hidden_layers" : 2, "hidden_size" : 252,}
config.update(tiny)
model = FlaxAutoModelForMaskedLM.from_config(config, seed=seed, dtype=getattr(jnp, dtype))

# Create learning rate schedule
warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps)
decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps - warmup_steps)
linear_decay_lr_schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])




def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=adam_beta1,
            b2=adam_beta2,
            eps=adam_epsilon,
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        )

# Setup train state
state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

# Define gradient update step fn
def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")

        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # take average
        loss = loss.sum() / label_mask.sum()

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

    # Define eval fn

def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

data = None
with open("Wikipedia/10kdata.pickle", "rb") as handle:
    data = pickle.load(handle)

train_time = 0
epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
for epoch in epochs:
    # ======================== Training ================================
    train_start = time.time()
    train_metrics = []

    def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int, data) -> jnp.ndarray:
        num_samples = len(samples_idx)
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        batch_idx = np.split(samples_idx, sections_split)
        return batch_idx

    def get_batch(batch_size: int, data):
        """
        Generator object that yields a batch every time
        next() is called on it.
        """
        current_sample, batch, special, special_tokens_mask = [], [], [], []
        for document in data:
            for i, sentence in enumerate(document["token_ids"]):
                if len(current_sample) + len(sentence) <= 512:
                    current_sample.extend(sentence)
                    special_tokens_mask.extend(document["special_tokens"][i])
                else:
                    batch.append(current_sample)
                    special.append(special_tokens_mask)
                    current_sample, special_tokens_mask = [], []
                if len(batch) == batch_size:
                    yield batch, special



    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    # Generate an epoch by shuffling sampling indices from the train dataset
    num_train_samples = len(data)
    train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))
    train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size, data)

    for step, (batch, special) in enumerate(tqdm(get_batch(batch_size=train_batch_size, data=data))):
        encoding = BatchEncoding({"input_ids": batch, "special": special})
        model_inputs = data_collator(encoding, pad_to_multiple_of=16)

    # Gather the indexes for creating the batch and do a training step
    for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):

        samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
        model_inputs = data_collator(samples, pad_to_multiple_of=16)

        # Model forward
        model_inputs = shard(model_inputs.data)
        state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
        train_metrics.append(train_metric)

        cur_step = epoch * (num_train_samples // train_batch_size) + step

        if cur_step % training_args.logging_steps == 0 and cur_step > 0:
            # Save metrics
            train_metric = jax_utils.unreplicate(train_metric)
            train_time += time.time() - train_start
            if has_tensorboard and jax.process_index() == 0:
                write_train_metric(summary_writer, train_metrics, train_time, cur_step)

            epochs.write(
                f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
            )

            train_metrics = []

        if cur_step % training_args.eval_steps == 0 and cur_step > 0:
            # ======================== Evaluating ==============================
            num_eval_samples = len(tokenized_datasets["validation"])
            eval_samples_idx = jnp.arange(num_eval_samples)
            eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

            eval_metrics = []
            for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
                samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
                model_inputs = data_collator(samples, pad_to_multiple_of=16)

                # Model forward
                model_inputs = shard(model_inputs.data)
                metrics = p_eval_step(state.params, model_inputs)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
            eval_normalizer = eval_metrics.pop("normalizer")
            eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

            # Update progress bar
            epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"

            # Save metrics
            if has_tensorboard and jax.process_index() == 0:
                write_eval_metric(summary_writer, eval_metrics, cur_step)

        if cur_step % training_args.save_steps == 0 and cur_step > 0:
            # save checkpoint after each epoch and push checkpoint to the hub
            if jax.process_index() == 0:
                params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                model.save_pretrained(training_args.output_dir, params=params)
                tokenizer.save_pretrained(training_args.output_dir)
                if training_args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)