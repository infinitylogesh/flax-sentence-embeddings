from dataclasses import dataclass, field, asdict, replace
from functools import partial
from typing import Callable, List, Union

import jax
from jax.config import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
from flax.training.common_utils import shard
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, FlaxBertModel
import wandb
import json
import os

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, FlaxAutoModel


@dataclass
class TrainingArgs:
    model_id: str = "DeepPavlov/bert-base-cased-conversational"
    max_epochs: int = 5
    batch_size_per_device: int = 256
    seed: int = 42
    lr: float = 2e-5
    init_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 1e-3

    current_context_maxlen: int = 128
    response_maxlen:int = 128

    logging_steps: int = 20
    save_dir: str = "checkpoints"

    tr_data_files: List[str] = field(
        default_factory=lambda: [
            "data/dummy.jsonl",
        ]
    )

    val_data_files: List[str] = field(
        default_factory=lambda: [
            "data/dummy.jsonl",
        ]
    )

    def __post_init__(self):
        self.batch_size = self.batch_size_per_device * jax.device_count()


def scheduler_fn(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(init_value=init_lr, end_value=lr, transition_steps=warmup_steps)
    decay_fn = optax.linear_schedule(init_value=lr, end_value=1e-7, transition_steps=decay_steps)
    lr = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return lr


def build_tx(lr, init_lr, warmup_steps, num_train_steps, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale")) for k, v in params.items()}
        return traverse_util.unflatten_dict(mask)
    lr = scheduler_fn(lr, init_lr, warmup_steps, num_train_steps)
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask)
    return tx, lr


class TrainState(train_state.TrainState):
    loss_fn: Callable = struct.field(pytree_node=False)
    scheduler_fn: Callable = struct.field(pytree_node=False)


def multiple_negative_ranking_loss(embedding1, embedding2):
    def _cross_entropy(logits):
        bsz = logits.shape[-1]
        labels = (jnp.arange(bsz)[..., None] == jnp.arange(bsz)[None]).astype("f4")
        logits = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(labels * logits, axis=-1)
        return loss

    batch_similarity = jnp.dot(embedding1, jnp.transpose(embedding2))
    return _cross_entropy(batch_similarity)


@partial(jax.pmap, axis_name="batch")
def train_step(state, current_context_input, response_input, drp_rng):
    train = True
    new_drp_rng, drp_rng = jax.random.split(drp_rng, 2)

    def loss_fn(params, current_context_input, response_input, drp_rng):
        def _forward(model_input):
            attention_mask = model_input["attention_mask"][..., None]
            embedding = state.apply_fn(**model_input, params=params, train=train, dropout_rng=drp_rng)[0]
            attention_mask = jnp.broadcast_to(attention_mask, jnp.shape(embedding))

            embedding = embedding * attention_mask
            embedding = jnp.mean(embedding, axis=1)

            modulus = jnp.sum(jnp.square(embedding), axis=-1, keepdims=True)
            embedding = embedding / jnp.maximum(modulus, 1e-12)

            # gather all the embeddings on same device for calculation loss over global batch
            embedding = jax.lax.all_gather(embedding, axis_name="batch")
            embedding = jnp.reshape(embedding, (-1, embedding.shape[-1]))

            return embedding

        current_context_emb, response_emb = _forward(current_context_input), _forward(response_input)
        loss = state.loss_fn(current_context_emb, response_emb)
        # loss considering 
        # 1) the interaction between the immediate context and its accompanying response, 
        # 2) the interaction of the response with up to N past contexts from the conversation history,
        # as well as 3) the interaction of the full context with the response
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, current_context_input, response_input, drp_rng)
    state = state.apply_gradients(grads=grads)

    metrics = {"tr_loss": loss, "lr": state.scheduler_fn(state.step)}

    return state, metrics, new_drp_rng
@partial(jax.pmap, axis_name="batch")
def val_step(state, current_context_input, response_input):
    train = False

    def _forward(model_input):
        attention_mask = model_input["attention_mask"][..., None]
        embedding = state.apply_fn(**model_input, params=state.params, train=train)[0]
        attention_mask = jnp.broadcast_to(attention_mask, jnp.shape(embedding))

        embedding = embedding * attention_mask
        embedding = jnp.mean(embedding, axis=1)

        modulus = jnp.sum(jnp.square(embedding), axis=-1, keepdims=True)
        embedding = embedding / jnp.maximum(modulus, 1e-12)

        # gather all the embeddings on same device for calculation loss over global batch
        embedding = jax.lax.all_gather(embedding, axis_name="batch")
        embedding = jnp.reshape(embedding, (-1, embedding.shape[-1]))

        return embedding

    current_context_emb, response_emb = _forward(current_context_input), _forward(response_input)
    loss = state.loss_fn(current_context_emb, response_emb)
    # loss considering 
    # 1) the interaction between the immediate context and its accompanying response, 
    # 2) the interaction of the response with up to N past contexts from the conversation history,
    # as well as 3) the interaction of the full context with the response
   
    return jnp.mean(loss)



def get_batched_dataset(dataset, batch_size, seed=None):
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    for i in range(len(dataset) // batch_size):
        batch = dataset[i*batch_size: (i+1)*batch_size]
        yield dict(batch)


@dataclass
class DataCollator:
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
    current_context_maxlen: int = 128
    response_maxlen: int = 128

    def __call__(self, batch):
        examples = batch["text"]
        contexts = []
        responses = []
        for text in examples:
            context,response = eval(text)
            contexts.append(context)
            responses.append(response)

        current_context_input = self.tokenizer(contexts, return_tensors="jax", max_length=self.current_context_maxlen, truncation=True, padding="max_length")
        response_input = self.tokenizer(responses, return_tensors="jax", max_length=self.response_maxlen, truncation=True, padding="max_length")
        current_context_input, response_input = dict(current_context_input), dict(response_input)
        return shard(current_context_input), shard(response_input)
    
    


def save_checkpoint(save_dir, state, save_fn=None, training_args=None):
    print(f"saving checkpoint in {save_dir}", end=" ... ")

    os.makedirs(save_dir, exist_ok=True)
    state = jax_utils.unreplicate(state)

    if save_fn is not None:
        # saving model in HF fashion
        save_fn(save_dir, params=state.params)
    else:
        path = os.path.join(save_dir, "flax_model.msgpack")
        with open(path, "wb") as f:
            f.write(to_bytes(state.params))

    # this will save optimizer states
    path = os.path.join(save_dir, "opt_state.msgpack")
    with open(path, "wb") as f:
        f.write(to_bytes(state.opt_state))

    if training_args is not None:
        path = os.path.join(save_dir, "training_args.json")
        with open(path, "w") as f:
            json.dump(asdict(training_args), f)

    print("done!!")

def prepare_dataset(args):
    tr_dataset = load_dataset("text", data_files=args.tr_data_files, split="train")
    val_dataset = load_dataset("text", data_files=args.val_data_files, split="train")

    # ensures similar processing to all splits at once
    dataset = DatasetDict(train=tr_dataset, validation=val_dataset)

    # drop extra batch from the end
    for split in dataset:
        num_samples = len(dataset[split]) - len(dataset[split]) % args.batch_size
        dataset[split] = dataset[split].shuffle(seed=args.seed).select(range(num_samples))
    
    tr_dataset, val_dataset = dataset["train"], dataset["validation"]
    return tr_dataset, val_dataset






def main(args,logger):
    os.makedirs(args.save_dir, exist_ok=True)
    # code is generic to any other model as well
    model = FlaxBertModel.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    data_collator = DataCollator(
        tokenizer=tokenizer,
        current_context_maxlen=args.current_context_maxlen,
        response_maxlen=args.response_maxlen
    )

    tr_dataset, val_dataset = prepare_dataset(args)

    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": (len(tr_dataset) // args.batch_size) * args.max_epochs,
        "weight_decay": args.weight_decay,
    }
    tx, lr = build_tx(**tx_args)

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        loss_fn=multiple_negative_ranking_loss,
        scheduler_fn=lr,
    )
    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(args.seed)
    drp_rng = jax.random.split(rng, jax.device_count())
    for epoch in range(args.max_epochs):
        
        # training step
        total = len(tr_dataset) // args.batch_size
        batch_iterator = get_batched_dataset(tr_dataset, args.batch_size, seed=epoch)
        for i,batch in tqdm(enumerate(batch_iterator), desc=f"Running epoch-{epoch}",total=total):
            current_context_input, response_input = data_collator(batch)
            state, metrics, drp_rng = train_step(state,current_context_input, response_input, drp_rng)
            print(metrics)
            if (i + 1) % args.logging_steps == 0:
                tr_loss = jax_utils.unreplicate(metrics["tr_loss"]).item()
                tqdm.write(str(dict(tr_loss=tr_loss, step=i+1)))
                logger.log({
                    "tr_loss": tr_loss,
                    "step": i + 1,
                }, commit=True)

        val_loss = jnp.array(0.)
        total = len(val_dataset) // args.batch_size
        val_batch_iterator = get_batched_dataset(val_dataset, args.batch_size, seed=None)
        for j, batch in tqdm(enumerate(val_batch_iterator), desc=f"evaluating after epoch-{epoch}", total=total):
            current_context_input, response_input = data_collator(batch)
            val_step_loss = val_step(state, current_context_input, response_input)
            val_loss += jax_utils.unreplicate(val_step_loss)

        val_loss = val_loss.item() / (j + 1)
        print(f"val_loss: {val_loss}")
        logger.log({"val_loss": val_loss}, commit=True)

        save_dir = args.save_dir + f"-epoch-{epoch}"
        save_checkpoint(save_dir, state, save_fn=model.save_pretrained, training_args=args)




       
if __name__ == "__main__":
    import json,os

    jsons = [{'context': 'Taste good though. ',
    'context/0': 'Deer are, in fact, massive ****.',
    'context/1': "Freaking deer. You can always tell the country people. They can't stand deer. Just giant garden destroying rats. \n\nHell, that's",
    'context/3': 'I lived in Germany when I was 5. I have this memory of my dad stopping to pick up a hedgehog that was in the middle of the road',
    'context/2': "Kinda like when visitors from more populated areas see deer here. They're not at all rare but people act like they are",
    'context_author': 'KakarotMaag',
    'response': "Ground venison mixed with beef fat is the best burger I've ever had.",
    'response_author': 'SG804',
    'subreddit': 'gifs',
    'thread_id': '98fur0'},{
    'context/1': "Hello, how are you?",
    'context/0': "I am fine. And you?",
    'context': "Great. What do you think of the weather?",
    'response': "It doesn't feel like February.",
    'context_author': 'KakarotMaag',
    'response_author': 'SG804',
    'subreddit': 'gifs',
    'thread_id': '98fur0'
    }]

    os.makedirs("data/",exist_ok=True)
    with open('data/dummy.jsonl', 'w') as outfile:
        for entry in jsons:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    args = TrainingArgs()
    logger = wandb.init(project="dummy-flax-conversational-model", config=asdict(args))
    logging_dict = dict(logger.config); logging_dict["save_dir"] += f"-{logger.id}"
    args = replace(args, **logging_dict)
    print(args)
    main(args, logger)
    
