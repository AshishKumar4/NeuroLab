import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import orbax.checkpoint
from flax.metrics import tensorboard
from flax.training import train_state
import tensorflow_datasets as tfds
import optax
import orbax
from flax.training import orbax_utils
from typing import Any, Tuple, Mapping,Callable,List,Dict
import os
import tqdm
import time
import tensorflow as tf # For dataset
from clu import metrics
from flax import struct                # Flax dataclasses

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

# Define the TrainState 
class SimpleTrainState(train_state.TrainState):
    rngs: jax.random.PRNGKey
    metrics: Metrics

    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

class SimpleTrainer:
    state : SimpleTrainState
    best_state : SimpleTrainState
    best_loss : float
    model : nn.Module
    ema_decay:float = 0.999
    
    def __init__(self, 
                 model:nn.Module, 
                 input_shapes:Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 rngs:jax.random.PRNGKey,
                 train_state:SimpleTrainState=None,
                 name:str="Simple",
                 load_from_checkpoint:bool=False,
                 checkpoint_suffix:str="",
                 loss_fn=optax.l2_loss,
                 param_transforms:Callable=None,
                 ):
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(self.checkpoint_path() + checkpoint_suffix, checkpointer, options)

        if load_from_checkpoint:
            latest_step, old_state, old_best_state = self.load()
        else:
            latest_step, old_state, old_best_state = 0, None, None
            
        self.latest_step = latest_step

        if train_state == None:
            self.init_state(optimizer, rngs, existing_state=old_state, existing_best_state=old_best_state, model=model, param_transforms=param_transforms)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9
    
    def get_input_ones(self):
        return {k:jnp.ones((1, *v)) for k,v in self.input_shapes.items()}

    def init_state(self,
                   optimizer: optax.GradientTransformation, 
                   rngs:jax.random.PRNGKey,
                   existing_state:dict=None,
                   existing_best_state:dict=None,
                   model:nn.Module=None,
                   param_transforms:Callable=None
                   ):
        @partial(jax.pmap, axis_name="device")
        def init_fn(rngs):
            rngs, subkey = jax.random.split(rngs)

            if existing_state == None:
                input_vars = self.get_input_ones()
                params = model.init(subkey, **input_vars)

            # if param_transforms is not None:
            #     params = param_transforms(params)
                
            state = SimpleTrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optimizer,
                rngs=rngs,
                metrics=Metrics.empty()
            )
            return state
        self.state = init_fn(jax.device_put_replicated(rngs, jax.devices()))
        self.best_loss = 1e9
        if existing_best_state is not None:
            self.best_state = self.state.replace(params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            self.best_state = self.state
            
    def get_state(self):
        return flax.jax_utils.unreplicate(self.state)

    def get_best_state(self):
        return flax.jax_utils.unreplicate(self.best_state)

    def checkpoint_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./checkpoints'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def tensorboard_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./tensorboard'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self):
        step = self.checkpointer.latest_step()
        print("Loading model from checkpoint", step)
        ckpt = self.checkpointer.restore(step)
        state = ckpt['state']
        best_state = ckpt['best_state']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        print(f"Loaded model from checkpoint at step {step}", ckpt['best_loss'])
        return step, state, best_state

    def save(self, epoch=0):
        print(f"Saving model at epoch {epoch}")
        ckpt = {
            # 'model': self.model,
            'state': self.get_state(),
            'best_state': self.get_best_state(),
            'best_loss': self.best_loss
        }
        try:
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpointer.save(epoch, ckpt, save_kwargs={'save_args': save_args}, force=True)
            pass
        except Exception as e:
            print("Error saving checkpoint", e)

    def _define_train_step(self, **kwargs):
        model = self.model
        loss_fn = self.loss_fn
        
        @partial(jax.pmap, axis_name="device")
        def train_step(state:SimpleTrainState, batch):
            """Train for a single step."""
            images = batch['image']
            labels= batch['label']
            
            def model_loss(params):
                preds = model.apply(params, images)
                expected_output = labels
                nloss = loss_fn(preds, expected_output)
                loss = jnp.mean(nloss)
                return loss
            loss, grads = jax.value_and_grad(model_loss)(state.params)
            grads = jax.lax.pmean(grads, "device")
            state = state.apply_gradients(grads=grads) 
            return state, loss
        return train_step
    
    def _define_compute_metrics(self):
        model = self.model
        loss_fn = self.loss_fn
        
        @jax.jit
        def compute_metrics(state:SimpleTrainState, batch):
            preds = model.apply(state.params, batch['image'])
            expected_output = batch['label']
            loss = jnp.mean(loss_fn(preds, expected_output))
            metric_updates = state.metrics.single_from_model_output(loss=loss, logits=preds, labels=expected_output)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def summary(self):
        input_vars = self.get_input_ones()
        print(self.model.tabulate(jax.random.key(0), **input_vars, console_kwargs={"width": 200, "force_jupyter":True, }))
    
    def config(self):
        return {
            "model": self.model,
            "state": self.state,
            "name": self.name,
            "input_shapes": self.input_shapes
        }
        
    def init_tensorboard(self, batch_size, steps_per_epoch, epochs):
        summary_writer = tensorboard.SummaryWriter(self.tensorboard_path())
        summary_writer.hparams({
            **self.config(),
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size
        })
        return summary_writer
        
    def fit(self, data, steps_per_epoch, epochs, train_step_args={}):
        train_ds = iter(data['train']())
        if 'test' in data:
            test_ds = data['test']
        else:
            test_ds = None
        train_step = self._define_train_step(**train_step_args)
        compute_metrics = self._define_compute_metrics()
        state = self.state
        device_count = jax.device_count()
        # train_ds = flax.jax_utils.prefetch_to_device(train_ds, jax.devices())
        
        summary_writer = self.init_tensorboard(data['batch_size'], steps_per_epoch, epochs)
        
        for epoch in range(epochs):
            current_epoch = self.latest_step + epoch + 1
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0
            
            with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step') as pbar:
                for i in range(steps_per_epoch):
                    batch = next(train_ds)
                    batch = jax.tree.map(lambda x: x.reshape((device_count, -1, *x.shape[1:])), batch)
                    # print(batch['image'].shape)
                    state, loss = train_step(state, batch)
                    loss = jnp.mean(loss)
                    # print("==>", loss)
                    epoch_loss += loss
                    if i % 100 == 0:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
                        current_step = current_epoch*steps_per_epoch + i
                        summary_writer.scalar('Train Loss', loss, step=current_step)
                        
            end_time = time.time()
            self.state = state
            total_time = end_time - start_time
            avg_time_per_step = total_time / steps_per_epoch
            avg_loss = epoch_loss / steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = state
                self.save(current_epoch)
            
            # Compute Metrics
            metrics_str = ''
            # if test_ds is not None:
            #     for test_batch in iter(test_ds()):
            #         state = compute_metrics(state, test_batch)
            #     metrics = state.metrics.compute()
            #     for metric,value in metrics.items():
            #         summary_writer.scalar(f'Test {metric}', value, step=current_epoch)
            #         metrics_str += f', Test {metric}: {value:.4f}'
            #     state = state.replace(metrics=Metrics.empty())
                    
            print(f"\n\tEpoch {current_epoch} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Best Loss: {self.best_loss} {metrics_str}")
            
        self.save(epochs)
        return self.state
