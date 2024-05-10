from mu_transformer import MuTransformer


import functools
from typing import Any, Mapping

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import language

from collections.abc import Iterable
from typing import Any, Mapping, Tuple

from typing import Callable, Optional
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import numpy as onp

from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask, find_smallest_divisor

State = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray

from collections.abc import Sequence
from typing import Any, Union

from haiku._src.typing import Initializer
import jax
import jax.numpy as jnp
import numpy as np
import haiku
from haiku.initializers import * 

def _compute_fans(shape, fan_in_axes=None):
  """Computes the number of input and output units for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    if fan_in_axes is not None:
      # Compute fan-in using user-specified fan-in axes.
      fan_in = np.prod([shape[i] for i in fan_in_axes])
      fan_out = np.prod([s for i, s in enumerate(shape)
                         if i not in fan_in_axes])
    else:
      # If no axes specified, assume convolution kernels (2D, 3D, or more.)
      # kernel_shape: (..., input_depth, depth)
      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

class MupVarianceScaling(hk.initializers.Initializer):
  """Initializer which adapts its scale to the shape of the initialized array.

  The initializer first computes the scaling factor ``s = scale / n``, where n
  is:

    - Number of input units in the weight tensor, if ``mode = fan_in``.
    - Number of output units, if ``mode = fan_out``.
    - Average of the numbers of input and output units, if ``mode = fan_avg``.

  Then, with ``distribution="truncated_normal"`` or ``"normal"``,
  samples are drawn from a distribution with a mean of zero and a standard
  deviation (after truncation, if used) ``stddev = sqrt(s)``.

  With ``distribution=uniform``, samples are drawn from a uniform distribution
  within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.

  The variance scaling initializer can be configured to generate other standard
  initializers using the scale, mode and distribution arguments. Here are some
  example configurations:

  ==============  ==============================================================
  Name            Parameters
  ==============  ==============================================================
  glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
  glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
  lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
  lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
  he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
  he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
  ==============  ==============================================================
  """

  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal',
               fan_in_axes=None):
    """Constructs the :class:`VarianceScaling` initializer.

    Args:
      scale: Scale to multiply the variance by.
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``normal`` or ``uniform``.
      fan_in_axes: Optional sequence of int specifying which axes of the shape
        are part of the fan-in. If none provided, then the weight is assumed
        to be like a convolution kernel, where all leading dimensions are part
        of the fan-in, and only the trailing dimension is part of the fan-out.
        Useful if instantiating multi-headed attention weights.
    """
    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.fan_in_axes = fan_in_axes

  def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape, self.fan_in_axes)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      # distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      # stddev = stddev / distribution_stddev
      return TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      return RandomUniform(minval=-limit, maxval=limit)(shape, dtype)
    


class _MuTransformerTask(base.Task):
  """Tranformer from a dictionary configuration."""

  def __init__(self, datasets, cfg: Mapping[str, Any], name: str = '__TransformerTask'):
    self.datasets = datasets
    self._cfg = cfg
    self._net = hk.transform_with_state(self._hk_forward)
    self._name = name

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    vocab_size = self.datasets.extra_info['vocab_size']
    mod = MuTransformer(
        num_heads=self._cfg['num_heads'],
        num_layers=self._cfg['num_layers'],
        d_model=self._cfg['d_model'],
        dropout_rate=self._cfg['dropout_rate'],
        vocab_size=vocab_size)
    mask = (batch['image'] != 0)
    logits = mod(batch['image'], mask=mask, is_training=True)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], vocab_size))
    return jnp.sum(loss * mask) / jnp.sum(mask)

  def init(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def init_with_state(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def loss(self, params, key, data):
    return self._net.apply(params, key, data)

  def loss_with_state(self, params, state, key, data):
    return self._net.apply(params, state, key, data)
  
  


class MuMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      input_mult=1.0,
      output_mult=1.0,
      hidden_mult=1.0,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs an MLP with MuP following table 8 of tensor programs V.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.input_mult = input_mult
    self.hidden_mult = hidden_mult
    # self.output_mult = output_mult
    self.activation = activation
    self.activate_final = activate_final
    self.get_adam_mup_lr_mul = {}
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      if index ==0:
        #input layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w':1.0,'b':1.0}
        
      elif index == len(output_sizes) - 1:
        #output layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=jnp.zeros,# RandomNormal(stddev=1., mean=0.),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w':1.0,'b':1.0}
      else:
        #hidden layer
        layers.append(hk.Linear(output_size=output_size,
                                w_init=MupVarianceScaling(1.0, "fan_in",  "truncated_normal"),
                                b_init=hk.initializers.RandomNormal(stddev=1., mean=0.),
                                with_bias=with_bias,
                                name="linear_%d" % index))
        self.get_adam_mup_lr_mul["~/linear_%d"  % index] = {'w':1.0/output_sizes[index-1],'b':1.0}
        
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None
    
    assert len(output_sizes) >= 2, "need more than one layer for MuMLP"
    
    self.output_mul =  output_mult * 1 / output_sizes[-2]
    hk.set_state("mup_lrs",self.get_adam_mup_lr_mul)

  @property
  def mup_lrs(self):
    return hk.get_state("mup_lrs")
  
  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i == 0:
        out = out * self.input_mult
      elif i < (num_layers - 1):
        out = out * self.hidden_mult

      hk.set_state("layer_%d_act_l1" % i, jnp.mean(jnp.abs(out)))
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out * self.output_mul
        

  def reverse(
      self,
      activate_final: Optional[bool] = None,
      name: Optional[str] = None,
  ) -> "MLP":
    """Returns a new MLP which is the layer-wise reverse of this MLP.

    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once.

    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.

    >>> mlp = hk.nets.MLP([1, 2, 3])
    >>> mlp_in = jnp.ones([1, 2])
    >>> y = mlp(mlp_in)
    >>> rev = mlp.reverse()
    >>> rev_mlp_out = rev(y)
    >>> mlp_in.shape == rev_mlp_out.shape
    True

    Args:
      activate_final: Whether the final layer of the MLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.

    Returns:
      An MLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self.activate_final
    if name is None:
      name = self.name + "_reversed"

    output_sizes = tuple(
        layer.input_size
        for layer in reversed(self.layers)
        if layer.input_size is not None
    )
    if len(output_sizes) != len(self.layers):
      raise ValueError("You cannot reverse an MLP until it has been called.")
    return MuMLP(
        output_sizes=output_sizes,
        w_init=self.w_init,
        b_init=self.b_init,
        with_bias=self.with_bias,
        activation=self.activation,
        activate_final=activate_final,
        name=name,
    )
  



class _MuMLPImageTask(_MLPImageTask):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0,
               input_mult=1.0,
               output_mult=1.0,
               hidden_mult=1.0):
    super().__init__(
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0)
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return MuMLP( #hk.nets.MLP(
          sizes, activation=act_fn,
              input_mult=input_mult, 
              output_mult=output_mult,
              hidden_mult=hidden_mult)(
              inp, dropout_rate=dropout_rate, 
              rng=hk.next_rng_key())

    self._mod = hk.transform_with_state(_forward)

  def loss_with_state(self, params, state, key, data):
    num_classes = self.datasets.extra_info["num_classes"]
    logits, state = self._mod.apply(params, state, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss), state

  def init_with_state(self, key: PRNGKey) -> base.Params:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])
  
  def loss_and_accuracy_with_state(self, params: Params, state: State, key: PRNGKey, data: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_classes = self.datasets.extra_info["num_classes"]

    threshold = 10000
    if data["image"].shape[0] > threshold:
      # If the batch is too large, we split it into smaller chunks.
      # This is to avoid running out of memory.
      # This is not necessary for the task to work, but it is useful for
      # large batch sizes.
      data["image"] = jnp.array_split(data["image"], 
                                      find_smallest_divisor(data["image"].shape[0],threshold), 
                                      axis=0)
      # print(jax.tree_map(lambda x: x.shape, data["image"]))
      # exit(0)
      to_cat = [self._mod.apply(params, state, key, chunk)[0] for chunk in data["image"]]

      logits = jnp.concatenate(to_cat)
    else:
      logits = self._mod.apply(params, state, key, data["image"])[0]
    
    # Calculate the loss as before
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(vec_loss)
    
    # Calculate the accuracy
    predictions = jnp.argmax(logits, axis=-1)
    actual = data["label"]
    correct_predictions = predictions == actual
    accuracy = jnp.mean(correct_predictions.astype(jnp.float32))
    
    return loss, accuracy
  

