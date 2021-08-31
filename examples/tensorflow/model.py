from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from config_state import builder
from config_state import ConfigField
from config_state import ConfigState
from config_state import DeferredConf
from config_state import register
from config_state import stateproperty


@builder
class Model(ConfigState):
  input_shape: Tuple[int] = ConfigField(...,
                                        "Input shape of the model",
                                        type=tuple)
  output_units: Optional[int] = ConfigField(...,
                                            "Model's output units count",
                                            type=int)

  def __init__(self, config):
    super().__init__(config)
    self._keras_model = None

  @abstractmethod
  def _build_keras_model(self) -> tf.keras.Model:
    """Build the keras model"""

  @property
  def keras_model(self) -> tf.keras.Model:
    if self._keras_model is None and not isinstance(
        self.input_shape, DeferredConf) and not isinstance(
            self.output_units, DeferredConf):
      self._keras_model = self._build_keras_model()
    return self._keras_model

  @stateproperty
  def weights(self):
    return self.keras_model.get_weights()

  @weights.setter
  def weights(self, weights):
    self.keras_model.set_weights(weights)


@register
class MultiLayerPerceptron(Model):
  structure: List[int] = ConfigField([128], "hidden structure of the MLP")
  dropout_rate: float = ConfigField(
      0.0, "Dropout rate applied on the last "
      "hidden layer.")

  def _build_keras_model(self) -> tf.keras.Model:
    layers = [tf.keras.layers.Flatten(input_shape=self.input_shape)]
    for units in self.structure:
      layers.append(tf.keras.layers.Dense(units, activation='relu'))

    if self.dropout_rate > 0.0:
      layers.append(tf.keras.layers.Dropout(self.dropout_rate))

    if self.output_units is not None:
      layers.append(tf.keras.layers.Dense(self.output_units))

    return tf.keras.models.Sequential(layers)


@register
class CNN(Model):
  structure: List[Union[int, str]] = ConfigField([32, 'max', 64, 'max', 64],
                                                 "Convolutional structure. "
                                                 "Conv2D layers units "
                                                 "are integers, pooling "
                                                 "layers type are str among "
                                                 "'max' or 'average'.")

  def _build_keras_model(self) -> tf.keras.Model:
    layers = [tf.keras.layers.InputLayer(input_shape=self.input_shape)]

    for layer in self.structure:
      if isinstance(layer, int):
        layers.append(tf.keras.layers.Conv2D(layer, (3, 3), activation='relu'))
      elif layer == 'max':
        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
      elif layer == 'average':
        layers.append(tf.keras.layers.AveragePooling2D((2, 2)))
      else:
        raise ValueError(f"Unknown layer spec {layer}.")

    layers.append(tf.keras.layers.Flatten())

    if self.output_units is not None:
      layers.append(tf.keras.layers.Dense(self.output_units))

    return tf.keras.models.Sequential(layers)


@register
class Ensembler(Model):
  model: Model = ConfigField(type=Model, doc="The model to be ensembled")
  ensemble_size: int = ConfigField(2, "Size of the ensemble", force_type=True)
  input_shape = ConfigField(model.input_shape)
  output_units = ConfigField(model.output_units)

  def _build_keras_model(self) -> tf.keras.Model:
    models = [
        self.model._build_keras_model() for _ in range(self.ensemble_size)
    ]

    input = tf.keras.layers.InputLayer(input_shape=self.input_shape).output

    inputs = tf.keras.layers.Lambda(self.lambda_splitter)(input)

    outputs = []

    for sub_input, model in zip(inputs, models):
      sub_output = model(sub_input)
      outputs.append(sub_output)

    output = tf.keras.layers.Lambda(self.lambda_merger)(outputs)
    return tf.keras.Model(inputs=input, outputs=output)

  @tf.function
  def lambda_splitter(self, input: tf.Tensor, training: bool = False):
    outputs = []
    slice_size = tf.cast(tf.shape(input)[0] // self.ensemble_size, tf.int64)
    if training:
      tf.assert_equal(tf.math.mod(tf.shape(input)[0], self.ensemble_size), 0)
    for i in range(self.ensemble_size):
      if training:
        outputs.append(input[i * slice_size:(i + 1) * slice_size, :, :])
      else:
        outputs.append(input)
    return outputs

  @tf.function
  def lambda_merger(self, inputs, training=False):
    if training:
      output = tf.concat(inputs, axis=0)
    else:
      # Average during inference
      output = tf.add_n(inputs) / tf.cast(len(inputs), inputs[0].dtype)
    return output
