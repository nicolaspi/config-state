from abc import abstractmethod

import tensorflow as tf

from config_state import builder
from config_state import ConfigField
from config_state import ConfigState
from config_state import register


@builder
class Optimizer(ConfigState):
  learning_rate: float = ConfigField(0.01, "The learning rate", force_type=True)

  @property
  @abstractmethod
  def keras_optimizer(self) -> tf.keras.optimizers.Optimizer:
    """Return a keras optimizer"""


@register
class Adam(Optimizer):
  beta_1: float = ConfigField(0.9, "beta_1 parameter", force_type=True)
  beta_2: float = ConfigField(0.9999, "beta_2 parameter", force_type=True)
  epsilon: float = ConfigField(1e-7, "epsilon parameter", force_type=True)

  @property
  def keras_optimizer(self) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                    beta_1=self.beta_1, beta_2=self.beta_2,
                                    epsilon=self.epsilon)


@register
class RMSprop(Optimizer):
  rho: float = ConfigField(0.9, "rho parameter", force_type=True)
  momentum: float = ConfigField(0.0, "momentum parameter", force_type=True)
  epsilon: float = ConfigField(1e-7, "epsilon parameter", force_type=True)

  @property
  def keras_optimizer(self) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate,
                                       rho=self.rho, momentum=self.momentum,
                                       epsilon=self.epsilon)
