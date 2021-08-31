import os

import tensorflow as tf
import yaml

from config_state import ConfigField
from config_state import ConfigState
from config_state import DeferredConf
from config_state import StateVar
from examples.tensorflow.dataset import Dataset
from examples.tensorflow.model import Model
from examples.tensorflow.optimizer import Optimizer


class MLExperiment(ConfigState):
  dataset: Dataset = ConfigField(type=Dataset,
                                 doc="The dataset to train the model on")
  model: Model = ConfigField(type=Model, doc="The model to train.")
  optimizer: Optimizer = ConfigField(
      type=Optimizer, doc="The optimizer used to train the model")

  def __init__(self, config):
    super().__init__(config)

    if isinstance(self.model.input_shape, DeferredConf):
      self.model.input_shape = self.dataset.get_shape()

    if isinstance(self.model.output_units, DeferredConf):
      self.model.output_units = self.dataset.get_num_classes()

    self.model.keras_model.compile(
        optimizer=self.optimizer.keras_optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics="accuracy")
    self.epoch: int = StateVar(0, "the current training epoch")

  def run(self, epochs):
    print(f"Training for {epochs} epochs...")

    train_set = self.dataset.get_train_set()
    test_set = self.dataset.get_test_set()

    self.model.keras_model.fit(train_set,
                               initial_epoch=self.epoch,
                               epochs=epochs + self.epoch,
                               validation_data=test_set)
    self.epoch += epochs

    print("Training finished")


if __name__ == "__main__":
  config_path = os.path.join(os.path.dirname(__file__), "configs/cnn.yml")
  config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
  experiment = MLExperiment(config)
  print(experiment.model.keras_model.summary())
  experiment.run(epochs=15)
