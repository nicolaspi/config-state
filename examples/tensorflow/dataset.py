from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

from config_state import ConfigField
from config_state import ConfigState


class Dataset(ConfigState):
  name: str = ConfigField("mnist", "The dataset name to load", force_type=True)
  batch_size: int = ConfigField(32, "Batch size", force_type=True)

  def __init__(self, config):
    super().__init__(config)
    self._info: tfds.core.DatasetInfo = tfds.builder(self.name).info
    if 'image' not in self._info.features or 'label' not in self._info.features:
      raise ValueError(f"Dataset {self.name} doesn't represent an image "
                       f"classification problem.")

  def get_num_classes(self) -> int:
    return self._info.features['label'].num_classes

  def get_shape(self) -> List[int]:
    return self._info.features['image'].shape

  def get_train_set(self) -> tf.data.Dataset:
    """returns the training set"""
    ds: tf.data.Dataset = tfds.load(self.name,
                                    split='train',
                                    as_supervised=True)
    ds = ds.shuffle(buffer_size=10 * self.batch_size).batch(self.batch_size)
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

  def get_test_set(self) -> tf.data.Dataset:
    """returns the test set"""
    ds: tf.data.Dataset = tfds.load(self.name,
                                    split='test',
                                    batch_size=self.batch_size,
                                    as_supervised=True)
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
