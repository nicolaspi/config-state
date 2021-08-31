from abc import abstractmethod
from pathlib import Path
from typing import IO
from typing import Union

from config_state import ConfigField
from config_state import ConfigState
from config_state import ObjectState
from config_state.buildable import builder


@builder
class Serializer(ConfigState):
  is_binary: bool = ConfigField(None,
                                "Whether the serializer is binary",
                                required=True)

  def __init__(self, config):
    super().__init__(config)

  @abstractmethod
  def _dump(self, object_state: ObjectState, stream: IO):
    """
    Serializes `object_state` into `stream`
    """

  @abstractmethod
  def _load_state(self, stream: IO) -> ObjectState:
    """
    Deserializes an `ObjectState`
    """

  def load_state(self, file: Union[IO, Path, str]) -> ObjectState:
    if isinstance(file, (Path, str)):
      mode = 'rb' if self.is_binary else 'r'
      file = Path(file).absolute()
      with open(file, mode) as stream:
        return self._load_state(stream)
    else:
      return self._load_state(file)

  def load(self, file: Union[IO, Path, str]) -> 'ConfigState':
    """
    Deserialize a `ConfigState` from `stream`
    """
    state = self.load_state(file)
    instance = object.__new__(state.type)
    instance.set_state(state)
    return instance

  def save(self, object: ConfigState, file: Union[IO, Path, str]):
    """
    Serializes `object` into `stream`
    """
    object.check_validity()
    state: ObjectState = object.get_state()
    if isinstance(file, (Path, str)):
      mode = 'wb' if self.is_binary else 'w'
      file = Path(file).absolute()
      with open(file, mode) as stream:
        self._dump(state, stream)
    else:
      self._dump(state, file)
