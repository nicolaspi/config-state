import pickle
from typing import IO

from config_state import ConfigField
from config_state import register
from config_state.config_state import ObjectState
from config_state.serializers.serializer import Serializer


@register
class Pickle(Serializer):
  is_binary: bool = ConfigField(True,
                                "Whether the serializer is binary",
                                static=True)
  protocol_version: int = ConfigField(pickle.HIGHEST_PROTOCOL, "Pickle's "
                                      "protocol version")

  def __init__(self, config=None):
    super().__init__(config)

  def _dump(self, object_state: ObjectState, stream: IO[bytes]):
    pickle.dump(object_state, stream, protocol=self.protocol_version)

  def _load_state(self, stream: IO[bytes]) -> ObjectState:
    object = pickle.load(stream)
    return object
