import dataclasses
import json
from pydoc import locate
from typing import IO

from config_state import ConfigField
from config_state import register
from config_state.config_state import ConfigState
from config_state.config_state import FrozenPortableField
from config_state.config_state import ObjectState
from config_state.config_state import PortableField
from config_state.serializers.serializer import Serializer


@register
class Json(Serializer):
  is_binary: bool = ConfigField(False,
                                "Whether the serializer is binary",
                                static=True)

  def __init__(self, config=None):
    super().__init__(config)

  def _dump(self, object_state: ObjectState, stream: IO):

    def _dict_factory(elements):

      def convert(k, v):
        if isinstance(v, type):
          v = '.'.join([v.__module__, v.__name__])
        elif isinstance(v, ConfigState):
          v = dataclasses.asdict(v.get_state(), dict_factory=_dict_factory)
        return k, v

      elements = [convert(k, v) for k, v in elements]
      return dict(elements)

    state_dict = dataclasses.asdict(object_state, dict_factory=_dict_factory)
    json.dump(state_dict, stream)

  def _convert(self, object, klass):
    if klass is ObjectState:
      _type = locate(object['type'])
      config = object['config']
      for k, v in config.items():
        config[k] = self._convert(v, FrozenPortableField)
      internal = object['internal_state']
      for k, v in internal.items():
        internal[k] = self._convert(v, PortableField)
      return ObjectState(type=_type, config=config, internal_state=internal)
    elif klass in [PortableField, FrozenPortableField]:
      type = locate(object['type'])

      object['value'] = self._convert(object['value'], type)

      fields = {f.name: object[f.name] for f in dataclasses.fields(klass)}
      return klass(**fields)
    elif object is None:
      return None
    else:
      if not isinstance(object, klass):
        raise TypeError(f"Could not deserialize value '{object}' into "
                        f"type '{klass.__name__}'")
      return object

  def _load_state(self, stream: IO) -> ObjectState:
    object_dict = json.load(stream)
    object_state: ObjectState = self._convert(object_dict, ObjectState)
    return object_state
