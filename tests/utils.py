import json
from pathlib import Path

from config_state import ObjectState
from config_state import Serializer
from config_state.config_state import ConfigState


def compare_states(state1: ObjectState, state2: ObjectState):

  def compare_dicts(dict1, dict2):

    def compare_field(v1, v2):
      assert type(v1) is type(v2)
      if isinstance(v1, ConfigState):
        compare_states(v1.get_state(), v2.get_state())
      elif isinstance(v1, ObjectState):
        compare_states(v1, v2)
      elif isinstance(v1, (list, tuple)):
        assert len(v1) == len(v2)
        [compare_field(v, w) for v, w in zip(v1, v2)]
      elif isinstance(v1, dict):
        assert set(v1.keys()) == set(v2.keys())
        for k in v1.keys():
          compare_field(v1[k], v2[k])
      else:
        assert v1 == v2

    assert set(dict1.keys()) == set(dict2.keys())

    for k in dict1.keys():
      v1 = dict1[k]
      v2 = dict2[k]
      compare_field(v1.value, v2.value)

  compare_dicts(state1.internal_state, state2.internal_state)
  compare_dicts(state1.config, state2.config)


def save_load_compare(serializer: Serializer, foo: ConfigState, tmpdir):
  state = foo.get_state()

  output_path = Path(tmpdir) / "test.pkl"
  serializer.save(foo, output_path)
  assert output_path.exists()

  del foo
  foo = serializer.load(output_path)
  state_loaded = foo.get_state()

  compare_states(state, state_loaded)


def config_factory(config_info) -> dict:
  if config_info is None:
    return None
  if isinstance(config_info, dict):
    return config_info
  if isinstance(config_info, str):
    config_path = Path(config_info)

    if config_path.suffix == '.json':
      config = json.load(open(config_path, 'r'))
    else:
      raise ValueError(f"Unknown file format for {config_path}")

  return config
