from pathlib import Path

from config_state import ObjectState
from config_state import Serializer
from config_state.config_state import ConfigState


def compare_states(state1: ObjectState, state2: ObjectState):

  def compare_dicts(dict1, dict2):

    def compare_field(v1, v2):
      if isinstance(v1.value, ConfigState):
        compare_states(v1.value.get_state(), v2.value.get_state())
      else:
        assert v1 == v2

    assert set(dict1.keys()) == set(dict2.keys())

    for k in dict1.keys():
      v1 = dict1[k]
      v2 = dict2[k]
      compare_field(v1, v2)

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
