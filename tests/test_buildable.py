import pickle
from abc import abstractmethod
from pathlib import Path

import pytest

from config_state import ConfigField
from config_state import ConfigState
from config_state.buildable import builder
from config_state.buildable import register


@builder
class MasterFoo(ConfigState):
  def __init__(self, config):
    super().__init__(config)

  @abstractmethod
  def method(self):
    pass


@builder
@register
class Foo(MasterFoo):
  param1: int = ConfigField(0, "Param1")

  def __init__(self, config):
    super().__init__(config)


@register
class SubFoo(Foo):
  param1: int = ConfigField(10, "Param1")

  def __init__(self, config):
    super().__init__(config)

  def method(self):
    pass


@register
class SubFoo2(Foo):
  param2: int = ConfigField(10, "Param2")

  def __init__(self, config):
    super().__init__(config)

  def method(self):
    pass


@register
class SubFoo3(SubFoo2):
  param3: int = ConfigField(1, "Param3")

  def __init__(self, config):
    super().__init__(config)


@builder
@register
class SubFoobuilder(Foo):
  def method(self):
    pass


@register
class SubFoo4(SubFoobuilder):
  pass


@register
class SubFoo5(SubFoobuilder):
  sub_foo: Foo = ConfigField(None, "sub foo", type=Foo)


class SubFoo6(Foo):
  """Not registered sub class"""

  def method(self):
    pass


def test_builder():
  config = {}

  foo = SubFoo(config)

  assert isinstance(foo, Foo)
  assert foo.param1 == 10

  config = {'class': 'SubFoo', 'param1': 13}

  subfoo = Foo(config)

  assert isinstance(subfoo, SubFoo)
  assert subfoo.param1 == 13

  config = {'class': 'SubFoo2', 'param2': 13}

  subfoo2 = Foo(config)

  assert isinstance(subfoo2, SubFoo2)
  assert subfoo2.param2 == 13

  config = {'class': 'SubFoo3', 'param2': 13, 'param3': 2}

  subfoo3 = Foo(config)

  assert isinstance(subfoo3, SubFoo3)
  assert subfoo3.param2 == 13
  assert subfoo3.param3 == 2

  config = {'class': 'SubFoo4'}
  SubFoobuilder(config)

  with pytest.raises(ValueError):
    Foo(config)

  # Ok, instantiation without using the builder
  SubFoo6({})


def test_nested_foo():
  config = {'class': 'SubFoo5',
            'sub_foo': {'class': 'SubFoo3', 'param3': 3, 'param1': 3}}
  foo = SubFoobuilder(config)
  assert isinstance(foo.sub_foo, SubFoo3)
  assert foo.sub_foo.param3 == 3
  assert foo.sub_foo.param1 == 3


def test_nested_foo_pickle(tmpdir):
  config = {'class': 'SubFoo5',
            'sub_foo': {'class': 'SubFoo3', 'param3': 3, 'param1': 3}}
  foo = SubFoobuilder(config)

  # 2. Save
  output_path = Path(tmpdir) / "test.pkl"
  pickle.dump(foo, open(output_path, 'wb'))
  assert output_path.exists()

  # 3. Check state after reload
  del foo
  foo = pickle.load(open(output_path, 'rb'))

  assert isinstance(foo.sub_foo, SubFoo3)
  assert foo.sub_foo.param3 == 3
  assert foo.sub_foo.param1 == 3


def test_master_builder():
  config = {'class': 'Foo.SubFoobuilder.SubFoo5',
            'sub_foo': {'class': 'SubFoo3', 'param3': 3, 'param1': 3}}

  foo = MasterFoo(config)
  assert isinstance(foo, SubFoo5)
  assert isinstance(foo.sub_foo, SubFoo3)
  assert foo.sub_foo.param3 == 3
  assert foo.sub_foo.param1 == 3
  assert config['class'] == 'Foo.SubFoobuilder.SubFoo5'
  from config_state.buildable import config_field_name_internal
  assert config_field_name_internal not in config
