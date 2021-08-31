from datetime import datetime
from pathlib import Path

from config_state.config_state import ConfigField
from config_state.config_state import ConfigState
from config_state.config_state import reference
from config_state.config_state import stateproperty
from config_state.config_state import StateVar


def date_factory(str_date):
  return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')


class Foo(ConfigState):
  log_dir = ConfigField("./", "Path to output folder", static=True)
  learning_rate = ConfigField(0.1, "The learning rate", force_type=True)
  class_rates: dict = ConfigField({'cat': 0.5, 'dog': 0.5}, "Class rates")
  license_key: str = ConfigField(None,
                                 "License authorization key",
                                 mandatory=True)
  date: datetime = ConfigField(value='2019-01-01 00:00:00',
                               type=datetime,
                               doc="some date",
                               factory=date_factory)
  path: Path = ConfigField(value="./", type=Path, doc="path param")

  def __init__(self, config=None):
    super().__init__(config=config)
    self._param: float = 0.0
    self.iteration: int = StateVar(0, "Training iterations")

  @stateproperty
  def param(self):
    '''Model's parameter'''
    return self._param

  @param.setter
  def param(self, val):
    self._param = val


class SubFoo(Foo):
  learning_rate = ConfigField(0.5,
                              "Overrides the base class config",
                              force_type=True)
  use_dropout = ConfigField(True, force_type=True)
  none_field = ConfigField(None, "None field")

  def __init__(self, config=None):
    super().__init__(config)
    self.dropout_rate: float = StateVar(0.0, "Mutable dropout rate")
    self._param: int = 0


class NestedFoo(Foo):
  sub_conf: SubFoo = ConfigField(SubFoo({'license_key': 0}),
                                 "A ConfigState as config field")

  def __init__(self, config=None):
    super().__init__(config)


class NestedFoo2(Foo):
  sub_foo: SubFoo = ConfigField(type=SubFoo,
                                doc="A MetaConfigState as config field",
                                mandatory=True)

  def __init__(self, config=None):
    super().__init__(config)
    # logic inside __init__ using nested config state conf should work
    self.boosted_learning_rate = self.sub_foo.learning_rate * 10.0


class Base(ConfigState):
  param = ConfigField('default', "param description")

  def __init__(self, config=None):
    super().__init__(config)
    self.state_var = StateVar(0, "doc")


class Sub1(Base):

  def __init__(self, config=None):
    super().__init__(config)


class Sub2(Base):
  sub1: Base = ConfigField(type=Sub1, doc='Sub1 conf')

  def __init__(self, config=None):
    super().__init__(config)


class Sub3(Sub2):

  def __init__(self, config=None):
    super().__init__(config)


class SubFooWithRef(ConfigState):
  nested_foo: NestedFoo = ConfigField(type=NestedFoo)
  nested_foo2: NestedFoo = ConfigField(type=NestedFoo)
  param_ref = ConfigField(nested_foo.license_key,
                          doc="Reference to "
                          "nested_foo's "
                          "license id")
  param_ref2 = ConfigField(nested_foo2.license_key,
                           doc="Reference to "
                           "nested_foo2's "
                           "license id")
  date_ref = ConfigField(nested_foo.date)

  def __init__(self, config):
    super().__init__(config)


class SubFooWithRef2(ConfigState):
  value: int = ConfigField(0, "conf field with potentially conflicting name")
  nested_foo: Sub3 = ConfigField(type=Sub3)
  ref = ConfigField(nested_foo.sub1.param)


class SubFooWithRef3(ConfigState):
  foo_with_ref: SubFooWithRef2 = ConfigField(type=SubFooWithRef2)
  ref = ConfigField(foo_with_ref.ref)
  value_ref = ConfigField(foo_with_ref.value)


class SubFooWithMultiRef(ConfigState):
  nested_foo: Sub3 = ConfigField(type=Sub3)
  nested_foo2: Sub3 = ConfigField(type=Sub3)
  ref = ConfigField([nested_foo.sub1.param, nested_foo2.sub1.param])


class SubFooWithAliasRef(SubFooWithRef):
  alias_param_ref = reference(
      ['nested_foo.license_key', 'nested_foo2.license_key'])
  alias_date_ref = reference('date_ref')


class SubFooWithAliasRef2(SubFooWithRef):
  alias_date_ref = ConfigField(SubFooWithRef.date_ref)
  alias_param_ref = ConfigField([
      SubFooWithRef.nested_foo.license_key,
      SubFooWithRef.nested_foo2.license_key
  ])
  alias_of_alias = ConfigField(alias_date_ref)


class SubFooWithMultiRef2(SubFooWithRef):
  dates_ref = ConfigField([
      SubFooWithRef.nested_foo.date, SubFooWithRef.nested_foo2.date,
      SubFooWithRef.date_ref
  ])
  licenses_ref = ConfigField(
      [SubFooWithRef.param_ref, SubFooWithRef.param_ref2])


class FooWithConfStateRef(ConfigState):
  sub_foo: SubFooWithRef = ConfigField(type=SubFooWithRef)
  nested_foo_ref = ConfigField(sub_foo.nested_foo)
  nested_foo_ref2 = ConfigField(sub_foo.nested_foo2)
  license_key = ConfigField(
      [sub_foo.nested_foo.license_key, sub_foo.nested_foo2.license_key])


class FooWithMultiConfStateRef(ConfigState):
  sub_foo: SubFooWithRef = ConfigField(type=SubFooWithRef)
  nested_foo_ref = ConfigField([sub_foo.nested_foo, sub_foo.nested_foo2])
  license_key = ConfigField(
      [sub_foo.nested_foo.license_key, sub_foo.nested_foo2.license_key])

  def __init__(self, config=None):
    super().__init__(config=config)


class JsonableFoo(ConfigState):
  log_dir = ConfigField("./", "Path to output folder.", static=True)
  learning_rate = ConfigField(0.1, "The learning rate", force_type=True)

  def __init__(self, config=None):
    super().__init__(config=config)
    self.iteration = StateVar(0, "Training iterations")
