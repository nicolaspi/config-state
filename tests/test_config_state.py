import json
import pickle
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from config_state import ConfigField
from config_state import ConfigState
from config_state import stateproperty
from config_state import StateVar
from config_state.config_state import _ReferenceContext
from config_state.config_state import DeferredConf
from config_state.exceptions import ConfigError
from tests.objects import Base
from tests.objects import Foo
from tests.objects import FooWithConfStateRef
from tests.objects import FooWithMultiConfStateRef
from tests.objects import NestedFoo
from tests.objects import NestedFoo2
from tests.objects import NestedListOfFoo
from tests.objects import Sub1
from tests.objects import Sub2
from tests.objects import Sub3
from tests.objects import SubFoo
from tests.objects import SubFooWithAliasRef
from tests.objects import SubFooWithAliasRef2
from tests.objects import SubFooWithMultiRef
from tests.objects import SubFooWithMultiRef2
from tests.objects import SubFooWithRef
from tests.objects import SubFooWithRef2
from tests.objects import SubFooWithRef3
from tests.utils import compare_states
from tests.utils import config_factory


def test_config_consistency():
  # missing required fields
  with pytest.raises(ConfigError):
    Foo(config={})

  # undeclared config field
  with pytest.raises(ConfigError):
    Foo(config={'license_key': '1234', 'undeclared_config': True})

  # instantiate objects with different configs
  foo = Foo(config={'license_key': '1234'})
  foo2 = Foo(config={'license_key': '1225'})

  assert (foo.license_key == '1234')
  assert (foo2.license_key == '1225')

  # Can not change a static config
  with pytest.raises(ConfigError):
    Foo(config={'license_key': '1234', 'log_dir': "log_dir/is/static"})

  # config has wrong type
  with pytest.raises(ConfigError):
    Foo(config={'license_key': '1234', 'learning_rate': '0.1'})

  # stateproperty without setter
  with pytest.raises(TypeError):

    class BadFoo(ConfigState):

      @stateproperty
      def param(self):
        return None


def test_dynamic_code_not_run():

  class PropFoo(ConfigState):
    def __init__(self, config):
      super().__init__(config)
      self._param = 0

    @stateproperty
    def param(self):
      assert '___props_initialized___' in type(self).__dict__, "This code should not run uppon class initialization"
      return self._param

    @param.setter
    def param(self, value):
      self._param = value

  foo = PropFoo(config={})
  assert foo.param == 0


def test_consistency():

  class BadConfig(ConfigState):

    def __init__(self, config=None):
      super().__init__(config=config)
      self.invalid_field = ConfigField(
          None, "ConfigFields should be class "
          "attributes")

  badconfig = BadConfig()
  with pytest.raises(SyntaxError):
    badconfig.check_validity()

  class BadStateVar(ConfigState):

    def not_the_init_method(self):
      self.invalid_var = StateVar(None,
                                  "StateVars should be declared in `__init__`")

  badvar = BadStateVar()
  badvar.not_the_init_method()
  with pytest.raises(SyntaxError):
    badvar.check_validity()

  # type of ConfigField should be a `type`
  with pytest.raises(AttributeError):

    class BadFoo3(ConfigState):
      param = ConfigField(type=Foo({'license_key': 123}),
                          doc="type should be a `type`")


def test_nested_config_state():
  foo = NestedFoo2(
      config={
          'sub_foo': {
              'learning_rate': 0.123,
              'license_key': '4321'
          },
          'license_key': '2134'
      })
  assert isinstance(foo.sub_foo, SubFoo)
  assert foo.sub_foo.learning_rate == 0.123
  assert foo.sub_foo.license_key == '4321'
  assert foo.license_key == '2134'

  foo = NestedFoo2(
      config={
          'sub_foo': SubFoo({
              'learning_rate': 0.123,
              'license_key': '4321'
          }),
          'license_key': '2134'
      })

  assert isinstance(foo.sub_foo, SubFoo)
  assert foo.sub_foo.learning_rate == 0.123
  assert foo.sub_foo.license_key == '4321'
  assert foo.license_key == '2134'


def test_config_state_kwargs_init():
  foo = NestedFoo2(
      **{
          'sub_foo': {
              'learning_rate': 0.123,
              'license_key': '4321'
          },
          'license_key': '2134'
      })
  assert isinstance(foo.sub_foo, SubFoo)
  assert foo.sub_foo.learning_rate == 0.123
  assert foo.sub_foo.license_key == '4321'
  assert foo.license_key == '2134'

  foo = NestedFoo2(
      **{
          'sub_foo': SubFoo({
              'learning_rate': 0.123,
              'license_key': '4321'
          }),
          'license_key': '2134'
      })

  assert isinstance(foo.sub_foo, SubFoo)
  assert foo.sub_foo.learning_rate == 0.123
  assert foo.sub_foo.license_key == '4321'
  assert foo.license_key == '2134'


def test_nested_with_default_value():

  class NestedFooWithDefaultValue(Foo):
    sub_foo: SubFoo = ConfigField(value={'license_key': 0},
                                  type=SubFoo,
                                  doc="A ConfigState as config field with a "
                                  "default license value")

  foo = NestedFooWithDefaultValue({'license_key': '2134'})

  assert foo.sub_foo.license_key == 0

  foo = NestedFooWithDefaultValue({
      'sub_foo': SubFoo({
          'learning_rate': 0.123,
          'license_key': '4321'
      }),
      'license_key': '2134'
  })

  assert foo.sub_foo.license_key == '4321'


def test_config_update():
  foo = Foo(config={'license_key': '1234'})

  assert foo.learning_rate == 0.1  # default value
  # config fields are immutable
  with pytest.raises(AttributeError):
    foo.learning_rate = 0.01

  foo = Foo(config={'license_key': '1234', 'learning_rate': 0.01})
  assert foo.learning_rate == 0.01  # user configured value

  class_rates = {'cat': 0.2, 'dog': 0.8}
  foo = Foo(config={'license_key': '1234', 'class_rates': class_rates})
  assert foo.class_rates == {'cat': 0.2, 'dog': 0.8}

  # /!\ The config updates are not deep copied /!\
  class_rates['cat'] = 0.5
  assert foo.class_rates == class_rates


def test_config_implicit_conversion():

  class Foo(ConfigState):
    path: Path = ConfigField(type=Path, doc="path param")

  foo = Foo({'path': "str_path/folder"})
  assert isinstance(foo.path, Path)
  assert foo.path == Path("str_path/folder")

  class Foo2(ConfigState):
    param: float = ConfigField(0.0)

  foo = Foo2({'param': '0.1'})
  assert isinstance(foo.param, float)
  assert foo.param == 0.1

  foo = Foo2({'param': True})
  assert isinstance(foo.param, float)
  assert foo.param == 1.0

  foo = Foo2({'param': False})
  assert isinstance(foo.param, float)
  assert foo.param == 0.0

  # Falling conversion
  with pytest.raises(ConfigError):
    Foo2({'param': 'ONE'})


def test_factory():

  def factory(str_date):
    return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

  class Foo(ConfigState):
    date: datetime = ConfigField(type=datetime,
                                 doc="some date",
                                 factory=factory)

  foo = Foo({'date': '2019-01-01 00:00:00'})
  assert isinstance(foo.date, datetime)
  assert foo.date == datetime.strptime('2019-01-01 00:00:00',
                                       '%Y-%m-%d %H:%M:%S')


def test_state_var_manip():
  foo = Foo(config={'license_key': '1234'})

  # state variables can mutate
  assert isinstance(foo.iteration, int)
  assert foo.iteration == 0
  foo.iteration = 2
  assert foo.iteration == 2
  foo.iteration += 1
  assert foo.iteration == 3

  assert isinstance(foo.param, float)
  foo.param = 3.14
  assert foo.param == 3.14


def test_object_state_manip():
  foo = Foo(config={'license_key': '1234'})
  assert foo.iteration == 0

  # extract the state of the object
  state = foo.get_state()
  state.internal_state['iteration'].value = 10

  # cannot change config field values in state object
  with pytest.raises(FrozenInstanceError):
    state.config['learning_rate'].value = 0.01

  # update object with new state
  foo.set_state(state)
  assert foo.iteration == 10


def test_config_in_init_access():

  class SubFoo(Foo):
    license_key = ConfigField('1234')
    alpha_param = ConfigField(0.6)

    def __init__(self, config=None):
      super().__init__(config=config)
      # Accessing the config fields in the constructor is Ok
      assert isinstance(self.learning_rate, float)
      assert isinstance(self.alpha_param, float)

  SubFoo()
  SubFoo({'license_key': '4321', 'alpha_param': 0.1})


def test_inheritance():
  foo = SubFoo({'license_key': 0, 'class_rates': {'cat': 1.0}})

  assert foo.learning_rate == 0.5  # overridden default config value
  assert isinstance(foo.param, int)  # overridden state var init type
  assert foo.use_dropout  # sub class config
  assert foo.dropout_rate == 0.0
  assert isinstance(foo.date, datetime)  # parent class config
  assert foo.class_rates == {'cat': 1.0}  # parent class config update
  # parent class state var update
  foo.iteration = 10
  foo.param = 42
  assert foo.iteration == 10
  assert foo.param == 42


def test_nested():
  sub2 = Sub2()
  assert isinstance(sub2.sub1, Sub1)

  sub3 = Sub3()
  assert isinstance(sub3.sub1, Sub1)

  sub3 = Sub3({'param': 'value', 'sub1': {'param': 'sub1_value'}})
  assert sub3.param == 'value'
  assert sub3.sub1.param == 'sub1_value'


def test_nested_using_factory(tmpdir):
  config = {'learning_rate': 0.321, 'license_key': 0}

  obj = NestedFoo({'sub_conf': config, 'license_key': 0})
  assert obj.sub_conf.learning_rate == 0.321

  config_file = Path(tmpdir) / 'config.json'
  json.dump(config, open(config_file, 'w'))

  obj = NestedFoo({'sub_conf': str(config_file), 'license_key': 0})
  assert obj.sub_conf.learning_rate == 0.321

  class LocalNestedFoo(Foo):
    sub_conf: SubFoo = ConfigField(str(config_file),
                                   type=SubFoo,
                                   doc="A ConfigState as config field",
                                   factory=config_factory)

  obj = LocalNestedFoo({'license_key': 0})
  assert obj.sub_conf.learning_rate == 0.321


def test_deferred():
  foo = Foo({'license_key': '...'})

  # ok, Ellipsis
  foo.license_key = 1337

  # not Ok
  with pytest.raises(AttributeError):
    foo.license_key = '1338'

  foo = Foo({'license_key': '123', 'learning_rate': '...'})
  with pytest.raises(ConfigError):
    foo.learning_rate = '0.1'

  foo.learning_rate = 0.1

  with pytest.raises(ConfigError):
    Foo({'license_key': '123', 'log_dir': '...'})

  sub3 = Sub3({'param': 'value', 'sub1': {'param': '...'}})
  assert sub3.param == 'value'
  assert isinstance(sub3.sub1.param, DeferredConf)

  sub3.sub1.param = 'updated_value'
  assert sub3.sub1.param == 'updated_value'

  with pytest.raises(AttributeError):
    sub3.sub1.param = 'new_updated_value'

  class SubFoo(Foo):
    license_key = ConfigField(..., "Deferred by default")

  foo = SubFoo()
  foo.license_key = '1233'
  assert foo.license_key == '1233'

  foo2 = SubFoo()
  assert isinstance(foo2.license_key, DeferredConf)
  foo2.license_key = '4567'
  assert foo2.license_key == '4567'


def test_no_init():

  class SubNoInit(Base):
    pass

  sub = SubNoInit({'param': 'value'})
  assert sub.param == 'value'


def test_multi_bases_multi_param_constructor():

  class Base0:

    def __init__(self, param1=None, param2=None):
      self.param1 = param1
      self.param2 = param2

  class Sub(Base0, Base):

    def __init__(self, param1=None, config=None):
      Base.__init__(self, config=config)
      Base0.__init__(self, param1)

  sub = Sub(config={'param': 'value'})
  assert sub.param == 'value'
  assert sub.param1 is None
  assert sub.param2 is None

  sub = Sub(param1=True, config={'param': 'value'})
  assert sub.param == 'value'
  assert sub.param1 == True
  assert sub.param2 is None

  #Empty kwargs constructor call
  sub = Sub(True, {'param': 'value'})
  assert sub.param == 'value'
  assert sub.param1 == True
  assert sub.param2 is None


def test_references():
  foo = SubFooWithRef({
      'param_ref': 'ref_license',
      'param_ref2': 'ref2_license',
      'date_ref': '2020-02-02 00:00:00'
  })
  foo2 = SubFooWithRef({
      'param_ref': 'ref_license2',
      'param_ref2': 'ref2_license2',
      'date_ref': '2021-02-02 00:00:00'
  })

  assert foo.nested_foo.license_key == 'ref_license'
  assert foo.nested_foo2.license_key == 'ref2_license'
  assert foo.nested_foo.date == datetime.strptime('2020-02-02 00:00:00',
                                                  '%Y-%m-%d %H:%M:%S')
  assert foo.param_ref == 'ref_license'
  assert foo.param_ref2 == 'ref2_license'
  assert foo.nested_foo.date is foo.date_ref

  assert foo2.nested_foo.license_key == 'ref_license2'
  assert foo2.nested_foo2.license_key == 'ref2_license2'
  assert foo2.param_ref == 'ref_license2'
  assert foo2.param_ref2 == 'ref2_license2'
  assert foo2.nested_foo.date == datetime.strptime('2021-02-02 00:00:00',
                                                   '%Y-%m-%d %H:%M:%S')
  assert foo2.nested_foo.date is foo2.date_ref

  foo3 = SubFooWithRef({
      'param_ref': 'ref_license2',
      'param_ref2': 'ref2_license2'
  })
  assert foo3.nested_foo.date == datetime.strptime('2019-01-01 00:00:00',
                                                   '%Y-%m-%d %H:%M:%S')
  assert foo3.nested_foo.date is foo3.date_ref

  foo4 = SubFooWithRef2({"ref": "ref_param"})

  assert foo4.nested_foo.sub1.param == "ref_param"
  assert foo4.ref == "ref_param"
  assert foo4.ref is foo4.nested_foo.sub1.param

  foo5 = SubFooWithRef3({'ref': 'ref_param2'})

  assert foo5.ref == 'ref_param2'
  assert foo5.ref is foo5.foo_with_ref.nested_foo.sub1.param
  assert foo5.value_ref == 0
  assert foo5.value_ref is foo5.foo_with_ref.value

  foo6 = SubFooWithMultiRef({'ref': 'ref_param3'})

  assert foo6.ref == 'ref_param3'
  assert foo6.ref is foo6.nested_foo.sub1.param
  assert foo6.ref is foo6.nested_foo2.sub1.param

  class SubListNotRef(ConfigState):
    ref = ConfigField([])

  foo = SubListNotRef()
  assert isinstance(foo.ref, list)
  assert len(foo.ref) == 0

  foo = SubFooWithAliasRef({
      'alias_param': 1,
      'alias_param_ref': '12345',
      'alias_date_ref': '2021-03-03 00:00:00'
  })

  assert foo.alias_date_ref == foo.date_ref
  assert foo.alias_date_ref == foo.nested_foo.date
  assert foo.alias_date_ref == datetime.strptime('2021-03-03 00:00:00',
                                                 '%Y-%m-%d %H:%M:%S')
  assert foo.alias_param_ref == foo.param_ref
  assert foo.alias_param_ref == foo.nested_foo.license_key
  assert foo.alias_param_ref == '12345'

  assert foo.alias_param == foo.param
  assert foo.alias_param == 1

  foo = SubFooWithAliasRef({
      'param': 0,
      'alias_param_ref': '12345'
  })
  assert foo.alias_param == foo.param
  assert foo.alias_param == 0

  foo = SubFooWithAliasRef2({
      'alias_param_ref': '12345',
      'alias_date_ref': '2021-03-03 00:00:00'
  })

  assert foo.alias_date_ref == foo.date_ref
  assert foo.alias_date_ref == foo.nested_foo.date
  assert foo.alias_date_ref == datetime.strptime('2021-03-03 00:00:00',
                                                 '%Y-%m-%d %H:%M:%S')
  assert foo.alias_date_ref == foo.alias_of_alias

  assert foo.alias_param_ref == foo.param_ref
  assert foo.alias_param_ref == foo.nested_foo.license_key
  assert foo.alias_param_ref == '12345'

  foo = SubFooWithMultiRef2({
      'licenses_ref': '1234567',
      'dates_ref': '2021-04-04 00:00:00'
  })

  assert foo.licenses_ref == foo.nested_foo.license_key
  assert foo.licenses_ref == foo.nested_foo2.license_key
  assert foo.licenses_ref == '1234567'

  assert foo.dates_ref == foo.nested_foo.date
  assert foo.dates_ref == foo.nested_foo2.date
  assert foo.dates_ref == datetime.strptime('2021-04-04 00:00:00',
                                            '%Y-%m-%d %H:%M:%S')

  foo = FooWithConfStateRef({
      'license_key': '7890',
  })

  assert foo.sub_foo.nested_foo.license_key == '7890'
  assert foo.sub_foo.nested_foo2.license_key == '7890'

  assert len(_ReferenceContext.updated_refs) == 0

  # FooWithMultiConfStateRef.license_key links to multiple refs and should
  # be defined upstream to avoid ambiguity
  with pytest.raises(ConfigError):
    foo = FooWithMultiConfStateRef({'nested_foo_ref': {'license_key': '47586'}})

  # Not Ok
  with pytest.raises(ConfigError):
    foo = FooWithMultiConfStateRef({'license_key': '47586'})

  # Ok
  foo = FooWithMultiConfStateRef({'license_key': '47586', 'nested_foo_ref': {}})


def test_references_priority():
  # check config antecedence
  foo = SubFooWithAliasRef2({
      'alias_date_ref': '2021-03-03 00:00:00',
      'date_ref': '2021-04-04 00:00:00',
      'param_ref': 'should_be_shadowed',
      'nested_foo': {
          'license_key': 'should_be_shadowed2'
      },
      'alias_param_ref': '12345'
  })

  assert foo.alias_date_ref == foo.date_ref
  assert foo.alias_date_ref == foo.nested_foo.date
  assert foo.alias_date_ref == datetime.strptime('2021-03-03 00:00:00',
                                                 '%Y-%m-%d %H:%M:%S')
  assert foo.alias_param_ref == foo.param_ref
  assert foo.alias_param_ref == foo.nested_foo.license_key
  assert foo.alias_param_ref == foo.nested_foo2.license_key
  assert foo.alias_param_ref == '12345'

  # check config antecedence bis
  with pytest.raises(ConfigError):
    foo = SubFooWithAliasRef2({
        'date_ref': '2021-05-05 00:00:00',
        'nested_foo': {
            'license_key': '54321'
        },
        'nested_foo2': {
            'license_key': '54322'
        }
    })
  assert len(_ReferenceContext.updated_refs) == 0


def test_deferred_reference():
  foo = SubFooWithMultiRef2({'licenses_ref': ..., 'dates_ref': ...})
  foo.licenses_ref = '8765'
  foo.dates_ref = '2021-05-15 00:00:00'

  assert foo.licenses_ref == foo.nested_foo.license_key
  assert foo.licenses_ref == foo.nested_foo2.license_key
  assert foo.licenses_ref == '8765'

  assert foo.dates_ref == foo.nested_foo.date
  assert foo.dates_ref == foo.nested_foo2.date
  assert foo.dates_ref == datetime.strptime('2021-05-15 00:00:00',
                                            '%Y-%m-%d %H:%M:%S')


def test_save_load_cycle(tmpdir):

  class LocalFoo(Foo):
    local_param = ConfigField(1, "local param")

    def __init__(self, config=None):
      super().__init__(config)

  def change_state(foo: Foo):
    foo.iteration = 1337
    new_param = 38
    foo.param = new_param
    assert foo.param == new_param
    assert foo.iteration == 1337

  def test_object_pickle(foo: Foo):
    change_state(foo)
    test_pickle(foo, tmpdir)

  def test_pickle(foo: ConfigState, tmpdir):
    foo_type = type(foo)
    state = foo.get_state()
    foo_: ConfigState = object.__new__(type(foo))
    foo_.set_state(state)
    compare_states(state, foo_.get_state())

    output_path = Path(tmpdir) / "self.pkl"
    pickle.dump(foo, open(output_path, 'wb'))
    assert output_path.exists()

    type_output_path = Path(tmpdir) / "cls.pkl"
    pickle.dump(type(foo), open(type_output_path, 'wb'))
    assert type_output_path.exists()
    _foo_type = pickle.load(open(type_output_path, 'rb'))
    assert _foo_type == foo_type

    del foo
    foo = pickle.load(open(output_path, 'rb'))
    assert isinstance(foo, foo_type)
    state2 = foo.get_state()

    compare_states(state, state2)

  # Can't pickle dynamic, local object
  with pytest.raises(AttributeError):
    test_object_pickle(LocalFoo({'license_key': 0}))

  test_object_pickle(Foo({'license_key': 0}))
  test_object_pickle(SubFoo({'license_key': 0}))
  test_object_pickle(NestedFoo({'license_key': 0}))
  nested = NestedFoo2({'license_key': 0, 'sub_foo': {'license_key': 0}})
  test_object_pickle(nested)
  test_pickle(
      SubFooWithRef({
          'param_ref': 'ref_license',
          'param_ref2': 'ref2_license'
      }), tmpdir)
  test_pickle(SubFooWithRef2({"ref": "ref_param"}), tmpdir)
  test_pickle(
      SubFooWithAliasRef({
          'param': 0,
          'alias_param_ref': '12345',
          'alias_date_ref': '2021-03-03 00:00:00'
      }), tmpdir)


def test_config_hash():
  foo = Foo(config={'license_key': '1234', 'path': 'this/path'})
  nfoo = NestedFoo2({
      'license_key': 0,
      'sub_foo': {
          'license_key': 0
      },
      'path': 'this/path'
  })

  foo2 = Foo(config={'license_key': '1234', 'path': 'that_other/path'})
  nfoo2 = NestedFoo2({
      'license_key': 0,
      'sub_foo': {
          'license_key': 0
      },
      'path': 'that_other/path'
  })

  assert (foo.config_hash() == foo2.config_hash())
  assert (nfoo.config_hash() == nfoo2.config_hash())

  class NestedList(ConfigState):
    foos: List[Foo] = ConfigField(None, "List of foos")

  list_foo = NestedList(config={'foos': [foo, foo2]})
  list_foo2 = NestedList(config={'foos': [foo, foo]})
  list_foo3 = NestedList(config={'foos': [foo2, foo2]})

  assert (list_foo.config_hash() == list_foo2.config_hash())
  assert (list_foo.config_hash() == list_foo3.config_hash())


def test_clone():
  foo = Foo(config={'license_key': '1234'})
  foo.iteration == 2

  foo_clone = foo.clone()
  compare_states(foo.get_state(), foo_clone.get_state())

  foo_clone.iteration = 3
  assert foo.iteration != foo_clone.iteration

  foo1 = Foo(config={'license_key': '4321', 'path': 'that/path'})
  foo1.iteration = 123
  foo1.param = Foo(config={'license_key': 'foo_param'})

  list_foo = NestedListOfFoo(config={'foos': [foo1]})

  clone_list_foo = list_foo.clone()

  compare_states(list_foo.get_state(), clone_list_foo.get_state())

  list_foo.foos[0].iteration = 567
  list_foo.foos[0].param = 'a_string'

  assert list_foo.foos[0].iteration != clone_list_foo.foos[0].iteration
  assert list_foo.foos[0].param == 'a_string'
  assert isinstance(clone_list_foo.foos[0].param, Foo)
  assert clone_list_foo.foos[0].param.license_key == 'foo_param'
  assert hasattr(clone_list_foo.foos[0].param, 'ran_custom_builder_new')
