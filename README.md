# config_state
![CI Status](https://github.com/nicolaspi/config-state/actions/workflows/ci.yml/badge.svg)

The **python** language is a flexible language often used as an interface to manipulate high performance libraries coded in less flexible native languages like **C/C++**. **ConfigState** is this idea applied on an higher level in the hierarchy, it provides a frame to bridge human-readable *configuration languages* (e.g. **json** or **yaml**) with **python**.

With *ConfigState* one can configure a complex hierarchy of python classes and instantiate them using a single configuration file.  To avoid pitfalls and enhance the developer's experience, *ConfigState* provides a frame preventing inconsistencies and raising explicit explanation in failing situations. The performance is optimized for low runtime overhead, most of the logic is done during the class definition.

 ## The ConfigState class
The core component is the class `ConfigState` that defines a pattern to represent python classes with two distinctive set of attributes: a set of immutable **configuration** values and a set of mutable **state** values.

The configuration is set upon initialization and is passed through the constructor. Once initialized, the configuration is frozen and cannot change.
The state variables constitute the mutable state of the instance and can be updated throughout its lifetime.
The configuration and state variables are meant to represent the necessary and sufficient information required to clone the object's instance. They can be used to save and restore the object from disk.

 - The configuration fields are defined using `ConfigField` class attributes. They can have typing constraints and be provided with a factory method for building complex types out of simpler/built-in ones.
 - State variables are defined using `StateVar` attributes within the constructor. They can alternatively be defined as class properties using `@stateproperty` if random logic execution is needed upon accession/modification.

Implementing a class inheriting from `ConfigState` as parent offers the following benefits:
- Provides clear semantic separation between the static configuration values and the mutable state variables.
- Configuration values and state variables are accessible through pythonic syntax and benefit from the IDE's type hinting feature.
- Using a configuration file, one can instantiate a complex hierarchy of python classes. A config field may be another `ConfigState` object allowing to define tree-like structured `ConfigState` hierarchies.
- A config field can be a reference to a nested `ConfigState` object's config field. This allows coupling between config fields. For example, configuration of a log folder path can be injected into the nested `ConfigState` objects through the configuration of the topmost `ConfigState` object.
- `ConfigState` objects can be serialized/deserialized into/from a stream. They are *pickleable* and in some cases *jsonable*.


## Basic usage
```python
from pathlib import Path

from config_state import ConfigField
from config_state import ConfigState
from config_state import StateVar
import numpy as np

class Foo(ConfigState):
    learning_rate: float = ConfigField(0.1, 'The learning rate', force_type=True)
    license_key: str = ConfigField(None, 'License key', required=True)
    log_dir: Path = ConfigField('./', 'Path to a folder', type=Path)

    def __init__(self, config=None):
        super().__init__(config=config)
        self.weights: np.ndarray = StateVar(np.random.random((10, 10)),
                                            'The weights of the model')
        self.iteration: int = StateVar(0, 'Training iterations')
```
We can instantiate a `ConfigState` with a dictionary (that may have been obtained from loading a json or yaml file):

```python
conf = {
        'learning_rate': 0.1,
        'license_key': 'ID123',
        'log_dir': 'logs/'
    }
foo = Foo(conf)
```
The configuration of `foo` can be summarized:
```python
print(foo.config_summary())
```

Output:
```sdtout
learning_rate: 0.1
license_key: ID123
log_dir: logs
```

Values are accessible with pythonic syntax (the IDE should be able to perform type hinting and code completion):
```python
assert isinstance(foo.learning_rate, float)
assert foo.learning_rate == 0.1
```
Config values are immutable:
```python
foo.learning_rate = 0.2 # Not OK, raises 'AttributeError: Updating a conf field is forbidden'
```
But changing a state variable is ok:
```python
foo.iteration += 1 # Ok, state variable
```

Missing required fields raises an exception:
```python
conf = {
        'learning_rate': 0.1,
        'log_dir': 'logs/'
    }
foo = Foo(conf) # ConfigError: Configuring 'Foo': Those required fields have not been specified {'license_key'}
```

Configuring invalid fields raise an exception:
```python
conf = {
        'color': 'red',
        'license_key': 'ID123'
    }
foo = Foo(conf) # ConfigError: Configuring 'Foo': Trying to update the conf field 'color' which has not been defined
```
Configuring with an invalid type raise an exception:
```python
conf = {
        'learning_rate': '0.1',
        'license_key': 'ID123'
    }
foo = Foo(conf) # ConfigError: Configuring 'Foo': Value `0.1` of type `<class 'str'>` is not compatible with specified type `float`
```
## State property

A state variable can be defined using properties with the `@stateproperty` decorator, this is convenient in case some logic need to be run while accessing or setting the variable.
```python
from config_state import ConfigState
from config_state import stateproperty

import numpy as np

class Model(ConfigState):
    def __init__(self, config):
        super().__init__(config)
        self._weights: np.ndarray = np.random.random((10, 10))

    @stateproperty
    def weights(self) -> np.ndarray:
        '''Weights of the model'''
        return self._weights

    @weights.setter
    def weights(self, val):
        self._weights = val
```

## Serialization
`ConfigState` objects are serializable if their config and state variables are serializable too. The state of an object is considered to be entirely encapsulated within the config values and the state variables. The state can be obtained with `foo.get_state()` which returns an `ObjectState` instance. Those objects represent the serialized information of a `ConfigState` object.
```python
import pickle

pickle.dump(foo, open('foo.pkl', 'wb'))
foo2 = pickle.load(open('foo.pkl', 'rb'))
```
In some cases, `ConfigState` objects are *json serializable*:

```python
from config_state.serializers import Json

class JsonableFoo(ConfigState):
    log_dir: str = ConfigField('log_dir/', 'Path to output folder')
    learning_rate: float = ConfigField(0.1, 'The learning rate')

    def __init__(self, config=None):
        super().__init__(config=config)
        self.iteration = StateVar(0, 'Training iterations')

foo = JsonableFoo()
# saving
Json().save(foo, 'foo.json')

# loading
foo = Json().load('foo.json')
```
Content of `foo.json`:
```json
{
  "type": "__main__.JsonableFoo",
  "config": {
    "__VERSION__": {
      "value": 1.0,
      "doc": "ConfigState protocol's version",
      "type": "builtins.float"
    },
    "log_dir": {
      "value": "log_dir/",
      "doc": "Path to output folder.",
      "type": "builtins.str"
    },
    "learning_rate": {
      "value": 0.1,
      "doc": "The learning rate",
      "type": "builtins.float"
    }
  },
  "internal_state": {
    "iteration": {
      "value": 0,
      "doc": "Training iterations",
      "type": "builtins.int"
    }
  }
}
```
*Pickle* and *Json* serializers are available as plugin:
```python
serializer = Serializer({'class': 'Pickle'})
serializer.save(foo, 'foo.pkl')
```

## Config field factory
### Implicit factory
If a `ConfigField` has a specified `type` but the type of the provided `value` is different, `type` is used as an implicit factory by calling `type(value)`. This is useful for nested `ConfigState` objects:
```python
class NestedFoo(ConfigState):
    license_key: str = ConfigField(type=str, required=True)
    foo: Foo = ConfigField(type=Foo,
                           doc='A ConfigState as config field',
                           required=True)
```
```python
conf = {
    'license_key': '4321',
    'foo': {
        'learning_rate': 0.1,
        'license_key': 'ID123',
        'log_dir': 'logs/'
    }
}
nested_foo = NestedFoo(conf) # Ok, nested_foo.foo is instantiated using conf['foo']
isinstance(nested_foo.foo, Foo) # True
```
### Explicit factory
A factory can be explicitly provided through a *callable*:
```python
from datetime import datetime

def date_factory(str_date):
    return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')

class DateFoo(ConfigState):
    date: datetime = ConfigField(value='2019-01-01 00:00:00', type=datetime,
                                 doc='some date',
                                 factory=date_factory)

date_foo = DateFoo({'date': '2021-04-28 00:00:00'})
print(type(date_foo.date)) # <class 'datetime.datetime'>
```
## Deferred config fields

It may happen that the full configuration of an object is not known at the time of its instantiation. In such case it is possible to defer their specification at a later time using `Ellipsis` :
```python
foo = Foo({'license_key': ...})

foo.license_key is Ellipsis # True

foo.license_key = 1337 # ok, we can update an Ellipsis

foo.license_key = 42 # Not OK, raises 'AttributeError: Updating a conf field is forbidden'

# Note: For convenience with configs defined within json or yaml files, strings '...' are interpreted as Ellipsis:
foo = Foo({'license_key': str('...')})

foo.license_key is Ellipsis # True
```

## Reference fields
A `ConfigField` can be references to fields in nested `ConfigState` fields simplifying the configuration of complex hierarchies:
```python
class FooWithRef(ConfigState):
    foo: Foo = ConfigField(type=Foo) # a nested ConfigState
    license_key = ConfigField(foo.license_key) # Reference to a nested field

# FooWithRef.license_key is a reference to FooWithRef.foo.license_key
# allowing to simplify the configuration, instead of:
FooWithRef({'foo':  {'license_key':  'ABC123'}})

# we can do:
foo_with_ref = FooWithRef({'license_key':  'ABC123'})

foo_with_ref.license_key ==  'ABC123'  # True
foo_with_ref.foo.license_key ==  'ABC123'  # True
foo_with_ref.foo.license_key is foo_with_ref.license_key # True
```

A reference can point to another reference:
```python
class FooWithRef2(ConfigState):
    foo_with_ref: FooWithRef = ConfigField(type=FooWithRef)
    license_key = ConfigField(foo_with_ref.license_key) # It is a reference to another reference

foo = FooWithRef2({'license_key': 'ABC123'})

foo.foo_with_ref.license_key is foo.license_key # True
foo.foo_with_ref.foo.license_key is foo.license_key # True
foo.license_key == 'ABC123' # True
```
A reference can point to multiple fields using list or tuples:

```python
class SubFooWithMultiRef(ConfigState):
    foo1: Foo = ConfigField(type=Foo)
    foo2: Foo = ConfigField(type=Foo)
    license_key = ConfigField([foo1.license_key, foo2.license_key]) # Reference to foo1 and foo2's license_key field

# Now instead of:
conf = {'foo1': {'license_key': 'ABC123'}, 'foo2': {'license_key': 'ABC123'}}
SubFooWithMultiRef(conf)

# One can simply do:
foo = SubFooWithMultiRef({'license_key': 'ABC123'})

foo.license_key == 'ABC123' # True
foo.foo1.license_key is foo.license_key # True
foo.foo2.license_key is foo.license_key # True
```

## Plugins management

A `ConfigState` class can be decorated with `@builder`, this registers the class as a *builder*, this allows its sub classes to be decorated with `@register`, that registers them as plugins and enable their instantiation using the builder parent.

```python
from config_state import builder
from config_state import register

@builder
class ColoredFoo(ConfigState):
    color: str = ConfigField(None, "Color", static=True)
    value: int = ConfigField(type=int, doc="Value")

@register
class RedFoo(ColoredFoo):
    color: str = ConfigField("Red", "Color", static=True)

@register
class BlueFoo(ColoredFoo):
    color: str = ConfigField("Blue", "Color", static=True)

colored_foo = ColoredFoo({'class': 'BlueFoo', 'value': 1})

print(type(colored_foo)) # <class '__main__.BlueFoo'>
print(colored_foo.color) # Blue
print(colored_foo.value) # 1
```

Builders can be defined in a hierarchy. For instance, we can define a *master builder* from which every *builder* can inherit. Building an object is made by specifying the hierarchy path:

```python
@builder
class MasterBuilder(ConfigState):
    pass

@builder
@register
class ColoredFoo(MasterBuilder):
   pass

colored_foo = MasterBuilder({'class': 'ColoredFoo.BlueFoo', 'value': 1})
```