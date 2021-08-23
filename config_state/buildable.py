from config_state.config_state import ConfigField
from config_state.config_state import ConfigState

reg_attr = '_buildable_types_registry'
config_field_name = 'class'
config_field_name_internal = '__builder_class__'


def builder(cls):
  """A Decorator that add factory logic into a `ConfigState`. A builder can
  produce instance of subclasses that have been decorated with `@register`.
  >>> @builder
  >>> class Factory(ConfigState):
  >>>     ...
  >>>
  >>> @register
  >>> class Product(Factory):
  >>>     ...
  >>>
  >>> product = Factory({'class': 'Product'})
  """
  if reg_attr in cls.__dict__:
    raise SyntaxError(f"Already decorated {cls}")
  if not issubclass(cls, ConfigState):
    raise SyntaxError(f"Only `ConfigState` subclasses can be "
                      f"decorated as "
                      f"Buildable objects")

  setattr(cls, reg_attr, {cls.__name__: cls})
  conf_field = ConfigField(cls.__name__, "Type of the built object")
  config_defaults = cls._config_fields_default[cls]

  config_defaults[config_field_name] = conf_field

  def __new__(kls, *args, **kwargs):
    registry = getattr(kls, reg_attr, {})

    if "config" in kwargs:
      config = kwargs['config']
    elif len(args) > 0:
      config = args[0]
    else:
      config = None

    if config is None or config_field_name not in config:
      return object.__new__(kls)
    else:
      if config_field_name_internal in config:
        types = config[config_field_name_internal]
      else:
        types = config[config_field_name]

    types = types.split(sep='.')
    type = types[0]

    if type not in registry:
      raise ValueError(f"Builder {kls.__name__}: Unknown or unregistered "
                       f"class {type}, make sure the "
                       f"class has been registered using the @"
                       f"{register.__name__}"
                       f" decorator, that it is "
                       f"visible from the builder, and "
                       f"that you are building from "
                       f"the correct base class. Registered classes: "
                       f"{list(registry.keys())}")

    subcls = registry[type]

    if len(types) > 1:
      config[config_field_name_internal] = '.'.join(types[1:])
      return subcls.__new__(subcls, config)

    # todo: find a proper way to call the super.__new__ instead
    instance = object.__new__(subcls)
    if not isinstance(instance, kls):
      # In this case we need to invoke the __init__ ourselves
      instance.__init__(config)

    if config_field_name_internal in config:
      del config[config_field_name_internal]

    return instance

  def __getnewargs__(self):
    return (None,)

  cls.__new__ = __new__

  cls.__getnewargs__ = __getnewargs__
  return cls


def register(cls):
  """
  A Decorator that registers a `ConfigState` so that it can be built using a
  factory that is a base class decorated with `@builder`.
  """
  registry_base = None
  for base in cls.mro()[1:]:
    if hasattr(base, reg_attr):
      registry_base = base
      break

  if registry_base is None:
    raise SyntaxError(f"{cls.__name__} must be a subclass of a @"
                      f"{builder.__name__} "
                      f"decorated class")

  registry = getattr(registry_base, reg_attr, {})

  registry[cls.__name__] = cls
  setattr(registry_base, reg_attr, registry)

  return cls
