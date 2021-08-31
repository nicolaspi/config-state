import inspect
from abc import ABCMeta
from copy import copy
from dataclasses import dataclass
from pydoc import locate
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import yaml

from config_state.exceptions import exception_handler

__VERSION__ = 1.0


class StateProperty(property):
  """Use the `@stateproperty` decorator to define a `ConfigState` state
  variable as a property.
  """

  def __getstate__(self):
    return self.fget, self.fset

  def __setstate__(self, state):
    self.__init__(*state)


stateproperty = StateProperty


class _MetaConfigState(ABCMeta):
  """Meta class converting the `ConfigField` attributes of a `ConfigState`
  into immutable properties. Wraps `__init__` to convert the `StateVar`
  attributes into mutable properties.
  """

  class ConfProperty(property):

    def __new__(cls, *args, **kwargs):
      conf_field = kwargs.get('conf_field', None)
      if conf_field is not None \
        and not isinstance(conf_field, RefConfigField) \
        and isinstance(conf_field._type_, _MetaConfigState) \
        and not isinstance(conf_field._value_, ConfigState):

        class TaintedConfProperty(cls):

          def __init__(self, *args, conf_field, **kwargs):
            super().__init__(*args, conf_field=conf_field, **kwargs)
            self.make_refs(conf_field)

        instance = property.__new__(TaintedConfProperty)
      else:
        instance = property.__new__(cls)
      return instance

    def __init__(self, *args, conf_field, **kwargs):
      super().__init__(*args, **kwargs)
      self.conf_field = conf_field

    def make_refs(self, conf_field: 'ConfigField'):
      klass = conf_field._type_
      conf_fields = klass._config_fields_default[klass]
      for k, _ in conf_fields.items():
        setattr(type(self), k, getattr(type(conf_field), k))

    def __getstate__(self):
      return self.fget, self.fset, self.conf_field,

    def __setstate__(self, state):
      property.__init__(self, *state[:2])
      self.conf_field = state[2]

  class StateProperty(StateProperty):
    pass

  @staticmethod
  def _get_attr_type(cls, type, item_func=lambda k, v: (k, v)):
    attrs = inspect.getmembers(cls, predicate=lambda m: isinstance(m, type))
    return [item_func(k, v) for k, v in attrs]

  @staticmethod
  def build_get(j, dict_attr: str, is_conf_field=False):

    def _get(_self):
      return getattr(_self, dict_attr)[j]._value_

    def _get_conf(_self):
      attr = getattr(_self, dict_attr)[j]
      if isinstance(attr, _DeferredConf):
        if attr._value_.value is not Ellipsis:
          # the conf have been updated, we update the field
          _self._update_conf(j, attr._value_.value)
          return getattr(_self, j)

      return attr._value_

    return _get if not is_conf_field else _get_conf

  @staticmethod
  def build_set(j, dict_attr: str, is_conf_field=False):

    def _set(_self, value):
      var = getattr(_self, dict_attr)[j]
      var._value_ = value
      var._type_ = type(value)

    def _set_conf(_self: ConfigState, value):
      var = getattr(_self, dict_attr)[j]
      if isinstance(var, _DeferredConf):
        # A DeferredConf stores a mutable reference to allow update of all
        # potential ConfField that are references to this DeferredConf
        var._value_.value = value
        _self._update_conf(j, value)
      else:
        raise AttributeError("Updating a conf field is forbidden")

    return _set if not is_conf_field else _set_conf

  def _build_config_field_props(cls):
    # We create and set the properties for the conf.
    for k, v in cls._config_fields_default[cls].items():
      conf_prop = _MetaConfigState.ConfProperty(
          _MetaConfigState.build_get(k, '_config_fields', is_conf_field=True),
          _MetaConfigState.build_set(k, '_config_fields', is_conf_field=True),
          doc=v._doc_,
          conf_field=v)
      setattr(cls, k, conf_prop)

  def _clone_config_fields_default(cls):
    copy_config_fields = {}
    for k, v in cls._config_fields_default[cls].items():
      if isinstance(v, _DeferredConf):
        copy_config_fields[k] = ConfigField(copy(v._value_),
                                            v._doc_,
                                            type=v._type_,
                                            force_type=v._force_type_,
                                            factory=v._factory_,
                                            required=v._mandatory_)
      else:
        copy_config_fields[k] = v

    return copy_config_fields

  def __init__(cls, name, bases, attrs):
    """The `ConfigField` and `StateVar` attributes are converted
    into properties.
    """
    super().__init__(name, bases, attrs)
    cls.__id__ = hash('.'.join([cls.__module__, cls.__name__]))

    cls._config_fields_default = getattr(cls, "_config_fields_default", {})
    conf_fields_default = {}
    for base in bases:
      conf_fields_default.update(cls._config_fields_default.get(base, {}))
    conf_fields_default.update(
        dict([(k, v)
              for k, v in cls.__dict__.items()
              if isinstance(v, ConfigField)]))
    cls._config_fields_default[cls] = conf_fields_default

    _references_conf_fields = {}
    conf_fields_ids_to_key = dict([
        (id(v), k) for k, v in conf_fields_default.items()
    ])
    for k, v in conf_fields_default.items():
      if isinstance(v, RefConfigField):
        v._maybe_complete_path(conf_fields_ids_to_key)
        v._maybe_initialize_ref(cls)
        _references_conf_fields[k] = v

    state_vars_props = _MetaConfigState._get_attr_type(cls, StateProperty)
    missing_fset = set([k for k, v in state_vars_props if v.fset is None])
    if missing_fset:
      raise TypeError(f"{cls} has state variables properties "
                      f"{missing_fset} that have no defined setter.")

    cls._build_config_field_props()

    # resolve __init__:
    if "__init__" not in cls.__dict__:
      qual_bases = [
          base for base in bases if isinstance(base, _MetaConfigState)
      ]

      def __core_init__(self, config=None):
        for base in qual_bases:
          base.__init__(self, config=config)
    else:
      __core_init__ = cls.__dict__["__init__"]

    signature = inspect.signature(__core_init__)
    if len(signature.parameters) != 2 and "config" not in signature.parameters:
      raise TypeError(f"The constructor of {cls} should take one "
                      f"`config` parameter")

    @exception_handler
    def __init__(self, config: Optional[Dict] = None, *args, **kwargs):
      if config is not None:
        config = copy(config)
        # replace Ellipsis with DeferredConf
        for k, conf_item in config.items():
          # we containerize the Ellipsis in order to propagate its update to
          # all its references.
          if conf_item is Ellipsis or conf_item == '...':
            config[k] = DeferredConf()

      if not hasattr(self, "_config_fields"):
        self._config_fields = {}
        conf_fields_default = self._config_fields_default[type(self)]
        self._config_fields = _MetaConfigState._clone_config_fields_default(
            type(self))

        # We resolve special case where config fields are MetaConfigState
        with _ReferenceContext(config, _references_conf_fields,
                               self) as ref_context:
          for k, v in conf_fields_default.items():
            # if the type is a ConfigState and the value is a dict,
            # we interpret it as a config and instantiate a ConfigState
            if isinstance(v._type_, _MetaConfigState) and not isinstance(
                v._value_, ConfigState):

              if isinstance(config, dict) and k in config:
                conf = config[k]
              elif isinstance(v._value_, dict):
                conf = v._value_
              elif isinstance(config, dict):
                conf = {}
              else:
                conf = None

              if isinstance(conf, v._type_):
                value = conf
              else:
                exist = False
                if isinstance(v, RefConfigField):
                  exist, value = ref_context.check_for_updated_reference(k)
                  if exist and len(v._paths_) > 1 and k not in config:
                    ref_context.raise_multi_paths_error(k, v._paths_)
                if not exist:
                  value = ref_context.build_type(conf, v)

              resolved_conf_field = ConfigField(value,
                                                v._doc_,
                                                type=v._type_,
                                                static=v._static_,
                                                required=v._mandatory_)
              self._config_fields[k] = resolved_conf_field

      if "___props_initialized___" not in cls.__dict__:
        __core_init__(self, config=config, *args, **kwargs)

        self._state_vars = getattr(self, "_state_vars", {})
        self._state_vars.update(
            dict(_MetaConfigState._get_attr_type(self, StateVar)))
        # Build and initialize StateVar properties
        for k, v in self._state_vars.items():
          if not hasattr(cls, k):
            state_var_prop = _MetaConfigState.StateProperty(
                _MetaConfigState.build_get(k, '_state_vars'),
                _MetaConfigState.build_set(k, '_state_vars'),
                doc=v._doc_)
            setattr(cls, k, state_var_prop)
          else:
            if isinstance(v._value_, StateVar):
              self._state_vars[k] = v._value_

        cls.___props_initialized___ = True
      else:
        # StateVar properties are already initialized, they need to
        # be reprocessed as they have been unmangled in the `init`
        self._state_vars = getattr(self, "_state_vars", {})
        self._state_vars.update(
            dict(
                _MetaConfigState._get_attr_type(cls,
                                                _MetaConfigState.StateProperty,
                                                lambda j, _: (j, StateVar()))))
        __core_init__(self, config=config, *args, **kwargs)
        for k, v in self._state_vars.items():
          if isinstance(v._value_, StateVar):
            self._state_vars[k] = v._value_

    cls.__init__ = __init__

  def __hash__(cls):
    return cls.__id__

  def __eq__(cls, other):
    return hash(cls) == hash(other)


class ConfigField:
  """Defines a config attribute within a `ConfigState` class

  Attributes:
      value (Any): Default value or the builder used
      to instanciate the field, if it is a `MetaConfigState` the field must
      be configured with a dict that is used to instantiate the corresponding
      `ConfigState`
      description (str): Documentation string
      type (type): Type of the field, inferred from the type of `value` if
      `None`
      force_type (bool): If `type` is not `None`, forces updates to be of
      the same type as `type`, otherwise forces them to be of same type
      as `value`
      static (bool): If `True` the default value is used and can't be overridden
      required (bool): If `True` a value must be provided uppon
      configuration and the default `value` should be `None`
  """

  def __init__(self,
               value: Any = None,
               doc: str = '',
               type: type = None,
               force_type: bool = False,
               static: bool = False,
               required: bool = False,
               factory: Callable = None):
    self._value_ = value
    self._doc_ = doc
    self._type_ = type
    self._force_type_ = force_type  # Force config override to be of the
    # same type
    self._static_ = static  # Prevent default conf to be updated
    self._mandatory_ = required
    self._factory_ = factory  # factory invoked if value don't  match type
    self.__post_init__()

  def __post_init__(self):
    if self._value_ is not None and self._type_ is None:
      self._type_ = type(self._value_)

    if self._type_ is not None and not isinstance(self._type_, type):
      raise AttributeError(f"{type(self._type_).__name__} is not a `type`")

    if not isinstance(
        self._type_,
        _MetaConfigState) and self._value_ is not None and not isinstance(
            self._value_, self._type_):
      try:
        if self._factory_ is not None:
          value = self._factory_(self._value_)
        elif self._force_type_:
          raise TypeError()
        else:
          value = self._type_(self._value_)
      except:
        raise TypeError(f"Value `{self._value_}` of type "
                        f"`{type(self._value_)}` is not compatible with "
                        f"specified type `{self._type_.__name__}`")
      self._value_ = value

    self._maybe_create_reference()

  def __new__(cls, *args, **kwargs):

    def get_param(pos, key):
      value = None
      if len(args) > pos:
        value = args[pos]
      elif key in kwargs:
        value = kwargs[key]
      return value

    value = get_param(0, 'value')
    _type = get_param(2, 'type') or type(value)
    klass = cls

    if isinstance(value, DeferredConf) or value is Ellipsis or '...' == value:
      klass = _DeferredConf

    if isinstance(_type, _MetaConfigState):

      class TaintedConfigField(klass):
        pass

      TaintedConfigField.__name__ = _type.__name__ + klass.__name__
      klass = TaintedConfigField
    elif isinstance(value,
                    (_Ref, RefConfigField, _MetaConfigState.ConfProperty)):
      klass = RefConfigField
    elif isinstance(value, (list, tuple)):
      if value and all([
          isinstance(e, (_Ref, RefConfigField, _MetaConfigState.ConfProperty))
          for e in value
      ]):
        klass = RefConfigField

    return object.__new__(klass)

  def _maybe_create_reference(self):
    if not isinstance(self._type_, _MetaConfigState) \
      or isinstance(self._value_, ConfigState):
      # We don't create a reference if the ConfigState is already
      # initialized
      return

    # todo: optimize the ref building with a registry
    default_conf = self._type_._config_fields_default[self._type_]

    def build_fget(id, key, ref):

      def _get_attr(_):
        DynamicRef.maybe_save_path()
        DynamicRef.maybe_start_new_path(id, key, ref.ref)
        return ref

      return _get_attr

    def make_class(key):

      class TaintedDynamicRef(DynamicRef):

        def __init__(self, ref: ConfigField):
          super().__init__(ref)
          DynamicRef._should_save_paths_and_start_new_paths = False
          [
              self._set_prop(k)
              for k in dir(ref)
              if isinstance(getattr(ref, k), DynamicRef)
          ]
          DynamicRef._should_save_paths_and_start_new_paths = True

      TaintedDynamicRef.__name__ = f"{DynamicRef.__name__}({key})"
      return TaintedDynamicRef

    for k, v in default_conf.items():
      klass = make_class(k)
      prop = property(build_fget(id(self), k, klass(v)))
      setattr(type(self), k, prop)

  def __getstate__(self):
    _type = None if self._type_ is None else '.'.join(
        [self._type_.__module__, self._type_.__name__])
    return FrozenPortableField(self._value_, self._doc_, _type)

  def __setstate__(self, state):
    type = state.type and locate(state.type) or None
    self.__init__(state.value, state.doc, type)


@dataclass(frozen=True)
class FrozenPortableField:
  value: Any = None
  doc: str = ''
  type: str = None


@dataclass
class PortableField:
  value: Any = None
  doc: str = ''
  type: str = None


class _Ref:

  def __init__(self, ref: ConfigField):
    assert isinstance(ref, ConfigField)
    self.ref = ref
    self.static_path = None
    self.static_path_id = None


class StaticRef(_Ref):

  def __init__(self, ref, path, path_id):
    super().__init__(ref)
    self.static_path = path
    self.static_path_id = path_id


class DynamicRef(_Ref):
  """Represents a reference to a `ConfigField`
  """
  _path_id_ = None
  _path_ = None
  _paths_: List = []
  _paths_id_: List = []
  _should_save_paths_and_start_new_paths = True
  do_not_trace = False

  def __init__(self, ref: ConfigField):
    super().__init__(ref)

  @classmethod
  def maybe_save_path(cls):
    if DynamicRef.do_not_trace:
      return

    # Are we resolving a ref already ? If not, it is a new ref
    if DynamicRef._should_save_paths_and_start_new_paths:
      # save path of previous Ref if exists
      if cls._path_ is not None:
        path_id = tuple(cls._path_id_)
        cls._paths_.append(copy(cls._path_))
        cls._paths_id_.append(path_id)
        cls._path_ = None
        cls._path_id_ = None

  @classmethod
  def maybe_start_new_path(cls, begin_id, key, ref):
    if DynamicRef.do_not_trace:
      return

    # Are we resolving a ref already ? If not, it is a new ref
    if DynamicRef._should_save_paths_and_start_new_paths:
      DynamicRef._path_ = [begin_id, key]
      DynamicRef._path_id_ = [begin_id, id(ref)]

  def _set_prop(self, key):
    setattr(type(self), key, property(lambda _self: _self._get_attr(key)))

  def _get_attr(self, key):
    if DynamicRef.do_not_trace:
      return getattr(self.ref, key)

    DynamicRef._path_.append(key)
    DynamicRef._should_save_paths_and_start_new_paths = False
    ref = getattr(self.ref, key)
    DynamicRef._should_save_paths_and_start_new_paths = True
    DynamicRef._path_id_.append(id(ref.ref))
    return ref


def reference(path: Union[str, List[str]]):
  ref_conf = object.__new__(RefConfigField)
  ref_conf.__init__(None)
  if isinstance(path, str):
    paths = [path]
  else:
    paths = path

  ref_conf._paths_ = [path.split('.') for path in paths]
  return ref_conf


'''def link(ref_from: _Ref, ref_to: Union[_Ref, List[_Ref]]):
    name = inspect.stack()[1].frame.f_locals['__qualname__']
    links = MetaConfigState.__link_registry__.get(name, {})
    links_to = links.get(ref_from, [])

    if isinstance(ref_to, List):
        refs = [RefConfigField(ref) for ref in ref_to]
        links_to.extend(refs)
    else:
        links_to.append(RefConfigField(ref_to))

    links[ref_from] = links_to
    MetaConfigState.__link_registry__[name] = links'''


class RefConfigField(ConfigField):
  """Represents a reference to one or many `ConfigField`
  """
  _paths_: List = None
  _paths_id_: List = None

  def __post_init__(self):
    if self._value_ is None:
      return

    super().__post_init__()
    if not isinstance(self._value_, (list, tuple)):
      mangled_refs = [self._value_]
    else:
      mangled_refs = self._value_

    mangled_refs.reverse()

    # save path of the last resolved path
    DynamicRef.maybe_save_path()
    paths = []
    paths_id = []
    set_ids = set([])
    unmangled_refs = []
    for ref in mangled_refs:
      if isinstance(ref, _MetaConfigState.ConfProperty):
        ref = ref.conf_field

      if isinstance(ref, RefConfigField):
        for _ref, path, path_id in zip(ref._value_, ref._paths_,
                                       ref._paths_id_):
          if path_id not in set_ids:
            paths.append(path)
            paths_id.append(path_id)
            set_ids.add(path_id)
            unmangled_refs.append(_ref)
      else:
        if isinstance(ref, ConfigField):
          path_id = (id(ref),)
          path = [id(ref)]
          ref = StaticRef(ref, path, path_id)
        elif isinstance(ref, StaticRef):
          path_id = ref.static_path_id
          path = ref.static_path
        else:
          path_id = DynamicRef._paths_id_.pop()
          path = DynamicRef._paths_.pop()

        if path_id not in set_ids:
          paths.append(path)
          paths_id.append(path_id)
          set_ids.add(path_id)
          unmangled_refs.append(ref)

    is_all_refs = all([isinstance(e, _Ref) for e in unmangled_refs])
    if not is_all_refs:
      raise TypeError(f"type {self._value_} is not a {DynamicRef} or a "
                      f"list of {DynamicRef}")

    type = unmangled_refs[0].ref._type_
    is_all_ref_same_type = all([e.ref._type_ == type for e in unmangled_refs])
    if not is_all_ref_same_type:
      raise TypeError(f"Refs should have the same type, but are "
                      f"{[e.ref._type_ for e in unmangled_refs]}")

    factory = unmangled_refs[0].ref._factory_
    is_all_ref_same_factory = all(
        [e.ref._factory_ is factory for e in unmangled_refs])

    if not is_all_ref_same_factory:
      raise TypeError(f"Refs should have the same factory")

    self._paths_ = paths
    self._paths_id_ = paths_id

    required = any([ref.ref._mandatory_ for ref in unmangled_refs])
    self._mandatory_ = required
    self._type_ = type
    self._value_ = unmangled_refs
    self._factory_ = factory
    assert not DynamicRef._paths_

  def __getstate__(self):
    return self._paths_

  def __setstate__(self, state):
    self.__init__()
    self._paths_ = state

  def _is_path_complete(self, path):
    return isinstance(path[0], str)

  def _maybe_complete_path(self, conf_fields_ids_to_key: dict):
    for path in self._paths_:
      if not self._is_path_complete(path):
        if path[0] in conf_fields_ids_to_key:
          path[0] = conf_fields_ids_to_key[path[0]]

  def _maybe_initialize_ref(self, klass: _MetaConfigState):
    if self._paths_id_ is None:
      refs = []
      DynamicRef.do_not_trace = True
      for path in self._paths_:
        obj = klass
        ids = []
        for key in path:
          obj = getattr(obj, key)
          if isinstance(obj, _Ref):
            ids.append(id(obj.ref))
          elif isinstance(obj, _MetaConfigState.ConfProperty):
            ids.append(id(obj.conf_field))
          else:
            ids.append(id(obj))

        if isinstance(obj, _MetaConfigState.ConfProperty):
          obj = obj.conf_field
        elif isinstance(obj, _Ref):
          obj = obj.ref

        if isinstance(obj, (RefConfigField)):
          refs.append(obj)
        else:
          ref = StaticRef(obj, path, tuple(ids))
          refs.append(ref)
      DynamicRef.do_not_trace = False
      self._value_ = refs
      self.__post_init__()


class _DeferredConf(ConfigField):
  """Represents a config field whose initialization has been deferred
  """

  def __post_init__(self):
    value = self._value_
    self._value_ = None
    super().__post_init__()
    if isinstance(value, DeferredConf):
      self._value_ = value
    else:
      self._value_ = DeferredConf()
    self._type_ = None


@dataclass
class DeferredConf(object):
  """ Represent a config value that has been initialized with an Ellipsis.
  """
  value: Any = Ellipsis


class _ReferenceContext(object):
  references_to_init = set([])
  references_context = []
  _ids_path = []
  _path = []
  updated_refs = {}

  def __init__(self, config: Optional[dict], references: Dict,
               config_state: 'ConfigState'):
    self._config = config
    self._default_config = config_state._config_fields_default[type(
        config_state)]
    self._references = {}
    self._config_state = config_state
    if _ReferenceContext.references_context:
      _references = _ReferenceContext.references_context[-1]
    else:
      _references = {}
      _ReferenceContext.references_context.append(_references)

    for k, ref in references.items():
      for path_id, path in zip(ref._paths_id_, ref._paths_):
        full_path_id = tuple(_ReferenceContext._ids_path + list(path_id))
        self._references[k] = full_path_id, path
        _ReferenceContext.references_to_init.add(path_id)
        if k in config:
          conf_item = config[k]
          _ReferenceContext.populate_ref_dict(_references, path_id, conf_item)

  @staticmethod
  def populate_ref_dict(ref_dict, id_path, conf_item):
    id = id_path[0]
    refs = ref_dict.get(id, {})
    ref_dict[id] = refs
    if len(id_path) == 1:
      refs['conf'] = conf_item
    else:
      _ReferenceContext.populate_ref_dict(refs, id_path[1:], conf_item)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, value, traceback):
    if exc_type is not None:
      return False

    to_delete_keys = []
    for k, (path, _) in self._references.items():
      if path in _ReferenceContext.updated_refs:
        val = _ReferenceContext.updated_refs[path]
        self.update_config(k, val)
        to_delete_keys.append(k)

    for k in to_delete_keys:
      del self._references[k]

    for k, (_, path) in self._references.items():
      attr = self._config_state
      for key in path:
        attr = getattr(attr, key)
      self.update_config(k, attr)

    if not _ReferenceContext._ids_path:
      _ReferenceContext.reset()

    return True

  def raise_multi_paths_error(self, key, paths):
    pretty_paths = ['.'.join(path) for path in paths]
    self.reset()
    raise ValueError(f"The reference field '{key}' with "
                     f"multiple paths {pretty_paths} has no "
                     f"precedence on the configuration "
                     f"and can not guaranty equality of "
                     f"the references.")

  def update_config(self, k, val):
    default_ref = self._default_config[k]
    if k not in self._config and len(default_ref._paths_) > 1:
      self.raise_multi_paths_error(k, default_ref._paths_)
    self._config[k] = val

  @staticmethod
  def reset():
    _ReferenceContext.references_to_init = set([])
    _ReferenceContext.references_context = []
    _ReferenceContext._ids_path = []
    _ReferenceContext.updated_refs = {}

  def check_for_updated_reference(self, key):
    if key in self._references:
      path_id, _ = self._references[key]
      if path_id in _ReferenceContext.updated_refs:
        del self._references[key]
        return True, _ReferenceContext.updated_refs[path_id]
    return False, None

  def build_type(self, config: Dict, conf_field: ConfigField) -> 'ConfigState':
    type = conf_field._type_
    if len(_ReferenceContext.references_to_init) > 0:
      _id = id(conf_field)
      _ReferenceContext._ids_path.append(_id)

      refs = _ReferenceContext.references_context[-1].get(_id, {})
      _ReferenceContext.references_context.append(refs)

      fields = dict([
          (id(v), k) for k, v in type._config_fields_default[type].items()
      ])

      for _id, references in refs.items():
        if _id in fields and 'conf' in references:
          conf = references['conf']
          config[fields[_id]] = conf

      ret_value = type(config)
      id_path = _ReferenceContext._ids_path + [0]
      for id_end, k in fields.items():
        id_path[-1] = id_end
        _id_path = tuple(id_path)
        if _id_path in _ReferenceContext.references_to_init:
          _ReferenceContext.updated_refs[_id_path] = getattr(ret_value, k)
          _ReferenceContext.references_to_init.remove(_id_path)

      _ReferenceContext._ids_path.pop()
      _ReferenceContext.references_context.pop()
    else:
      ret_value = type(config)

    return ret_value


class StateVar:
  """Defines a state attribute within a `ConfigState` class

  Attributes:
      value (Any): The default value
      description (str): Documentation string
  """

  def __init__(self, value: Any = None, doc: str = None, _type: type = None):
    self._value_ = value
    self._doc_ = doc
    if _type is None:
      self._type_ = type(value)
    else:
      if not isinstance(value, _type):
        raise TypeError(f"Type mismatch {type(value)} and {_type}")
      self._type_ = _type

  def __getstate__(self):
    _type = None if self._type_ is None else '.'.join(
        [self._type_.__module__, self._type_.__name__])
    return PortableField(self._value_, self._doc_, _type)

  def __setstate__(self, state):
    self.__init__(state.value, state.doc, state.type)


@dataclass
class ObjectState:
  """This class represents a serializable representation of a `ConfigState` and
  its state.
  Attributes:
      version (Any): Version data for compatibility checking
      type (Any): the represented `ConfigState` type
      config (dict): A dict of `FrozenPortableField` representing the conf
      internal_state (dict): A dict of `PortableField` representing the state
  """
  version = __VERSION__
  type: Any = None
  config: Dict[str, FrozenPortableField] = None
  internal_state: Dict[str, PortableField] = None


class ConfigState(metaclass=_MetaConfigState):
  """Parent class for objects implementing the `ConfigState` pattern.

  A `ConfigState` defines a pattern aiming to represent python classes with
  two distinctive set of attributes: a set of immutable configuration values
  and a set of mutable state values.

  The configuration is set upon initialization and is passed through the
  constructor. Once initialized, the configuration is frozen and cannot
  change.

  The state variables constitute the mutable state of the instance and can be
  updated throughout its lifetime.

  The configuration and state variables are meant to represent the necessary and
  sufficient information required to clone the object's instance.
  They can be used to save and restore the object from disk.

  - The configuration fields are defined using `ConfigField` class
    attributes. They can have typing constraints and be provided with a factory
    method for building complex types out of simpler/built-in ones.

  - State variables are defined using `StateVar` attributes within the
    constructor. They can alternatively be defined as class properties using
    `@statepropercy` if random logic execution is needed upon
    accession/modification.

  Implementing a class using the `ConfigState` pattern offers the following
  benefits:
    - Provides clear semantic separation between the static configuration
      values and the mutable state variables.
    - Configuration values and state variables are accessible through pythonic
      syntax and benefit from the IDE's type hinting feature.
    - Using a configuration file, one can instantiate a complex hierarchy of
      python classes. A config field may be another `ConfigState` object
      allowing to define tree-like structured `ConfigState` hierarchies.
    - A config field can be a reference to a nested `ConfigState` object's
      config field. This allows coupling between config fields. For
      example, configuration of a log folder path can be injected into the
      nested `ConfigState` objects through the configuration of the topmost
      `ConfigState` object.
    - `ConfigState` objects can be serialized/deserialized into/from a
      stream. They are *pickleable* and in some cases *jsonable*.

  """
  __VERSION__ = __VERSION__  # ConfigState protocol's version

  def __init__(self, config: Optional[dict] = None):
    if config is None:
      config = {}
    if not isinstance(config, dict):
      raise TypeError("`config` must be a dict")

    conf_dict = self._config_fields
    mandatory_fields = set([k for k, v in conf_dict.items() if v._mandatory_])
    for k, v in config.items():
      if k in conf_dict:
        if k in mandatory_fields:
          mandatory_fields.remove(k)
        self._update_conf(k, v)
      else:
        raise AttributeError(
            f"Trying to update the conf field '{k}' which has not been defined")

    if mandatory_fields:
      raise AttributeError(f"Those "
                           f"required fields have not been specified "
                           f"{mandatory_fields}")

  @exception_handler
  def _update_conf(self, key, value):
    attr = self._config_fields_default[type(self)][key]
    if attr._static_:
      raise AttributeError(f"Trying to update the static conf '{key}'")

    if isinstance(attr._type_, _MetaConfigState):
      # This case has been handled in MetaConfigState
      return

    # Infer the type if not provided
    if attr._type_ is None and value is not None:
      _type = type(value)
    else:
      _type = attr._type_

    self._config_fields[key] = ConfigField(value,
                                           attr._doc_,
                                           type=_type,
                                           force_type=attr._force_type_,
                                           factory=attr._factory_)

  def get_state(self) -> ObjectState:
    """Returns the state of the object as a `ObjectState` instance that can be
    serialized.

    Returns:
      ObjectState: the state of the object
    """
    return self.__getstate__()

  def set_state(self, state: ObjectState):
    """
    Set the state of instance with a given `ObjectState`.

    Args:
      state (ObjectState): the state to update the instance's state with

    Returns:
      ConfigState: self
    """
    self.__setstate__(state)

  def __getstate__(self) -> ObjectState:
    self.check_validity()
    conf_fields = dict([(k, v.__getstate__())
                        for k, v in self._config_fields.items()
                        if not v._static_])

    def export_state_var(prop_type, key, object):
      if isinstance(prop_type, _MetaConfigState.StateProperty):
        return object._state_vars[key].__getstate__()
      else:
        value = getattr(object, key)
        state_var = StateVar(value, prop_type.__doc__)
        return state_var.__getstate__()

    state_vars = _MetaConfigState._get_attr_type(
        type(self), StateProperty, lambda k, v:
        (k, export_state_var(v, k, self)))
    state_vars = dict(state_vars)
    obj_state = ObjectState(type(self), conf_fields, state_vars)
    return obj_state

  def __setstate__(self, state: ObjectState):
    if not hasattr(self, "_config_fields"):
      config = {}
      for k, v in state.config.items():
        config[k] = v.value
      self.__init__(config)
    self.check_validity()
    if not isinstance(self, state.type):
      raise TypeError(
          f"Error setting state of type {state.type} to instance of "
          f"type {type(self)}")
    if self.__VERSION__ != state.version:
      raise TypeError(f"ConfigState version mismatch {self.__VERSION__} != "
                      f"{state.version}")

    for k, v in state.config.items():
      type = locate(v.type) if v.type is not None else None
      self._config_fields[k] = ConfigField(v.value, v.doc, type=type)

    for k, v in state.internal_state.items():
      if k in self._state_vars:
        type = locate(v.type) if v.type is not None else None
        self._state_vars[k] = StateVar(v.value, v.doc, type)
      else:
        setattr(self, k, v.value)

    return self

  def check_validity(self):
    """Validate the instance by checking that the `ConfigField` attributes are
    defined as class attributes and `StateVar` attributes have been declared
    within the constructor.

    Returns:
      bool: True is the check tests passes.
    """
    valid_config_keys = set(self._config_fields.keys())
    all_config_keys = set(
        _MetaConfigState._get_attr_type(self, ConfigField, lambda k, _: k))
    diff_keys = all_config_keys - valid_config_keys
    if diff_keys:
      raise SyntaxError(
          f"`{self}` has config fields {diff_keys} that haven't been "
          f"defined as class attributes.")

    valid_vars_keys = set(self._state_vars.keys())
    all_vars_keys = set(
        _MetaConfigState._get_attr_type(self, StateVar, lambda k, _: k))
    diff_keys = all_vars_keys - valid_vars_keys
    if diff_keys:
      raise SyntaxError(
          f"`{self}` has state variables {diff_keys} that have been "
          f"defined outside its `__init__`.")

  def config_summary(self):
    """Returns a string representing the configuration values of the instance.
    If the instance contain nested `ConfigState` objects, their configuration
    are represented as nested config values.

    Returns:
      str: String representation of the instance's configuration.
    """

    def get_nested_config(object: Any):
      if isinstance(object, ConfigState):
        config = dict([(k, get_nested_config(getattr(object, k)))
                       for k in object._config_fields.keys()])
        return config
      else:
        return str(object)

    config = get_nested_config(self)
    stream = yaml.dump(config, default_flow_style=False)

    return stream.replace("'", '')

  def config_hash(self):
    """Compute the a hash representation of the configuration.

    Returns:
      The sha256 hash representation of the instance's configuration.
    """
    from config_state.misc import make_hash_sha256
    from config_state.misc import make_hashable

    hashes = []
    none_hash = "@!mrandom_string__None__"
    for k, field in self._config_fields.items():
      v = field._value_
      if isinstance(v, ConfigState):
        hashes.append(v.config_hash())
      elif v is None:
        hashes.append(none_hash)
      else:
        hashes.append(make_hashable(v))

    return make_hash_sha256(hashes).rstrip("=")
