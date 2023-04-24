import base64
import hashlib

# https://stackoverflow.com/a/42151923


def make_hash_sha256(o):
  hasher = hashlib.sha256()
  hasher.update(repr(make_hashable(o)).encode())
  return base64.b32encode(hasher.digest()).decode('utf-8')


def make_hashable(o):
  if isinstance(o, (tuple, list)):
    return tuple((make_hashable(e) for e in o))

  elif isinstance(o, dict):
    return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

  elif isinstance(o, (set, frozenset)):
    return tuple(sorted(make_hashable(e) for e in o))

  return o


def make_config_hashable(o):
  o = make_hashable(o)
  from config_state import ConfigState
  if isinstance(o, ConfigState):
    return o.config_hash()
  elif isinstance(o, tuple):
    return tuple((make_config_hashable(e) for e in o))
  return o


def get_state(o):
  """Get the state of a potentially nested object."""
  _get_state = getattr(o, '__getstate__', None)
  if _get_state is not None:
    return _get_state()
  elif isinstance(o, list):
    return [get_state(s) for s in o]
  elif isinstance(o, tuple):
    return tuple([get_state(s) for s in o])
  elif isinstance(o, set):
    return set([get_state(s) for s in o])
  elif isinstance(o, dict):
    copy_dict = {}
    for k, v in o.items():
      copy_dict[k] = get_state(v)
    return copy_dict
  else:
    return o