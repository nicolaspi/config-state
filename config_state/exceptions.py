class ConfigError(Exception):
  pass


def exception_handler(func):

  def wrapped(*args, **kwargs):
    self = args[0]
    try:
      func(*args, **kwargs)
    except Exception as e:
      if not isinstance(e, ConfigError):
        raise ConfigError(f"Configuring '{type(self).__name__}': {e}") from e
      else:
        raise e

  return wrapped
