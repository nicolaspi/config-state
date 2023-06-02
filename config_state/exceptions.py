class ConfigError(Exception):
  pass


def exception_handler(func):

  def wrapped(*args, **kwargs):
    self = args[0]
    try:
      func(*args, **kwargs)
    except ConfigError as e:
      raise e from e
    except Exception as e:
      raise ConfigError(f"Configuring '{type(self).__name__}': {e}") from e

  return wrapped
