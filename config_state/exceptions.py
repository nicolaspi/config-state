class ConfigError(Exception):
  pass


def exception_handler(func):

  def wrapped(*args, **kwargs):
    self = args[0]
    try:
      func(*args, **kwargs)
    except ConfigError as e:
      raise e
    except Exception as e:
      raise ConfigError(f"Configuring '{type(self).__name__}'") from e

  return wrapped
