import pytest

from config_state import Serializer
from tests.objects import Foo
from tests.objects import JsonableFoo
from tests.objects import SubFoo
from tests.objects import SubFooWithRef
from tests.objects import SubFooWithRef2
from tests.utils import save_load_compare

default_conf = {'license_key': 1234}
conf_ref = {
    'param_ref': 'ref_license',
    'param_ref2': 'ref2_license',
    'date_ref': '2020-02-02 00:00:00'
}
conf_ref2 = {"ref": "ref_param"}


def test_serialize_with_pickle(tmpdir):
  serializer = Serializer({'class': 'Pickle'})
  save_load_compare(serializer, Foo(default_conf), tmpdir)
  save_load_compare(serializer, SubFoo(default_conf), tmpdir)
  save_load_compare(serializer, JsonableFoo(), tmpdir)
  save_load_compare(serializer, SubFooWithRef(conf_ref), tmpdir)
  save_load_compare(serializer, SubFooWithRef2(conf_ref2), tmpdir)


def test_serialize_with_json(tmpdir):
  serializer = Serializer({'class': 'Json'})

  # Not JSON serializable
  with pytest.raises(TypeError):
    save_load_compare(serializer, Foo(default_conf), tmpdir)

  # ndarray Not JSON serializable
  with pytest.raises(TypeError):
    save_load_compare(serializer, SubFoo(default_conf), tmpdir)

  foo = JsonableFoo()
  foo.iteration += 1
  save_load_compare(serializer, foo, tmpdir)
