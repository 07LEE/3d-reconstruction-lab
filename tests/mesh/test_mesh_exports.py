import importlib.util

import src.mesh


def test_all_matches_the_lazy_export_map():
    assert sorted(src.mesh.__all__) == sorted(src.mesh._EXPORTS)


def test_every_lazy_export_exists_in_its_module():
    for name, module in src.mesh._EXPORTS.items():
        spec = importlib.util.find_spec(f"src.mesh.{module}")
        assert spec is not None and spec.origin is not None
        with open(spec.origin, encoding="utf-8") as f:
            assert f"def {name}(" in f.read(), f"{name} not defined in src/mesh/{module}.py"
