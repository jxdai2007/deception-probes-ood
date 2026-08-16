def test_package_imports():
    import jrp_common
    assert isinstance(jrp_common.__version__, str)
