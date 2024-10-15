default: pytest
##################### TESTS #####################
test_env_setup:
	@pytest
	tests/test.py::TestParams::test_MODEL_TARGET
test:

	pytest tests/test.py::TestParams::test_MODEL_TARGET
