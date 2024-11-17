default: pytest
##################### TESTS #####################
test_env_setup:
	@pytest
	tests/test.py::TestParams::test_MODEL_TARGET
test:

	pytest tests/test.py::TestParams::test_MODEL_TARGET

# Target to run Streamlit app
run_streamlit:
	streamlit run project/app/app.py --server.port 8501

# Target to run both Streamlit and the HTTP server
run_app:
	# Run the Streamlit app in the background
	make run_streamlit & \
	# Start the HTTP server on port 8000
	cd venv/front && python -m http.server 8000 --bind 127.0.0.1 & \
  echo "\033[1;32m\033[48;5;16m\033[1m\033[3m\033[4m Open your browser and go to: \033[1;34mhttp://127.0.0.1:8000/\033[0m"
