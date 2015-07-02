.PHONY: all

# Source Files
ENVIRONMENT = progressReportEval
AGENT = marioAgent
SRC_PATH = ./src

# Path to RL-Glue to find RL_interface and related files.
PYTHON_CODEC_PATH = ../external/python-codec/src/
PARAMETER_HOLDER_PATH = ../external/15-rl-competition-2009/system/includes
LOADER_HELPER_PATH = ../external/15-rl-competition-2009/trainers/consoleTrainerPython/src/

all: mario_environment mario_agent

mario_environment:
	echo -e "#! /bin/bash\n  PYTHONPATH=$(PYTHON_CODEC_PATH):$(PARAMETER_HOLDER_PATH):$(LOADER_HELPER_PATH):$(SRC_PATH) python $(SRC_PATH)/$(ENVIRONMENT).py " | cat > mario_environment
	chmod u+x mario_environment

mario_agent:
	echo "#! /bin/bash\n  PYTHONPATH=$(PYTHON_CODEC_PATH) python $(SRC_PATH)/$(AGENT).py " | cat > mario_agent
	chmod u+x mario_agent

clean:
	rm -f mario_environment mario_agent *.pyc $(SRC_PATH)/*.pyc
