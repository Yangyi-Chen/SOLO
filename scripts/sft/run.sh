export PYTHONPATH='pwd':$PYTHONPATH
accelerate launch --main_process_port 11111  src/scripts/hf_train.py config/SFT.yml  