WEIGTHS="network_config_agent_0_layers_13_13_13_6_1.txt"
python3 run_car.py -s 1000 -f "$WEIGTHS" -fb
./eval.sh