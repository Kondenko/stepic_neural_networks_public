WEIGTHS="network_config_agent_0_layers_13_17_1.txt"
cp $WEIGTHS $WEIGTHS".bak"
python3 run_car.py -s 2000 -f "$WEIGTHS"