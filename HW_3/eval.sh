WEIGTHS="network_config_agent_0_layers_13_13_13_6_1.txt"
python3 run_car.py -s 800 --seed 3 -f "$WEIGTHS" -e True
python3 run_car.py -s 800 --seed 13 -f "$WEIGTHS" -e True
python3 run_car.py -s 800 --seed 18 -f "$WEIGTHS" -e True