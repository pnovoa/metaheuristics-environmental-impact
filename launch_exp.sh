printf '%s\n' '--algo '{de,cmaes}' --func '{sphere,rastrigin,griewank}' --func_dim '{10,30,50}' --run '{1..30}' --gen '500' --output_folder 'my_data | xargs -n 12 -P 1 python3 green_experiment.py

899