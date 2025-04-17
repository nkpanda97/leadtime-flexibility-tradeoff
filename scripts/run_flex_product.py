""" This script is used to generate flexibility products for single EVs provided in the EV data."""
#%%
import numpy as np
import pandas as pd
from pyomo.environ import units as u
from pyomo.environ import *
import sys
import os
from itertools import product
from joblib import Parallel, delayed
import datetime
import argparse

# Add the parent directory to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.powertech_helper import *

def main(root_path:str,
         category_cs:str,
        lead_time_list = ['inf'],
         bi_directional_list = [True],
         f_start_list = list(range(24)),
            window_length_list = [1,2,3,4,5,6],
            product_list = ['re_dispatch_down', 'capacity_limitation'],
            base_profiles_to_run = ['Dumb'],
            delta_t = 1,
            lambda_value = 1e-6,
            time_horizon = 3*24,
            key_dict = {'Actual energy charged (kWh)': 'VOL',
            'Arrival_time': 'START',
            'Arrival_time_UTC_rounded':'START_UTC_rounded',
            'Departure_time_UTC_rounded':'STOP_UTC_rounded',
            'Departure_time': 'STOP',
            'Max charging power (kW)' : 'P_MAX',
            'Connector_id': 'Connector_id',
            # --- For optimization model ---
            'actual_energy_charged_value' : 'VOL',
            'arrival_time_integer' : 'START_int',
            'departure_time_integer' : 'STOP_int',
            'departure_time_integer_adjusted' : 'STOP_int_adj',
            'max_charging_power' :  'P_MAX',
            'duration_adj':'DUR_int_adj',
            'arrival_time': 'START_int',
            'departure_time': 'STOP_int',
            'energy_charged': 'VOL',
            'max_charging_power': 'P_MAX'
            }
            ):
    """
    Main function to run the flexibility product generation for iter rows
    """
    print(f'Running for {category_cs}')
    all_iter_list = pd.read_pickle(root_path + 'inputs/iter_list_all_categories_2023.pkl')[category_cs]
    combinations_to_simulate = generate_combinations(lead_time_list=lead_time_list,
                                                        bi_directional_list=bi_directional_list,
                                                        f_start_list=f_start_list,
                                                        window_length_list=window_length_list,
                                                        product_list=product_list,
                                                        base_profile_list=base_profiles_to_run,
                                                        data_index_list=list(range(len(all_iter_list))))
    print('---------------------------------------------------------------------------')
    # Display the result
    print(f"Total combinations: {len(combinations_to_simulate)} are generated to be simulated.")
    for combination in combinations_to_simulate[:5]:  # Print only the first 5 for brevity
        print(combination)
    print('---------------------------------------------------------------------------')

    all_runs = Parallel(n_jobs=-1, verbose=10)(delayed(single_iter_run)(one_iter_data=all_iter_list[item['data_index']],
                                                            product=item['product'],
                                                            bi_directional=item['bi_directional'],
                                                            base_profile_type=item['base_profile'],
                                                            lead_time=item['lead_time'],
                                                            f_start=item['f_start'],
                                                            f_end=item['f_end'],
                                                            delta_t=delta_t,
                                                            lambda_value=lambda_value,
                                                            time_horizon=time_horizon,
                                                            keys_dict=key_dict, data_index=item['data_index']) for item in combinations_to_simulate)

    #remove empty results dict
    all_runs = [x for x in all_runs if x is not None]

    to_save_path = root_path + 'outputs' + '/flexibility_products_results_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") 
    pd.DataFrame(all_runs).to_parquet(to_save_path+'.parquet', compression='snappy')
    print(f"Results saved to {to_save_path}")



if __name__ == "__main__":
    # Define the input parameters
    parser = argparse.ArgumentParser(description='Run the flexibility product generation for EVs')
    parser.add_argument('--root_path', type=str, help='Root path for the input data')
    parser.add_argument('--category_cs', type=str, help='Category of the EVs to be simulated')
    args = parser.parse_args()
    main(args.root_path, args.category_cs) 

    