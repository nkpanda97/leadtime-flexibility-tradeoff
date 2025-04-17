#%% IMPORTS AND PATHS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
import itertools
from pyomo.environ import units as u
from pyomo.environ import *

def remove_infeasibility_transactions(ev_data:pd.DataFrame, delta_t:float):
        """ This function removes the infeasible transactions from the data
        param ev_data: data frame containing the ev data
        type ev_data: pd.DataFrame
        return: data frame containing the feasible transactions
        rtype: pd.DataFrame
        """
        data = ev_data.copy()

        def flag_infeasible(vol, pmax, dur, time_step):
            if (vol/pmax)/time_step>dur:
                return 1
            else:
                return 0
            

        # create a new columns 'infeasibility_flag' and set its value to 0

        data['infeasibility_flag'] = data.apply(lambda x: flag_infeasible(x['VOL'], x['P_MAX'], x['DUR_int_adj'], delta_t), axis=1)
        data['infesibility_flag_dur'] = data.apply(lambda x: 1 if x['DUR_int_adj'] <= 0  else 0, axis=1)
        data_feasible = data[(data['infeasibility_flag'] == 0)]
        data_feasible = data_feasible[(data_feasible['infesibility_flag_dur'] == 0)]

        len_infeasible = len(data) - len(data_feasible)
        #drop the infeasible column
        data_feasible = data_feasible.drop(columns=['infeasibility_flag', 'infesibility_flag_dur'])
        return data_feasible, len_infeasible 

def process_for_optimization(ev_data: pd.DataFrame,
                             year_: int,
                             month_: int,
                             day_: int,
                             delta_t: float,
                             key_dict: dict,
                             max_connection_time: int,
                             all_dates:bool=False):
    """This function processes the data for optimization model
    param ev_data: data frame containing the ev data
    type ev_data: pd.DataFrame
    param year_: year of the data
    type year_: int
    param month_: month of the data
    type month_: int
    param day_: day of the data
    type day_: int
    param delta_t: time step size
    type delta_t: float
    param key_dict: dictionary containing the keys for the data
    type key_dict: dict
    param max_connection_time: maximum connection time
    type max_connection_time: int
    param all_dates: flag to process all dates rather than one day
    type all_dates: bool
    return: processed data frame
    rtype: pd.DataFrame
    """

    if all_dates:
        date_range_to_choose = [pd.Timestamp(year=year_,
                                            month=1,
                                            day = 1),
                                            pd.Timestamp(year=year_,
                                            month=12,
                                            day = 30)]
        ev_data_ = ev_data.loc[ev_data[key_dict['Arrival_time']].dt.date.between(date_range_to_choose[0].date(), date_range_to_choose[-1].date(), inclusive='both')].copy()

    else:
        # Filter out the data by choosing one day back and forth
        date_range_to_choose = [pd.Timestamp(year=year_,
                                            month=month_,
                                            day = day_)-pd.Timedelta(days=1),pd.Timestamp(year=year_,
                                            month=month_,
                                            day = day_),pd.Timestamp(year=year_,
                                            month=month_,
                                            day = day_)+pd.Timedelta(days=1)]

        ev_data_ = ev_data.loc[ev_data[key_dict['Arrival_time']].dt.date.between(date_range_to_choose[0].date(), date_range_to_choose[-1].date(), inclusive='left')].copy()


    # --------------------Intial data processing for optimization model --------------------
    # Calculating the integer time steps for arrival and departure time 
    # Make a new column with UTC arrivat and departure time
    ev_data_['START_UTC'] = ev_data_[key_dict['Arrival_time']].dt.tz_convert('UTC')
    ev_data_['STOP_UTC'] = ev_data_[key_dict['Departure_time']].dt.tz_convert('UTC')
    # Calculate the reference time for integer time steps calculation
    midnight_time = ev_data_['START_UTC'].min().replace(hour=0, minute=0, second=0, microsecond=0)
    start_int = ((ev_data_['START_UTC'] - midnight_time) / pd.Timedelta(hours=delta_t)).apply(lambda x: np.floor(x)).astype('int64')
    stop_int = ((ev_data_['STOP_UTC'] - midnight_time) / pd.Timedelta(hours=delta_t)).apply(lambda x: np.ceil(x)).astype('int64')
    dur_int = stop_int - start_int
    dur_int_adj = dur_int.apply(lambda x: min(x,max_connection_time/delta_t )).astype('int64')
    ev_data_processed = pd.DataFrame(data={'START_UTC_rounded':ev_data_['START_UTC'].dt.floor(f'{delta_t}h'),
                                            'STOP_UTC_rounded': ev_data_['STOP_UTC'].dt.ceil(f'{delta_t}h'),
                                            'START_int': start_int,
                                            'STOP_int': dur_int_adj + start_int,
                                            'VOL': ev_data_[key_dict['Actual energy charged (kWh)']],
                                            'P_MAX': ev_data_[key_dict['Max charging power (kW)']],
                                            'DUR_int': dur_int,
                                            'DUR_int_adj': dur_int_adj,
                                            'Connector_id': ev_data_[key_dict['Connector_id']]})
    
    # Remove the infeasible transactions
    ev_data_processed, len_infeasible = remove_infeasibility_transactions(ev_data_processed, delta_t)
    # Reset the index
    ev_data_processed = ev_data_processed.reset_index(drop=True)
    print(f'## Removed {len_infeasible} infeasible transactions')
    return ev_data_processed

def get_cost_data(cost_data:pd.DataFrame,
                  emisson_data:pd.DataFrame,
                  ev_processed_data:pd.DataFrame):
    """ This function reads the cost and emission data and returns only the set of data that is needed for the optimization model
    param cost_data: Data frame containing the cost data
    type cost_data: pd.DataFrame
    param emisson_data: Data frame containing the emission data
    type emisson_data: pd.DataFrame
    param ev_processed_data: processed data frame
    type ev_processed_data: pd.DataFrame
    return: cost and emission data, with the following columns:
            * 'Day ahead price
                - date: date time in UTC
                - Day ahead price [EUR/MWh]: day ahead price in EUR/kWh
            * 'MEF'
                - date: date time in UTC
                - MEF: emission factor in kgCO2/kWh
    rtype: pd.DataFrame, pd.DataFrame"""
    
    cost_data['date'] = cost_data.index
    # get delta_t from the cost_data index (in hours)
    delta_t = (cost_data['date'].iloc[1] - cost_data['date'].iloc[0]).seconds/3600

    all_datetimes = [
        pd.Timestamp(date, tz='UTC') + timedelta(seconds=delta)
        for date in pd.concat([
            pd.Series(ev_processed_data['START_UTC_rounded'].dt.date.unique()),
            pd.Series(ev_processed_data['START_UTC_rounded'].dt.date.unique()) + timedelta(days=1)
        ]).unique()
        for delta in range(0, 86400, int(timedelta(hours=delta_t).total_seconds()))
    ]

    useful_cost_data = cost_data.loc[cost_data['date'].isin(all_datetimes)]
    useful_cost_data = useful_cost_data.reset_index(drop=True)

    # Just keep the columns that are needed
    useful_cost_data = useful_cost_data[['date', 'Day-ahead Price [EUR/kWh]']]

    emisson_data['date'] = emisson_data.index

    useful_emisson_data = emisson_data.loc[emisson_data['date'].isin(all_datetimes)]
    useful_emisson_data = useful_emisson_data.reset_index(drop=True)
    useful_emisson_data = useful_emisson_data[['date', 'MEF']]

    #Assert the length of the cost and mef data are always equal to (unique_dates in ev_data*24)/delta_t
    assert len(useful_cost_data) == len(ev_processed_data["START_UTC_rounded"].dt.date.unique())*24/delta_t+24, f'Length of cost data is not equal to the desired length, needed {len(ev_processed_data["START_UTC_rounded"].dt.date.unique())*24/delta_t+24} but got {len(useful_cost_data)}'
    assert len(useful_emisson_data) == len(ev_processed_data["START_UTC_rounded"].dt.date.unique())*24/delta_t+24, f'Length of emission data is not equal to the desired length, needed {len(ev_processed_data["START_UTC_rounded"].dt.date.unique())*24/delta_t+24} but got {len(useful_emisson_data)}'

    return useful_cost_data, useful_emisson_data

def create_single_ev_model(single_ev_data, delta_t, key_dict):
    Set_T_length = int(single_ev_data[key_dict['duration_adj']]+1)

    # Create Model
    model = ConcreteModel()
    model.T = Set(ordered=True, initialize=np.arange(Set_T_length))

    # Initialize Parameters
    model.step_size = Param(initialize=delta_t, mutable=True)
    model.vol_ev = Param(initialize=single_ev_data[key_dict['actual_energy_charged_value']])
    model.t_d_ev = Param(initialize=single_ev_data[key_dict['duration_adj']])
    model.p_max_ev = Param(initialize=single_ev_data[key_dict['max_charging_power']])
    model.p_min_ev = Param(initialize=0)
    model.p_min_ev_bi = Param(initialize=-single_ev_data[key_dict['max_charging_power']])
    model.cost = Param(model.T, initialize=[single_ev_data['day_ahead_price'][t] for t in model.T])
    model.emission = Param(model.T, initialize=[single_ev_data['mef'][t] for t in model.T])


    # Variables
    model.p_ch_ev = Var(model.T, within=Reals)
    model.soe_ev = Var(model.T,within=NonNegativeReals, initialize=0)

    # Constraints
    def charging_power_limits(model_, t):
        if t >= model.t_d_ev:
            return model_.p_ch_ev[t] == 0
        else:
            return model.p_min_ev, model_.p_ch_ev[t], model_.p_max_ev
        
    def charging_power_limits_bi(model_, t):
        if t >= model.t_d_ev:
            return model_.p_ch_ev[t] == 0
        else:
            return model.p_min_ev_bi, model_.p_ch_ev[t], model_.p_max_ev

    def final_soe(model_, t):
        if t >= model_.t_d_ev:
            return model_.soe_ev[t] == model_.vol_ev
        return Constraint.Skip

    def soe_update(model_, t):
        if t > 0:
            return model_.soe_ev[t] == model_.soe_ev[t-1,] + model_.step_size * model_.p_ch_ev[t-1]

        return model_.soe_ev[t] == 0

    model.con_charging_power_limits = Constraint(model.T, rule=charging_power_limits)
    model.con_charging_power_limits_bi = Constraint(model.T, rule=charging_power_limits_bi)
    model.con_final_soe = Constraint(model.T,rule=final_soe)
    model.con_soe_update = Constraint(model.T,  rule=soe_update)

    return model

def solve_single_ev_model(model, profiles_to_solve:list):
    opt = SolverFactory('gurobi')
    # Objectives
    objectives = {
        'MEF': Objective(expr=sum(model.p_ch_ev[t] * model.step_size * model.emission[t]  for t in model.T), sense=minimize),
        'Cost': Objective(expr=sum(model.p_ch_ev[t] * model.step_size * model.cost[t]   for t in model.T), sense=minimize),
        'Dumb': Objective(expr=sum(model.soe_ev[t] for t in model.T), sense=maximize),
    }
 

    result = {}
    
    for profile_key in profiles_to_solve:
        if profile_key == 'Dumb':
            model.con_charging_power_limits_bi.deactivate()
            model.con_charging_power_limits.activate()
            model.obj = objectives[profile_key]
            res = opt.solve(model)
            
            if res.solver.status == SolverStatus.ok:
                result[f'profile_{profile_key}_bi_directional_False'] = [model.p_ch_ev[t].value for t in model.T]
                result[f'objective_value_{profile_key}_bi_directional_False'] = model.obj()
            else:
                result[f'profile_{profile_key}_bi_directional_False'] = None
                result[f'objective_value_{profile_key}_bi_directional_False'] = None
            model.del_component(model.obj)
        else:
            for bi in [True, False]:
                if bi:
                    model.con_charging_power_limits_bi.activate()
                    model.con_charging_power_limits.deactivate()
                else:
                    model.con_charging_power_limits_bi.deactivate()
                    model.con_charging_power_limits.activate()
                model.obj = objectives[profile_key]
                res = opt.solve(model)
                
                if res.solver.status == SolverStatus.ok:
                    result[f'profile_{profile_key}_bi_directional_{bi}'] = [model.p_ch_ev[t].value for t in model.T]
                    result[f'objective_value_{profile_key}_bi_directional_{bi}'] = model.obj()
                else:
                    result[f'profile_{profile_key}_bi_directional_{bi}'] = None
                    result[f'objective_value_{profile_key}_bi_directional_{bi}'] = None
                model.del_component(model.obj)
    return result

def create_ev_data_for_bau_profile_generation(ev_data_raw: pd.DataFrame, 
                                              delta_t: float, key_dict: dict, 
                                              max_connection_time: int, cost_df: 
                                              pd.DataFrame, mef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes raw EV data and generates a business-as-usual (BAU) profile with cost and emission data.

    Args:
        ev_data_raw (pd.DataFrame): Raw electric vehicle data.
        delta_t (int): Time step interval in minutes.
        key_dict (dict): Dictionary containing key mappings for data processing.
        max_connection_time (int): Maximum connection time for EVs in hours.
        cost_df (pd.DataFrame): DataFrame containing cost data.
        mef_df (pd.DataFrame): DataFrame containing marginal emission factor (MEF) data.

    Returns:
        pd.DataFrame: Processed EV data with additional columns for day-ahead price and MEF.
    """
    ev_data_processed = process_for_optimization(
                                                ev_data=ev_data_raw,
                                                delta_t=delta_t,
                                                year_=2023,
                                                month_=1,
                                                day_=1,
                                                key_dict=key_dict,
                                                max_connection_time=max_connection_time,
                                                all_dates=True
                                            )
    useful_cost_data, useful_emission_data = get_cost_data(cost_df, mef_df, ev_processed_data=ev_data_processed)
    ev_data_processed['day_ahead_price'] = ev_data_processed.apply(lambda x: useful_cost_data.loc[(useful_cost_data['date'] >= x['START_UTC_rounded']) & (useful_cost_data['date'] <= x['STOP_UTC_rounded']), 'Day-ahead Price [EUR/kWh]'].values, axis=1)
    ev_data_processed['mef'] = ev_data_processed.apply(lambda x: useful_emission_data.loc[(useful_emission_data['date'] >= x['START_UTC_rounded']) & (useful_emission_data['date'] <= x['STOP_UTC_rounded']), 'MEF'].values, axis=1)
    return ev_data_processed

def generate_iter_list(ev_data:pd.DataFrame, 
                       bau_profiles:dict, 
                       iter_date_range:list, cs_categories:list[str], cs_per_cat:dict, 
                       key_dict:dict, delta_t:float, time_horizon:int, to_save_path:str,
                       day_ahead_df:pd.DataFrame, mef_df:pd.DataFrame):
    """
    Generate iteration list for all categories based on the provided EV data and base profiles.
    
    Parameters:
    ev_data (pd.DataFrame): Input EV data.
    bau_profiles (pd.DataFrame): DataFrame containing base profiles.
    iter_date_range (pd.DatetimeIndex): Date range for iteration.
    cs_categories (list): List of categories.
    cs_per_cat (dict): Dictionary mapping category to list of connectors.
    key_dict (dict): Dictionary mapping column names.
    delta_t (int): Time granularity in hours.
    time_horizon (int): Total time horizon.
    to_save_path (str): Path to save the resulting pickle file.
    day_ahead_df (pd.DataFrame): DataFrame containing day-ahead price data.
    mef_df (pd.DataFrame): DataFrame containing marginal emission factor data.

    
    Returns:
    dict: A dictionary containing the processed EV data and aggregated profiles for each category.
    """
    def process_and_generate_profiles(ev_data, bau_profiles, iter_date, key_dict, delta_t, time_horizon, filter_connector):
        """
        Process EV data and generate complete profiles for the specified date range.
        """
        # Generate date range for filtering
        date_range = [
            iter_date - pd.Timedelta(days=1),
            iter_date,
            iter_date + pd.Timedelta(days=1)
        ]

        # Filter EV data for the date range and the given connector type
        ev_data_filtered = ev_data[
            ev_data[key_dict['Arrival_time_UTC_rounded']].dt.date.between(
                date_range[0].date(), date_range[-1].date(), inclusive='left'
            ) & ev_data[key_dict['Connector_id']].isin(filter_connector)
        ].copy()

        # Calculate reference start time (midnight of the earliest arrival time)
        ref_time_start = ev_data_filtered[key_dict['Arrival_time_UTC_rounded']].min().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Calculate arrival and departure time in integer format
        ev_data_filtered[key_dict['arrival_time_integer']] = (
            (ev_data_filtered[key_dict['Arrival_time_UTC_rounded']] - ref_time_start) / pd.Timedelta(hours=delta_t)
        ).apply(np.floor).astype('int64')

        ev_data_filtered[key_dict['departure_time_integer']] = (
            (ev_data_filtered[key_dict['Departure_time_UTC_rounded']] - ref_time_start) / pd.Timedelta(hours=delta_t)
        ).apply(np.ceil).astype('int64')

        ev_data_filtered[key_dict['departure_time_integer_adjusted']] = (
            (ev_data_filtered[key_dict['Departure_time_UTC_rounded']] - ref_time_start) / pd.Timedelta(hours=delta_t)
        ).apply(np.ceil).astype('int64').clip(upper=time_horizon-1)

        def generate_complete_profile(profile_index, t_a):
            profile = bau_profiles.loc[profile_index]

            key_list = [
                'profile_Cost_bi_directional_True',
                'profile_Cost_bi_directional_False',
                'profile_MEF_bi_directional_True',
                'profile_MEF_bi_directional_False',
                'profile_Dumb_bi_directional_False'
            ]

            complete_profile = {
                profile_type: [0] * t_a + profile[profile_type] + [0] * (time_horizon - t_a-len(profile[profile_type]))
                for profile_type in key_list
            }

            for obj in [
                'objective_value_Cost_bi_directional_True',
                'objective_value_Cost_bi_directional_False',
                'objective_value_MEF_bi_directional_True',
                'objective_value_MEF_bi_directional_False',
                'objective_value_Dumb_bi_directional_False'
            ]:
                complete_profile[obj] = profile[obj]

            return complete_profile

        # Generate EV profiles
        ev_data_filtered['Base_profiles'] = ev_data_filtered.apply(
            lambda x: generate_complete_profile(
                profile_index=x.name,
                t_a=x[key_dict['arrival_time_integer']]
            ),
            axis=1
        )

        # Aggregate profiles
        keys_to_aggregate = ['profile_Cost_bi_directional_True', 'profile_Cost_bi_directional_False', 'profile_MEF_bi_directional_True', 'profile_MEF_bi_directional_False', 'profile_Dumb_bi_directional_False']
        aggeregate_profiles = {}

        for prof_ in keys_to_aggregate:
            aggeregate_profiles[prof_] = sum(np.array([ev_data_filtered['Base_profiles'].iloc[n][prof_] for n in range(len(ev_data_filtered))]))

        # Sanity check: Ensure the sum of individual volumes matches the aggregated profiles
        agg_profile_energy = [sum(aggeregate_profiles[key]) for key in aggeregate_profiles.keys()] + [ev_data_filtered['VOL'].sum()]
        assert np.all(np.isclose(agg_profile_energy, agg_profile_energy[0], atol=1e-6)), "Mismatch between total energy among different aggregate base profiles"

        # Reset index for ev_data_filtered
        ev_data_filtered.reset_index(drop=True, inplace=True)

        day_ahead_price, mef_price = get_cost_data(cost_data=day_ahead_df, emisson_data=mef_df, ev_processed_data=ev_data_filtered)

        return {'individual_data': ev_data_filtered, 'aggeregate_profiles': aggeregate_profiles, 'signals':{'day_ahead_price':day_ahead_price, 'mef':mef_price}}

    # Initialize result container
    result = {cs_: [] for cs_ in cs_categories}

    # Iterate over date range and categories, process the data, and store results
    for iter_date in tqdm(iter_date_range):
        for cat_ in cs_categories:
            cat_list = cs_per_cat[cat_]
            result[cat_].append(process_and_generate_profiles(
                ev_data=ev_data,
                bau_profiles=bau_profiles,
                iter_date=iter_date,
                key_dict=key_dict,
                delta_t=delta_t,
                time_horizon=time_horizon,
                filter_connector=cat_list
            ))

    # Save the results to the specified path
    pd.to_pickle(result, to_save_path)
    print(f"Iter list saved to {to_save_path}")

    return result

def print_model_structure(model):
    """
    Prints a structured summary of model components: Parameters, Variables, and Constraints,
    including the names of the index sets.
    """
    def get_set_names(index_set):
        """Helper function to retrieve the names of index sets."""
        # Check if the set is a tuple (indicating a product set) or a single set
        if isinstance(index_set, tuple):
            # If it's a tuple, concatenate the names of all sets in the tuple
            return ", ".join([s.name if hasattr(s, "name") else "Unnamed Set" for s in index_set])
        else:
            # Single set case
            return index_set.name if hasattr(index_set, "name") else "Unnamed Set"

    print("Model Parameters")
    print("--------------------")
    print(f"{'Name':<20}{'Sets':<30}{'Index'}")
    print("-" * 80)
    for param in model.component_objects(Param, active=True):
        set_names = get_set_names(param.index_set()) if param.is_indexed() else "Scalar"
        index_set = f"[{list(param.index_set())[0]},...,{list(param.index_set())[-1]}]" if param.is_indexed() else "Scalar"
        print(f"{param.name:<20}{set_names:<30}{index_set}")

    print("\nModel Variables")
    print("--------------------")
    print(f"{'Name':<20}{'Sets':<30}{'Index'}")
    print("-" * 80)
    for var in model.component_objects(Var, active=True):
        set_names = get_set_names(var.index_set()) if var.is_indexed() else "Scalar"
        index_set = f"[{list(var.index_set())[0]},...,{list(var.index_set())[-1]}]" if var.is_indexed() else "Scalar"
        print(f"{var.name:<20}{set_names:<30}{index_set}")

    print("\nModel Constraints")
    print("--------------------")
    print(f"{'Name':<20}{'Sets':<30}{'Index'}")
    print("-" * 80)
    for con in model.component_objects(Constraint, active=True):
        set_names = get_set_names(con.index_set()) if con.is_indexed() else "Scalar"
        index_set = f"[{list(con.index_set())[0]},...,{list(con.index_set())[-1]}]" if con.is_indexed() else "Scalar"
        print(f"{con.name:<20}{set_names:<30}{index_set}")

def create_base_flex_model(iter_data:dict,
                              delta_t:float, time_horizon:int,  flex_start:int,
                               flex_end:int,
                              lambda_value=0.000001, keywords=None):
                         
    """
    This function creates and returns an optimization model for EV charging with flexibility constraints.

    param iter_data: A dictionary containing the data for the optimization model for a single iteration step. The dictionary should contain the following
                        keys:
                        - 'individual_data': A pandas DataFrame containing the data for individual EVs
                        - 'aggeregate_profiles':  A dictionary containing the base profiles for the charging stations
                        - 'signals': A dictionary containing the cost and emission signals
    type iter_data: dict
    param delta_t: The time step of the optimization model in hours
    type delta_t: float
    param felx_start: The time step from which the flexibility is required
    type flex_start: int
    param flex_end: The time step till which the flexibility is required
    type flex_end: int
    param time_horizon: The time horizon of the optimization model in hours
    type time_horizon: int
    param lambda_value: The sensitivity parameter for the objective function
    type lambda_value: float
    param keywords: A dictionary containing the keywords for the optimization model
    type keywords: dict

    return: A Pyomo model for the optimization problem
    rtype: ConcreteModel

    """

    ev_data = iter_data['individual_data']
    
    # Define the length of the sets
    Set_T_length = time_horizon
    Set_N_length = len(ev_data)
    
    # Create an empty Pyomo model
    model = ConcreteModel()
    
    # Define sets
    model.N = Set(ordered=True, initialize=np.arange(Set_N_length))
    model.T = Set(ordered=True, initialize=np.arange(Set_T_length))
    
    # Define parameters
    model.n_cs = Param(initialize=len(ev_data['Connector_id'].unique()), mutable=False, doc='Number of charging stations')
    model.vol_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kWh, doc='Maximum volume of charge in kWh', initialize=ev_data[keywords['actual_energy_charged_value']].to_dict())
    model.t_a_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step from which a vehicle is available', initialize=ev_data[keywords['arrival_time_integer']].to_dict())
    model.t_d_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step till which a vehicle is available', initialize=ev_data[keywords['departure_time_integer_adjusted']].to_dict())
    model.p_max_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kW, doc='Maximum charging power in kW', initialize=ev_data[keywords['max_charging_power']].to_dict())
    model.p_min_ev = Param(model.N, within=Reals, mutable=True, units=u.kW, doc='Minimum charging power in kW', initialize=0)
    model.base_profile = Param(model.T, mutable=True, doc='Aggregated profile of EVs as per dumb charging', initialize=0)
    model.step_size = Param(initialize=delta_t, mutable=False, doc='Step size of analysis in hours')
    model.flex_start = Param( mutable=False, doc='Integer timestep from which flexibility is required', initialize=int(24/model.step_size + flex_start/model.step_size))
    model.flex_end = Param(mutable=False, doc='Integer timestep till which flexibility is required', initialize=int(24/model.step_size +flex_end/model.step_size))
    model.lambda_ = Param(initialize=lambda_value, mutable=False, doc='Sensitivity parameter for the objective function')
    model.cost = Param(model.T, within=Reals, mutable=False, units=u.e/u.kWh, doc='Cost of energy at time t per kWh', 
                       initialize=iter_data['signals']['day_ahead_price'][ 'Day-ahead Price [EUR/kWh]'].to_dict())
    model.emission = Param(model.T, within=Reals, mutable=False, units=u.e/u.kWh, doc='Emission factor at time t per kWh',
                           initialize=iter_data['signals']['mef'][ 'MEF'].to_dict())

    # Define variables
    model.p_ch_ev = Var(model.T, model.N, within=Reals, units=u.kW, doc='Charging power of nth EV at time t')
    model.soe_ev = Var(model.T, model.N, within=NonNegativeReals, units=u.kWh, doc='State of charge of nth EV at time t')
    model.aux_var = Var(within=NonNegativeReals, doc='Auxiliary variable used for objective function')


    
    def charging_power_limits(model_, t, n):
        """
        Constraint rule to enforce limits on the charging power limits
        """
        if (t >= model_.t_a_ev[n].value) & (t < model_.t_d_ev[n].value):
            return  model_.p_min_ev[n], model_.p_ch_ev[t, n], model_.p_max_ev[n]
        else:
            return model_.p_ch_ev[t, n] == 0

    def final_soe(model_, t, n):
        """
        This constraint rule makes sure that all  the EVs are charged by the time they depart from the charger.
        """
        if t >= model_.t_d_ev[n].value:
            return model_.soe_ev[t, n] == model_.vol_ev[n]
        else:
            return model_.soe_ev[t, n] <= model_.vol_ev[n]

    def soe_update(model_, t, n):
        """
        This constraint updates the state of energy of the EVs based omn previous state and next step charging power.
        """
        if t <= model_.t_a_ev[n].value:
            return model_.soe_ev[t, n] == 0
        else:
            return model_.soe_ev[t, n] == model_.soe_ev[t-1, n] +  model_.step_size * (model_.p_ch_ev[t-1, n])

    def re_dispatch_down(model_, t):
        if (t >= model_.flex_start.value) & (t < model_.flex_end.value):
            return sum(model_.p_ch_ev[t, n] for n in model_.N) <= model_.base_profile[t] - model_.aux_var
        else:
            return Constraint.Skip
    
    def capacity_limitation(model_, t):
        if (t >= model_.flex_start.value) & (t < model_.flex_end.value):
            return sum(model_.p_ch_ev[t, n] for n in model_.N)<= model_.aux_var
        else:
            return Constraint.Skip
        
    # Add constraints to the model
    model.con_p_ch_limits = Constraint(model.T, model.N, rule=charging_power_limits)
    model.con_final_soe = Constraint(model.T, model.N, rule=final_soe)
    model.con_soe_update = Constraint(model.T, model.N, rule=soe_update)
    model.con_re_dispatch_down = Constraint(model.T, rule=re_dispatch_down)
    model.con_capacity_limitation = Constraint(model.T, rule=capacity_limitation)


    return model

def model_product_builder(model, product_type, base_profile_type):
    """ Based on the product type and base profile type, this function creates the objective function of the model.
    """
    # Expressions for base objectives
    base_expr = {'Cost': sum(sum(model.p_ch_ev[t, n] for n in model.N)*model.step_size*model.cost[t] for t in model.T),
                 'MEF': sum(sum(model.p_ch_ev[t, n] for n in model.N)*model.step_size*model.emission[t] for t in model.T),
                 'Dumb': sum(sum(model.soe_ev[t, n] for n in model.N) for t in model.T)}
    
    if product_type == 're_dispatch_down':
        model.con_re_dispatch_down.activate()
        model.con_capacity_limitation.deactivate()
        if base_profile_type == 'Cost':
            model.obj = Objective(expr=model.aux_var-model.lambda_*base_expr['Cost'], sense=maximize)   
        elif base_profile_type == 'MEF':
            model.obj = Objective(expr=model.aux_var-model.lambda_*base_expr['MEF'], sense=maximize)
        elif base_profile_type == 'Dumb':
            model.obj = Objective(expr=model.aux_var+model.lambda_*base_expr['Dumb'], sense=maximize)
    elif product_type == 'capacity_limitation':
        model.con_re_dispatch_down.deactivate()
        model.con_capacity_limitation.activate()
        if base_profile_type == 'Cost':
            model.obj = Objective(expr=model.aux_var + model.lambda_*base_expr['Cost'], sense=minimize)
        elif base_profile_type == 'MEF':
            model.obj = Objective(expr=model.aux_var + model.lambda_*base_expr['MEF'], sense=minimize)
        elif base_profile_type == 'Dumb':
            model.obj = Objective(expr=model.aux_var - model.lambda_*base_expr['Dumb'], sense=minimize)
    else :
        raise ValueError('Invalid product type, please choose from re_dispatch_down or capacity_limitation')

def update_flex_model_parameters(model, bi_directional:bool, base_profile_type:str, base_profile_dict:dict):
    """ This function updates the parameters of the model based on the flexibility type and base profile type.
    """
    for t in model.T:
        for n in model.N:
            model.p_ch_ev[t, n].free()
            model.p_ch_ev[t, n].clear()


    def _update_profile(model_, profile_):
        for t in model_.T:
            model_.base_profile[t] = profile_[t]
    if bi_directional:
        for n in model.N:
            model.p_min_ev[n] = -model.p_max_ev[n]
    else:
        for n in model.N:
            model.p_min_ev[n] = 0
    _update_profile(model, base_profile_dict[f'profile_{base_profile_type}_bi_directional_False'])

def fix_p_variables(model:ConcreteModel,
                    lead_time:int,
                    ev_df:pd.DataFrame,
                    base_profile_type:str):
    """ This fucntions fixes the power variables till the flex_start-lead time
    param model: Pyomo optimization model
    type model: pyomo.environ.ConcreteModel
    param lead_time: Lead time for fixing the variables
    type lead_time: int
    param ev_df: EV data
    type ev_df: pd.DataFrame
    """
    for n in model.N:
        for t in model.T:

            if t <= model.flex_start.value-lead_time-1:
                model.p_ch_ev[t, n].fix(ev_df['Base_profiles'].loc[n][f'profile_{base_profile_type}_bi_directional_False'][t])
            else:
                model.p_ch_ev[t, n].free()

def solve_flex_model(model:ConcreteModel):
    """
    This function returns all the information in dictionary format for the model.
    """
    solver = SolverFactory('gurobi')
    solver_status = solver.solve(model)
    result = {}
    if solver_status.solver.status == SolverStatus.ok:
        result['base_aggregate_profile_kW'] = list(model.base_profile.extract_values().values())
        result['optimized_aggregate_profile_kW'] =  [sum(model.p_ch_ev[t, n].value for n in model.N) for t in model.T]
        result['flexibility_kW'] = model.aux_var.value
        result['charging_cost_EUR'] = sum(sum(model.p_ch_ev[t, n].value for n in model.N)*model.step_size.value*model.cost[t].value for t in model.T)
        result['number_active_charging_station'] = model.n_cs.value
        result['charging_emission_kgCO2'] = sum(sum(model.p_ch_ev[t, n].value for n in model.N)*model.step_size.value*model.emission[t].value for t in model.T)
        result['status'] = 'Solved'
    else:
        result['base_aggregate_profile_kW'] = []
        result['optimized_aggregate_profile_kW'] = []
        result['flexibility_kW'] = None
        result['charging_cost_EUR'] = None
        result['number_active_charging_station'] = None
        result['charging_emission_kgCO2'] = None
        result['status'] = 'Infeasible'
    
    return result

def single_iter_run(one_iter_data, 
                    product, 
                    bi_directional, 
                    base_profile_type, 
                    lead_time, 
                    f_start, f_end,
                    delta_t, time_horizon, lambda_value, keys_dict,
                    return_model =False, data_index=None):
    """ This function runs a single iteration of the optimization model
    param one_iter_data: A dictionary containing the data for the optimization model for a single iteration step. The dictionary should contain the following
                        keys: 'individual_data', 'aggeregate_profiles', 'signals'
    type one_iter_data: dict
    param product: The type of flexibility product to be optimized
    type product: str
    param bi_directional: A boolean value indicating whether the charging is bi-directional
    type bi_directional: bool
    param base_profile_type: The type of base profile to be used
    type base_profile_type: str
    param lead_time: The lead time for fixing the power variables
    type lead_time: int
    param f_start: The start time of the flexibility requirement
    type f_start: int
    param f_end: The end time of the flexibility requirement
    param return_model: A boolean value indicating whether to return the model
    param delta_t: The time step of the optimization model in hours
    type delta_t: float
    param time_horizon: The time horizon of the optimization model in hours
    type time_horizon: int
    param lambda_value: The sensitivity parameter for the objective function
    type lambda_value: float
    param keys_dict: A dictionary containing the keywords for the optimization model
    type keys_dict: dict
    type return_model: bool
    type f_end_: int
    return: A dictionary containing the results of the optimization model
    rtype: dict
    """
    try:
        model = create_base_flex_model(iter_data=one_iter_data,
                                    delta_t=delta_t, time_horizon=time_horizon, 
                                    lambda_value=lambda_value,  keywords=keys_dict, flex_start=f_start, flex_end=f_end)


        model_product_builder(model, product_type=product, base_profile_type=base_profile_type)
        update_flex_model_parameters(model, bi_directional=bi_directional, base_profile_type=base_profile_type, base_profile_dict=one_iter_data['aggeregate_profiles'])
        if lead_time != 'inf':
            fix_p_variables(model, lead_time=lead_time, ev_df=one_iter_data['individual_data'], base_profile_type=base_profile_type)
        res = solve_flex_model(model)
        res['Index'] = str({'lead_time': lead_time, 'bi_directional': bi_directional, 'f_start': f_start, 'f_end': f_end, 'product': product, 'base_profile': base_profile_type, 'data_index': data_index})
        if return_model:
            return res,model
        else:
            return res
    except Exception as e:
        print(f'Error in running the optimization model: {e}')
        return {}

def generate_combinations(lead_time_list, bi_directional_list, f_start_list, 
                          window_length_list, product_list, base_profile_list, data_index_list):
    """
    Generates all combinations of parameter sets for the given inputs.
    
    Parameters:
        lead_time_list (list): Possible values for lead time.
        bi_directional_list (list): Possible values for bi_directional flag (True/False).
        f_start_list (list): Possible values for the start of flexibility window (f_start).
        window_length_list (list): Possible lengths of the flexibility window.
        product_list (list): Possible product types.
        base_profile_list (list): Possible base profiles.
        data_index_list (list): Possible data indices.

    Returns:
        list: A list of dictionaries, each containing a unique combination of parameters.
    """
    combinations = []
    for lead_time, bi_directional, f_start, window_length, product, base_profile, data_index in itertools.product(
        lead_time_list, bi_directional_list, f_start_list, window_length_list, product_list, base_profile_list, data_index_list
    ):
        f_end = f_start + window_length
        combinations.append({
            'lead_time': lead_time,
            'bi_directional': bi_directional,
            'f_start': f_start,
            'f_end': f_end,
            'product': product,
            'base_profile': base_profile,
            'data_index':data_index
        })
    return combinations
