#%% IMPORTS AND PATHS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from pyomo.environ import units as u
from pyomo.environ import *
path_cost_data = '/Users/nkpanda97/Desktop/GitHub_Macbook/ev_products_tradeoff/data/day_ahead_price.pkl'
path_emisson_data = '/Users/nkpanda97/Desktop/GitHub_Macbook/ev_products_tradeoff/data/mef.pkl'
ev_data = pd.read_pickle('/Users/nkpanda97/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/PhD_AllFolders/DatabaseCodes/ManuallyDownloadedData/Nico_26_Feb_2024/transactiondata_allstations_powerlog_cleaned_onlywithpowerlog.pkl')

# %% HELPER FUNCTIONS
##########################################################################################################################################################################################
def remove_infeasibility_transactions(ev_data, delta_t):
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
                             max_connection_time: int):
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
    return: processed data frame
    rtype: pd.DataFrame
    """

    # Filter out the data by choosing one day back and forth
    date_range_to_choose = [pd.Timestamp(year=year_,
                                        month=month_,
                                        day = day_)-pd.Timedelta(days=1),pd.Timestamp(year=year_,
                                        month=month_,
                                        day = day_),pd.Timestamp(year=year_,
                                        month=month_,
                                        day = day_)+pd.Timedelta(days=1)]

    ev_data = ev_data.loc[ev_data[key_dict['Arrival_time']].dt.date.between(date_range_to_choose[0].date(), date_range_to_choose[-1].date(), inclusive='left')].copy()


    # --------------------Intial data processing for optimization model --------------------
    # Calculating the integer time steps for arrival and departure time 
    # Make a new column with UTC arrivat and departure time
    ev_data['START_UTC'] = ev_data[key_dict['Arrival_time']].dt.tz_convert('UTC')
    ev_data['STOP_UTC'] = ev_data[key_dict['Departure_time']].dt.tz_convert('UTC')
    # Calculate the reference time for integer time steps calculation
    midnight_time = ev_data['START_UTC'].min().replace(hour=0, minute=0, second=0, microsecond=0)
    start_int = ((ev_data['START_UTC'] - midnight_time) / pd.Timedelta(hours=delta_t)).apply(lambda x: np.floor(x)).astype('int64')
    stop_int = ((ev_data['STOP_UTC'] - midnight_time) / pd.Timedelta(hours=delta_t)).apply(lambda x: np.ceil(x)).astype('int64')
    dur_int = stop_int - start_int
    dur_int_adj = dur_int.apply(lambda x: min(x,max_connection_time/delta_t )).astype('int64')
    ev_data_processed = pd.DataFrame(data={'START_UTC_rounded':ev_data['START_UTC'].dt.floor(f'{delta_t}h'),
                                            'STOP_UTC_rounded': ev_data['STOP_UTC'].dt.ceil(f'{delta_t}h'),
                                            'START_int': start_int,
                                            'STOP_int': dur_int_adj + start_int,
                                            'VOL': ev_data[key_dict['Actual energy charged (kWh)']],
                                            'P_MAX': ev_data[key_dict['Max charging power (kW)']],
                                            'DUR_int': dur_int,
                                            'DUR_int_adj': dur_int_adj,
                                            'Connector_id': ev_data[key_dict['Connector_id']]})
    
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

def bau_profile_generator(ev_data: pd.DataFrame,
                     useful_cost_data: pd.DataFrame,
                     useful_emisson_data: pd.DataFrame,
                     delta_t: float,
                     base_profiles_to_run,time_horizon):
    Set_T_length = int(time_horizon)
    Set_N_length = len(ev_data)
    # ---------------------------------- Creating empty model ----------------------------------
    model = ConcreteModel()
    # print('## A concrete optimization model is created for a total %.2f time steps' % Set_T_length)
    # -------------------------------------- Creating Sets --------------------------------------
    model.N = Set(ordered=True, initialize=np.arange(Set_N_length))
    model.T = Set(ordered=True, initialize=np.arange(Set_T_length))

    # ---------------------------------- Creating parameters ----------------------------------
    model.step_size = Param(initialize=delta_t, mutable=True, doc='Step size in hours')
    model.vol_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kWh,
                            doc='Assumed maximum volume of charge in '
                                'kWh that a vehicle can take or is allowed to '
                                'give')
    model.t_a_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step from which a vheicle '
                                                                                'is available')
    model.t_d_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step till which a '
                                                                                'vheicle is still '
                                                                                'available available')
    model.p_max_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kW, doc='Constraint on max power '
                                                                                            'value at which a vheicle can charge it')

    model.cost = Param(model.T, within=NonNegativeReals, mutable=True, units=u.e/u.kWh, doc='Cost of energy at time t per kWh')

    model.emission = Param(model.T, within=NonNegativeReals, mutable=True, units=u.e/u.kWh, doc='Emission factor at time t per kWh')

    for t in model.T:
        model.cost[t] = useful_cost_data['Day-ahead Price [EUR/kWh]'].iloc[t]
        model.emission[t] = useful_emisson_data['MEF'].iloc[t]
        for n in model.N:
            model.vol_ev[n] = ev_data[key_dict['actual_energy_charged_value']].iloc[n]
            model.t_a_ev[n] = ev_data[key_dict['arrival_time_integer']].iloc[n]
            model.t_d_ev[n] = ev_data[key_dict['departure_time_integer_adjusted']].iloc[n]
            model.p_max_ev[n] = ev_data[key_dict['max_charging_power']].iloc[n]

    # ---------------------------------- Creating Variables----------------------------------
    model.p_ch_ev = Var(model.T, model.N, within=NonNegativeReals, units=u.kW,
                        doc='A Real variable denoting the '
                            'charging power of nth EV at time t')
    model.soe_ev = Var(model.T, model.N, within=NonNegativeReals, units=u.kWh,
                        doc='A NonNegativeReals variable denoting the state '
                            'of charge of nth EV at time t')
    model.p_agg = Var(model.T, within=NonNegativeReals, units=u.kW,
                        doc='A Real variable denoting the aggregated power')

    def charging_power_limits(model_, t, n):
        """
        Constraint rule to enforce limits on the charging power limits
        """
        if (t >= model_.t_a_ev[n].value) & (t < model_.t_d_ev[n].value):
            return  model_.p_ch_ev[t, n]<=model_.p_max_ev[n]
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

    def agg_power(model_, t):
        """
        This constraint rule calculates the aggregate power at each time step.
        """
        return model_.p_agg[t] == sum(model_.p_ch_ev[t, n] for n in model_.N)

    model.con_charging_power_limits = Constraint(model.T, model.N, rule=charging_power_limits)
    model.con_final_soe = Constraint(model.T, model.N, rule=final_soe)
    model.con_soe_update = Constraint(model.T, model.N, rule=soe_update)
    model.con_pagg = Constraint(model.T, rule=agg_power)

    model.obj_cost = Objective(expr=sum(model.p_agg[t]*model.step_size*model.cost[t] for t in model.T), sense=minimize)
    model.obj_emission = Objective(expr=sum(model.p_agg[t]*model.step_size*model.emission[t] for t in model.T), sense=minimize)
    model.obj_dumb = Objective(expr = sum(sum(model.soe_ev[t,n] for n in model.N) for t in model.T), sense=maximize)
    opt = SolverFactory('gurobi')

    result = {}
    for key in base_profiles_to_run:
        if key == 'MEF':
            model.obj_cost.deactivate()
            model.obj_dumb.deactivate()
            model.obj_emission.activate()
            res = opt.solve(model)
            if res.solver.status == SolverStatus.ok:
                result[key] = {'profile': [model.p_agg[t].value for t in model.T],
                                'objective_value': model.obj_emission(),
                                'individual profiles': {n: [model.p_ch_ev[t, n].value for t in model.T] for n in model.N}}
            else:
                result[key] = {'profile': [np.nan for t in model.T],
                                'objective_value': np.nan,
                                'individual profiles': {n: [np.nan for t in model.T] for n in
                                                        model.N}}

            
        elif key == 'Cost':
            model.obj_emission.deactivate()
            model.obj_dumb.deactivate()
            model.obj_cost.activate()
            res = opt.solve(model)
            if res.solver.status == SolverStatus.ok:
                result[key] = {'profile': [model.p_agg[t].value for t in model.T],
                                'objective_value': model.obj_emission(),
                                'individual profiles': {n: [model.p_ch_ev[t, n].value for t in model.T] for n in model.N}}
                
            else:
                result[key] = {'profile': [np.nan for t in model.T],
                                'objective_value': np.nan,
                                'individual profiles': {n: [np.nan for t in model.T] for n in model.N}}
        elif key == 'Dumb':
            model.obj_emission.deactivate()
            model.obj_cost.deactivate()
            model.obj_dumb.activate()
            res = opt.solve(model)
            if res.solver.status == SolverStatus.ok:
                result[key] = {'profile': [model.p_agg[t].value for t in model.T],
                                'objective_value': model.obj_emission(),
                                'individual profiles': {n: [model.p_ch_ev[t, n].value for t in model.T] for n in model.N}}
            else:
                result[key] = {'profile': [np.nan for t in model.T],
                                'objective_value': np.nan,
                                'individual profiles': {n: [np.nan for t in model.T] for n in model.N}}


    return result, model

def update_model(model:ConcreteModel, 
                 ev_data:pd.DataFrame, 
                 flex_time_start: int,
                    flex_time_end: int, bi_directional_charging:bool,opt_horizon:int, keywords=None):
    """
    This function updates the Pyomo model parameters with EV data and base profile.

    param model: Pyomo optimization model
    type model: pyomo.environ.ConcreteModel
    param ev_data: DataFrame containing the EV data
    type ev_data: pd.DataFrame
    param flex_time_start: Hour of the day when flexibility starts
    type flex_time_start: int
    param flex_time_end: Hour of the day when flexibility ends
    type flex_time_end: int
    param bi_directional_charging: Flag to allow bi-directional charging
    type bi_directional_charging: bool
    param opt_horizon: Total time horizon for the optimization in hours
    type opt_horizon: int
    param keywords: Dictionary containing the keys for the data
    type keywords: dict
    """
    
    # Set default keywords if not provided
    if keywords is None:
        keywords = {
            'arrival_time': 'START_int',
            'departure_time': 'STOP_int',
            'energy_charged': 'VOL',
            'max_charging_power': 'P_MAX'
        }

    # Update flexibility start and end times in the model
    # Convert flexibility start and end times to integer time steps
    model.flex_start = flex_time_start
    model.flex_end = flex_time_end

    # Update model parameters with EV data
    for n in model.N:
        model.vol_ev[n] = ev_data[keywords['energy_charged']].iloc[n]
        model.t_a_ev[n] = ev_data[keywords['arrival_time']].iloc[n]
        model.t_d_ev[n] = min(opt_horizon-1,ev_data[keywords['departure_time']].iloc[n])
        model.p_max_ev[n] = ev_data[keywords['max_charging_power']].iloc[n]
        
        # Set minimum charging power based on bi-directional charging flag
        if bi_directional_charging:
            model.p_min_ev[n] = -ev_data[keywords['max_charging_power']].iloc[n]
        else:
            model.p_min_ev[n] = 0

def fix_p_variables(model:ConcreteModel,
                  lead_time:int,
                  values_to_fix_dict:dict,
                  f_start:int):
    """ This fucntions fixes the power variables till the flex_start-lead time
    param model: Pyomo optimization model
    type model: pyomo.environ.ConcreteModel
    param lead_time: Lead time for fixing the variables
    type lead_time: int
    param values_to_fix_dict: Dictionary containing the values to fix for the power variables. Its structure is {n:[0,1,...T]}
    type values_to_fix_dict: dict
    param f_start: Hour of the day when flexibility starts
    type f_start: int
    """
    for n in model.N:
        for t in range(24+f_start-lead_time):
            model.p_ch_ev[t, n].fix(values_to_fix_dict[n][t])

def activate_all(model__):
    """
    This function activates all the constraints and objective functions in the model.
    """
    model__.con_p_ch_limits.activate()
    model__.con_final_soe.activate() 
    model__.con_soe_update.activate()
    model__.con_re_dispatch_down.activate()
    model__.con_re_dispatch_up.activate()
    model__.con_capacity_limitation.activate()
    model__.obj_re_dispatch_dumb.activate()
    model__.obj_re_dispatch_cost.activate()
    model__.obj_re_dispatch_emission.activate()
    model__.obj_capacity_limitation_dumb.activate()
    model__.obj_capacity_limitation_cost.activate()
    model__.obj_capacity_limitation_emission.activate()


def model_selection(model_, option='', base_profile_type:str=''):
    """
    This function selects the model to be solved based on the option provided.
    :param model_: Concrete model
    :type model_: pyomo.core.base.PyomoModel.ConcreteModel
    :param option: Parameter to select the model to be solved
    :type option: string: 're-dispatch', 'capacity_limitation', 'greedy_dispatch'
    :param base_profile_type: Type of the base model
    :type base_profile_type: string, 'dumb', 'Cost', 'MEF'
    :return: pyomo.core.base.PyomoModel.ConcreteModel based on the option provided
    :rtype: pyomo.core.base.PyomoModel.ConcreteModel
    """
    for t in model_.T:
        for n in model_.N:
            model_.p_ch_ev[t, n].free()
            model_.p_ch_ev[t, n].clear()
    if option == 're_dispatch_down':
        if base_profile_type == 'dumb':
            activate_all(model_)
            #Deactivate the constraints for capacity limitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_up.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_cost.deactivate()
            model_.obj_re_dispatch_emission.deactivate()
        elif base_profile_type == 'Cost':
            activate_all(model_)
            #Deactivate the constraints for capacity limitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_up.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_dumb.deactivate()
            model_.obj_re_dispatch_emission.deactivate()
        elif base_profile_type == 'MEF':
            activate_all(model_)
            #Deactivate the constraints for capacity limitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_up.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_dumb.deactivate()
            model_.obj_re_dispatch_cost.deactivate()
    
    elif option == 're_dispatch_up':
        if base_profile_type == 'dumb':
            activate_all(model_)
            #Deactivate the constraints for capacity li mitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_down.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_cost.deactivate()
            model_.obj_re_dispatch_emission.deactivate()
        elif base_profile_type == 'Cost':
            activate_all(model_)
            #Deactivate the constraints for capacity limitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_down.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_dumb.deactivate()
            model_.obj_re_dispatch_emission.deactivate()
        elif base_profile_type == 'MEF':
            activate_all(model_)
            #Deactivate the constraints for capacity limitation
            model_.con_capacity_limitation.deactivate()
            model_.con_re_dispatch_down.deactivate()
            #Deactivate the objective functions for capacity limitation
            model_.obj_capacity_limitation_dumb.deactivate()
            model_.obj_capacity_limitation_cost.deactivate()
            model_.obj_capacity_limitation_emission.deactivate()
            model_.obj_re_dispatch_dumb.deactivate()
            model_.obj_re_dispatch_cost.deactivate()


    elif option == 'capacity_limitation':
        if base_profile_type == 'dumb':
                activate_all(model_)

                #Deactivate the constraints for re-dispatch
                model_.con_re_dispatch_down.deactivate()
                model_.con_re_dispatch_up.deactivate()
                #Deactivate the objective functions for re-dispatch
                model_.obj_re_dispatch_dumb.deactivate()
                model_.obj_re_dispatch_cost.deactivate()
                model_.obj_re_dispatch_emission.deactivate()
                model_.obj_capacity_limitation_cost.deactivate()
                model_.obj_capacity_limitation_emission.deactivate()
        elif base_profile_type == 'Cost':
                activate_all(model_)

                #Deactivate the constraints for re-dispatch
                model_.con_re_dispatch_down.deactivate()
                model_.con_re_dispatch_up.deactivate()
                #Deactivate the objective functions for re-dispatch
                model_.obj_re_dispatch_dumb.deactivate()
                model_.obj_re_dispatch_cost.deactivate()
                model_.obj_re_dispatch_emission.deactivate()
                model_.obj_capacity_limitation_dumb.deactivate()
                model_.obj_capacity_limitation_emission.deactivate()
        elif base_profile_type == 'MEF':
                activate_all(model_)

                #Deactivate the constraints for re-dispatch
                model_.con_re_dispatch_down.deactivate()
                model_.con_re_dispatch_up.deactivate()
                #Deactivate the objective functions for re-dispatch
                model_.obj_re_dispatch_dumb.deactivate()
                model_.obj_re_dispatch_cost.deactivate()
                model_.obj_re_dispatch_emission.deactivate()
                model_.obj_capacity_limitation_dumb.deactivate()
                model_.obj_capacity_limitation_cost.deactivate()

    else:
        raise ValueError('Invalid option provided. Please provide a valid option')

def create_optimization_model(ev_data:pd.DataFrame, base_profile_dict:dict, 
                              delta_t:float, time_horizon:int, 
                              flex_time_start:int, flex_time_end:int, 
                              useful_cost_data:pd.DataFrame, 
                              useful_emisson_data:pd.DataFrame,
                              lambda_value=0.000001, 
                              bi_directional_charging=False, keywords=None):
                         
    """
    This function creates and returns an optimization model for EV charging with flexibility constraints.

    param ev_data: DataFrame containing the EV data
    type ev_data: pd.DataFrame
    param base_profile_dict: Base profile of EVs as per desired charging strategy, it contains the following keys:
        * 'profile': List of aggregated power values for each time step
        * 'objective_value': Objective value of the base profile
        * 'individual profiles': Dictionary containing the individual charging profiles for each EV
    type base_profile_dict: dict
    param delta_t: Step size of analysis in hours
    type delta_t: float
    param time_horizon: Total time horizon for the optimization in hours
    type time_horizon: int
    param flex_time_start: Hour of the day when flexibility starts
    type flex_time_start: int
    param flex_time_end: Hour of the day when flexibility ends
    type flex_time_end: int
    param lambda_value: Sensitivity parameter for the objective function
    type lambda_value: float
    param bi_directional_charging: Flag to allow bi-directional charging
    type bi_directional_charging: bool
    param keywords: Dictionary containing the keys for the data
    type keywords: dict
    return: Pyomo optimization model
    rtype: pyomo.environ.ConcreteModel
    """
    
    # Convert flexibility start and end times to integer time steps
    flex_time_start_ = int(24/delta_t + flex_time_start/delta_t)
    flex_time_end_ = int(24/delta_t + flex_time_end/delta_t)
    
    # Define the length of the sets
    Set_T_length = time_horizon
    Set_N_length = len(ev_data)
    
    # Create an empty Pyomo model
    model = ConcreteModel()
    
    # Define sets
    model.N = Set(ordered=True, initialize=np.arange(Set_N_length))
    model.T = Set(ordered=True, initialize=np.arange(Set_T_length))
    
    # Define parameters
    model.vol_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kWh, doc='Maximum volume of charge in kWh')
    model.t_a_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step from which a vehicle is available')
    model.t_d_ev = Param(model.N, within=NonNegativeIntegers, mutable=True, doc='Time step till which a vehicle is available')
    model.p_max_ev = Param(model.N, within=NonNegativeReals, mutable=True, units=u.kW, doc='Maximum charging power in kW')
    model.p_min_ev = Param(model.N, within=Reals, mutable=True, units=u.kW, doc='Minimum charging power in kW')
    model.base_profile = Param(model.T, mutable=True, doc='Aggregated profile of EVs as per dumb charging')
    model.step_size = Param(initialize=delta_t, mutable=True, doc='Step size of analysis in hours')
    model.flex_start = Param(initialize=flex_time_start_, mutable=True, doc='Integer timestep from which flexibility is required')
    model.flex_end = Param(initialize=flex_time_end_, mutable=True, doc='Integer timestep till which flexibility is required')
    model.lambda_ = Param(initialize=lambda_value, mutable=False, doc='Sensitivity parameter for the objective function')
    model.cost = Param(model.T, within=NonNegativeReals, mutable=True, units=u.e/u.kWh, doc='Cost of energy at time t per kWh')
    model.emission = Param(model.T, within=NonNegativeReals, mutable=True, units=u.e/u.kWh, doc='Emission factor at time t per kWh')

    # Define variables
    model.p_ch_ev = Var(model.T, model.N, within=Reals, units=u.kW, doc='Charging power of nth EV at time t')
    model.soe_ev = Var(model.T, model.N, within=NonNegativeReals, units=u.kWh, doc='State of charge of nth EV at time t')
    model.aux_var = Var(within=NonNegativeReals, doc='Auxiliary variable used for objective function')
    model.p_agg = Var(model.T, within=Reals, units=u.kW, doc='Aggregated power at time t')
    
    # Update model parameters with EV data and base profile
    # Update model parameters
    update_model(model, ev_data, flex_time_start_, flex_time_end_, bi_directional_charging, Set_T_length, keywords)
    for t in model.T:
        model.cost[t] = useful_cost_data['Day-ahead Price [EUR/kWh]'].iloc[t]
        model.emission[t] = useful_emisson_data['MEF'].iloc[t]
    
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
        if (t >= model_.flex_start.value) & (t <= model_.flex_end.value):
            return sum(model_.p_ch_ev[t, n] for n in model_.N) <= model_.base_profile[t] - model_.aux_var
        else:
            return Constraint.Skip
    
    def re_dispatch_up(model_, t):
        if (t >= model_.flex_start.value) & (t <= model_.flex_end.value):
            return sum(model_.p_ch_ev[t, n] for n in model_.N) >= model_.base_profile[t] + model_.aux_var
        else:
            return Constraint.Skip
    
    def capacity_limitation(model_, t):
        if (t >= model_.flex_start.value) & (t <= model_.flex_end.value):
            return sum(model_.p_ch_ev[t, n] for n in model_.N) <= model_.aux_var
        else:
            return Constraint.Skip

    
    # Add constraints to the model
    model.con_p_ch_limits = Constraint(model.T, model.N, rule=charging_power_limits)
    model.con_final_soe = Constraint(model.T, model.N, rule=final_soe)
    model.con_soe_update = Constraint(model.T, model.N, rule=soe_update)
    model.con_re_dispatch_down = Constraint(model.T, rule=re_dispatch_down)
    model.con_re_dispatch_up = Constraint(model.T, rule=re_dispatch_up)
    model.con_capacity_limitation = Constraint(model.T, rule=capacity_limitation)
    model.con_p_agg = Constraint(model.T, rule=lambda model_, t: model_.p_agg[t] == sum(model_.p_ch_ev[t, n] for n in model_.N))
    
    # Define objectives
    model.obj_re_dispatch_dumb = Objective(expr=(model.aux_var+model.lambda_ * sum(sum(model.soe_ev[t, n] for n in model.N) for t in model.T)), sense=maximize)
    model.obj_re_dispatch_cost = Objective(expr=(model.aux_var-model.lambda_*sum(model.p_agg[t]*model.step_size*model.cost[t] for t in model.T)) , sense=maximize)
    model.obj_re_dispatch_emission = Objective(expr=(model.aux_var-model.lambda_*sum(model.p_agg[t]*model.step_size*model.emission[t] for t in model.T)) , sense=maximize)
    model.obj_capacity_limitation_dumb = Objective(expr=model.aux_var - model.lambda_ * sum(sum(model.soe_ev[t, n] for n in model.N) for t in model.T), sense=minimize)
    model.obj_capacity_limitation_cost = Objective(expr=model.aux_var + model.lambda_ * sum(model.p_agg[t]*model.step_size*model.cost[t] for t in model.T), sense=minimize)
    model.obj_capacity_limitation_emission = Objective(expr=model.aux_var + model.lambda_ * sum(model.p_agg[t]*model.step_size*model.emission[t] for t in model.T), sense=minimize)

    return model

def calculate_flexibility(ev_data:pd.DataFrame,
                          base_profile_dict: dict, 
                          product_list: list,
                          delta_t:float, 
                          time_horizon:int, 
                          flex_time_start:int, 
                          flex_time_end:int, 
                          lambda_value:float, 
                          bi_directional_charging_value:list, 
                          lead_time:list,
                          keywords_:dict,
                          cost_data:pd.DataFrame,
                          emisson_data:pd.DataFrame):
    """
    This function creates and solves an optimization model for EV charging based on the provided parameters and product type.
    
    param ev_data: DataFrame containing the EV data
    type ev_data: pd.DataFrame
    param base_profile_dict: Base profile of EVs as per dumb charging, it contains objective, aggregate profiles and individual profiles
    type base_profile_dict: dict
    param delta_t: Step size of analysis in hours
    type delta_t: float
    param time_horizon: Total time horizon for the optimization in hours
    type time_horizon: int
    param flex_time_start: Hour of the day when flexibility starts
    type flex_time_start: int
    param flex_time_end: Hour of the day when flexibility ends
    type flex_time_end: int
    param lambda_value: Sensitivity parameter for the objective function
    type lambda_value: float
    param bi_directional_charging_value: Flag to allow bi-directional charging
    type bi_directional_charging_value: bool
    param keywords_: Dictionary containing the keys for the data
    type keywords_: dict
    param product_type: List of product types to optimize for
    type product_type: list
    param lead_time: List of lead times for the optimization    
    type lead_time: list
    param cost_data: Data frame containing the cost data
    type cost_data: pd.DataFrame
    param emisson_data: Data frame containing the emission data
    type emisson_data: pd.DataFrame
    return: Dictionary containing the optimized profiles and flexibility values for each product type
    rtype: dict
    """
    result_dict = {}
    model = create_optimization_model(ev_data=ev_data, 
                                      base_profile_dict=base_profile_dict, 
                                      delta_t=delta_t, 
                                      time_horizon=time_horizon, 
                                      flex_time_start=flex_time_start, 
                                      flex_time_end=flex_time_end, 
                                      lambda_value=lambda_value, 
                                      bi_directional_charging=bi_directional_charging_value, 
                                      keywords=keywords_,
                                      useful_cost_data=cost_data,
                                        useful_emisson_data=emisson_data)
                                      

    opt = SolverFactory('gurobi')

    # For different charging strategy
    for bi in bi_directional_charging_value:
        if bi:
            for n in model.N:
                model.p_min_ev[n] = -model.p_max_ev[n]
        else:
            for n in model.N:
                model.p_min_ev[n] = 0
        for base_profile_type in base_profile_dict.keys():

            # Update base profile in the model
            for t in model.T:
                model.base_profile[t] = base_profile_dict[base_profile_type]['profile'][t]
            for product in product_list:
                for lead_time_ in lead_time:
                    print(f'Running for product: {product}, base_profile: {base_profile_type}, bi: {bi}, lead_time: {lead_time_}')
                    if product in ['capacity_limitation', 're_dispatch_down', 're_dispatch_up']:
                        if lead_time_ == 'inf':
                            model_selection(model, product,base_profile_type=base_profile_type)
                            res = SolverFactory('gurobi').solve(model) 
                        else:
                            model_selection(model, product, base_profile_type=base_profile_type)
                            fix_p_variables(model, lead_time=lead_time_, values_to_fix_dict=base_profile_dict[base_profile_type]['individual profiles'], f_start=flex_time_start)
                            res = SolverFactory('gurobi').solve(model)  
                        
                        result_dict[(f'product_{product}_bau_{base_profile_type}_bi_{bi}_lead_{lead_time_}')] = info(model, res)
                    else:
                        raise ValueError('Invalid product type provided. Please provide a valid product type')

    return result_dict, model

def info(model:ConcreteModel, solver_status:pyomo.opt.results.results_.SolverResults):
    """
    This function returns all the information in dictionary format for the model.
    """
    if solver_status.solver.status==SolverStatus.ok:
        dict_ro_return = {'flexibility (kW)': model.aux_var.value,
                          'Aggeregate profile (kW)': [sum(model.p_ch_ev[t, n].value for n in model.N) for t in model.T]}
    else:
        dict_ro_return = {'flexibility (kW)': None,
                          'Aggeregate profile (kW)': None}
    return dict_ro_return
    
      

##########################################################################################
##########################################################################################
# %%
# Process EV data for optimization
year_ = 2023
month_ = 4
day_ = 5
delta_t = 1
key_dict = {'Actual energy charged (kWh)': 'VOL',
            'Arrival_time': 'START',
            'Departure_time': 'STOP',
            'Max charging power (kW)' : 'P_MAX',
            'Connector_id': 'STA_NR',
            # --- For optimization model ---
            'actual_energy_charged_value' : 'VOL',
            'arrival_time_integer' : 'START_int',
            'departure_time_integer_adjusted' : 'STOP_int',
            'max_charging_power' :  'P_MAX',
            'arrival_time': 'START_int',
            'departure_time': 'STOP_int',
            'energy_charged': 'VOL',
            'max_charging_power': 'P_MAX'
            }
base_profiles_to_run = ['Cost','MEF','Dumb']
max_connection_time = 24  # in hours
time_horizon = 3*24
flex_time_start = 17  # Hour
flex_time_end = 22  # Hour
lambda_value = 1e-6
bi_directional_charging_value = [True,False]
product_type_list = ['re_dispatch_down','capacity_limitation']
base_profile_type = 'Cost'
lead_time = ['inf',1,6]

cost_df = pd.read_pickle(path_cost_data)
mef_df = pd.read_pickle(path_emisson_data)


# # Filter out the data for optimization
# ev_processed_data = process_for_optimization(ev_data, year_, month_, day_, delta_t, key_dict, max_connection_time)
# # Get cost and emission data
# useful_cost_data, useful_emisson_data = get_cost_data(cost_df, mef_df, ev_processed_data=ev_processed_data)
# # Generate base profiles
# bau_profiles, model = bau_profile_generator(ev_processed_data, useful_cost_data, useful_emisson_data, delta_t, base_profiles_to_run, time_horizon)
# result_dict, model_f = calculate_flexibility(ev_data=ev_processed_data, 
#                                                 base_profile = bau_profiles[base_profile_type],
#                                                 lead_time=lead_time,
#                                                 delta_t=delta_t, 
#                                                 time_horizon=time_horizon, 
#                                                 flex_time_start=flex_time_start, 
#                                                 flex_time_end=flex_time_end, 
#                                                 lambda_value=lambda_value, 
#                                                 bi_directional_charging_value=bi_directional_charging_value, 
#                                                 keywords_=key_dict, 
#                                                 product_list=product_type_list)

def single_run_flex_service(year_: int, 
                            month_: int, 
                            day_: int, 
                            delta_t: float, 
                            key_dict: dict, 
                            base_profiles_to_run: list, 
                            max_connection_time: int, 
                            time_horizon: int, 
                            flex_time_start: int, 
                            flex_time_end: int, 
                            lambda_value: float, 
                            bi_directional_charging_value: list, 
                            product_type_list: list, 
                            lead_time: list, 
                            path_cost_data: str, 
                            path_emisson_data: str, 
                            ev_data: pd.DataFrame):
    """
    This function performs a single run of the flexibility service calculation.

    param year_: Year of the data
    type year_: int
    param month_: Month of the data
    type month_: int
    param day_: Day of the data
    type day_: int
    param delta_t: Time step size
    type delta_t: float
    param key_dict: Dictionary containing the keys for the data
    type key_dict: dict
    param base_profiles_to_run: List of base profiles to run
    type base_profiles_to_run: list
    param max_connection_time: Maximum connection time in hours
    type max_connection_time: int
    param time_horizon: Total time horizon for the optimization in hours
    type time_horizon: int
    param flex_time_start: Hour of the day when flexibility starts
    type flex_time_start: int
    param flex_time_end: Hour of the day when flexibility ends
    type flex_time_end: int
    param lambda_value: Sensitivity parameter for the objective function
    type lambda_value: float
    param bi_directional_charging_value: List of flags to allow bi-directional charging
    type bi_directional_charging_value: list
    param product_type_list: List of product types to optimize for
    type product_type_list: list
    param base_profile_type: Base profile type
    type base_profile_type: str
    param lead_time: List of lead times
    type lead_time: list
    param path_cost_data: Path to the cost data file
    type path_cost_data: str
    param path_emisson_data: Path to the emission data file
    type path_emisson_data: str
    param ev_data: Data frame containing the EV data
    type ev_data: pd.DataFrame
    return: Dictionary containing the optimized profiles and flexibility values for each product type
    rtype: dict
    """
    cost_df = pd.read_pickle(path_cost_data)
    mef_df = pd.read_pickle(path_emisson_data)

    # Filter out the data for optimization
    ev_processed_data = process_for_optimization(ev_data, year_, month_, day_, delta_t, key_dict, max_connection_time)
    # Get cost and emission data
    useful_cost_data, useful_emisson_data = get_cost_data(cost_df, mef_df, ev_processed_data=ev_processed_data)
    # Generate base profiles
    bau_profiles, _ = bau_profile_generator(ev_processed_data, useful_cost_data, useful_emisson_data, delta_t, base_profiles_to_run, time_horizon)
    result_dict, _ = calculate_flexibility(ev_data=ev_processed_data, 
                                                 base_profile_dict=bau_profiles,
                                                 lead_time=lead_time,
                                                 delta_t=delta_t, 
                                                 time_horizon=time_horizon, 
                                                 flex_time_start=flex_time_start, 
                                                 flex_time_end=flex_time_end, 
                                                 lambda_value=lambda_value, 
                                                 bi_directional_charging_value=bi_directional_charging_value, 
                                                 keywords_=key_dict, 
                                                 product_list=product_type_list,
                                                 cost_data=useful_cost_data,
                                                 emisson_data=useful_emisson_data)
    return result_dict, bau_profiles

result_dict, bau_profiles = single_run_flex_service(year_, month_, day_, delta_t, key_dict, base_profiles_to_run, max_connection_time, time_horizon, flex_time_start, flex_time_end, lambda_value, bi_directional_charging_value, product_type_list, lead_time, path_cost_data, path_emisson_data, ev_data)
# %%

# %%
bau = 'Dumb'
f, ax = plt.subplots(len(product_type_list), len(bi_directional_charging_value), figsize=(15, 10), sharex=True, sharey='row')
if ax.ndim == 1:
    ax = ax.reshape(1, -1)
for i, product in enumerate(product_type_list):
    for j, bi in enumerate(bi_directional_charging_value):
        # plot the base profile
        ax[i,j].step(range(time_horizon), bau_profiles[bau]['profile'], label=bau, color='red')
        #Plot the optimized profile
        for lt in lead_time:
            ax[i,j].step(range(time_horizon), result_dict[(f'product_{product}_bau_{bau}_bi_{bi}_lead_{lt}')]['Aggeregate profile (kW)'], label=f'Lead time: {lt}', linestyle='--')
        #Shade the flexibility time
        ax[i,j].axvspan(24+flex_time_start, 24+flex_time_end, alpha=0.3, color='green')

        ax[i,j].set_xlabel('Time step')
        ax[i,j].set_ylabel('Power (kW)')
        ax[i,j].set_title(f'{product} - Bi-directional charging: {bi}')
        # ax[i,j].set_xlim(24, 48)
        ax[i,j].legend()
plt.show()
# %%


