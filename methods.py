from dataset import BootstrapSplitter, CallData
from FS import *
from FJ import *
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
import time
from itertools import product
import pandas as pd
import copy


def fullyjsequential_experiment(num_splits = 10, max_num_rules = 10, max_num_propos = 5, max_num_nonzero_coef = None, dataset_name = None, random_state = 111, lambda_value_rule_weights = [1e-10], epsilon_prop = 0.1, epsilon_nzco = 0.02):
    lam_prop = 1e-10
    dataset_name = dataset_name

    dataset = CallData().call(dataset_name = dataset_name, data_format = 'df')
    X = dataset.X
    y = dataset.y

    if len(X)>500:
        train_size = 500
    else:
        train_size = len(X)

    if max_num_nonzero_coef is None:
        max_num_nonzero_coef = X.shape[1]

    splitter = BootstrapSplitter(reps=num_splits, train_size=train_size, random_state=random_state, replace=True)
    all_rule_models = np.empty(shape=(num_splits, max_num_rules, len(lambda_value_rule_weights)), dtype=object)

    df = pd.DataFrame(columns=['rep no', 'max rule no', 'test loss', 'train loss', 'c1', 'c2', 'c3', 'l1 lambda', 'comp time (sec)'])
    elapsed_time = pd.DataFrame(columns=['dataset name', 'time (sec)'])
    split_ind = 0
    df_ind = 0
    print('regression type is '+dataset.learning_type)
    start_time = time.time()
    for outer_train_ind, outer_test_ind in splitter.split(X):
        print(f'----repetition number {split_ind+1}----')
        x_train = X.loc[outer_train_ind]
        y_train = y.loc[outer_train_ind]
        x_test = X.loc[outer_test_ind]
        y_test = y.loc[outer_test_ind]
        for num_rule_ind, num_rules in enumerate(np.arange(1, max_num_rules+1)):
            print(f'----learning rule number {num_rules}----')
            for l_ind, l in enumerate(lambda_value_rule_weights):
                print(f'----lambda value {l}----')
                start_time_one_iteration = time.time()
                model_rule = LinRepRuleEnsemble(num_rules=num_rules, max_props=max_num_propos, lambda_value_rule_weights=l, lambda_value_prop=lam_prop, max_num_nonzero_coef=max_num_nonzero_coef, optim_problem=dataset.learning_type, epsilon_nzco=epsilon_nzco, epsilon_prop=epsilon_prop).fit(x_train,y_train)
                end_time_one_iteration = time.time()
                if dataset.learning_type == 'linreg':
                    test_loss = mean_squared_error(y_test, model_rule.predict(x_test))
                    train_loss = mean_squared_error(y_train, model_rule.predict(x_train))
                else:
                    test_loss = log_loss(y_test, model_rule.predict(x_test))
                    train_loss = log_loss(y_train, model_rule.predict(x_train))
            

                all_rule_models[split_ind, num_rule_ind, l_ind] = model_rule

                df.loc[df_ind, 'rep no'] = split_ind+1
                df.loc[df_ind,'max rule no'] = num_rules
                df.loc[df_ind,'test loss'] = test_loss
                df.loc[df_ind,'train loss'] = train_loss
                df.loc[df_ind,'c1'] = model_rule.complexity_c1()
                df.loc[df_ind,'c2'] = model_rule.complexity_c2()
                df.loc[df_ind,'c3'] = model_rule.complexity_c3()
                df.loc[df_ind, 'l1 lambda'] = l
                df.loc[df_ind, 'comp time (sec)'] = end_time_one_iteration - start_time_one_iteration

                df_ind += 1
        split_ind += 1
    end_time = time.time()
    elapsed_time.loc[0,'time (sec)'] = end_time - start_time
    elapsed_time.loc[0,'dataset name'] = dataset_name
    df.to_excel('FS_'+dataset_name+'.xlsx', sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('FS_'+dataset_name+'.xlsx', engine='openpyxl', mode='a') as writer:
        elapsed_time.to_excel(writer, sheet_name='CalcTime', index=False)
    
    return df, elapsed_time, all_rule_models


def fullyjoint_experiment(num_splits = 10, max_num_rules = 10, max_num_propos = 5, num_restarts = 1, lambda_values = None, dataset_name = None, prunning_treshold = 10, random_state = 111):
    dataset_name = dataset_name
    dataset = CallData().call(dataset_name = dataset_name, data_format = 'arr')
    X = dataset.X
    y = dataset.y
    if len(X)>500:
        train_size = 500
    else:
        train_size = len(X)

    if lambda_values is None:
        lambda_values = np.concatenate(([0], np.logspace(-4, 0, 5)))
        l1_w_lambda_values = list(product(lambda_values, lambda_values))
    else:
        l1_w_lambda_values = lambda_values

    num_seeds = num_restarts*num_splits
    splitter = BootstrapSplitter(reps=num_splits, train_size=train_size, random_state=random_state, replace=True)
    seeds_list = np.random.choice(num_seeds+np.random.choice(range(50,5000), 1)[0], num_seeds)
    seeds_list = seeds_list.reshape(num_splits,num_restarts)

    modified_sigmoid = SigmoidActivation(a=20, b=10)
    lr = LearnRule(n_epochs=250, learning_rate=0.1)

    all_rule_models = np.empty(shape=(num_splits, len(l1_w_lambda_values)), dtype=object)
    all_total_loss = np.empty(shape=(num_splits, len(l1_w_lambda_values)), dtype=object)
    split_ind = 0
    df_ind = 0
    df = pd.DataFrame(columns=['rep no', 'max rule no', 'test loss', 'train loss', 'c1', 'c2', 'c3', 'l1 lambda', 'w lambda', 'comp time (sec)'])
    df_nd = pd.DataFrame(columns=['rep no', 'max rule no', 'test loss', 'train loss', 'c1', 'c2', 'c3', 'l1 lambda', 'w lambda', 'comp time (sec)'])
    elapsed_time = pd.DataFrame(columns=['dataset name', 'time (sec)'])
    start_time = time.time()
    for outer_train_ind, outer_test_ind in splitter.split(X):
        print(f'----repetition number {split_ind+1}----')
        x_train = X[outer_train_ind]
        y_train = y[outer_train_ind]
        x_test = X[outer_test_ind]
        y_test = y[outer_test_ind]
        for l_ind, l in enumerate(l1_w_lambda_values):
            print(f'----lambda value: {l}----')
            best_train_loss = np.inf
            best_train_loss_nd = np.inf
            for i in range(num_restarts):
                set_seed(int(seeds_list[split_ind, i]))
                model_rule = RuleArchitecture(input_size=x_train.shape[1], output_size=1, conditions_size= max_num_rules*[max_num_propos], regression_type=dataset.learning_type, q_act=modified_sigmoid)
                lf = LossFunction(model=model_rule, l1_propositions_penalty_lambda=l[0], l1_rule_weights_penalty_lambda=l[0], w_penalty_lambda=l[1])
                start_time_one_iteration = time.time()
                model_rule, total_loss = lr.fit(model=model_rule, x=x_train, y=y_train, loss_func=lf)
                end_time_one_iteration = time.time()
                if dataset.learning_type == 'linreg':
                    train_loss_nd = mean_squared_error(y_train, model_rule.predict_step_step(x_train))
                    test_loss_nd = mean_squared_error(y_test, model_rule.predict_step_step(x_test))
                else:
                    train_loss_nd = log_loss(y_train, model_rule.predict_step_step(x_train))
                    test_loss_nd = log_loss(y_test, model_rule.predict_step_step(x_test))

                if train_loss_nd < best_train_loss_nd:
                    best_train_loss_nd = train_loss_nd
                    best_test_loss_nd = test_loss_nd
                    best_model_rule_nd = copy.deepcopy(model_rule)
                    best_complexity_c1_nd, best_complexity_c2_nd, best_complexity_c3_nd = model_rule.complexity()
                    total_loss_sel_nd = total_loss

                model_rule.prune_rules(percent=prunning_treshold)
                model_rule.binarise_model()
                if dataset.learning_type == 'linreg':
                    train_loss = mean_squared_error(y_train, model_rule.predict_step_step(x_train))
                    test_loss = mean_squared_error(y_test, model_rule.predict_step_step(x_test))
                else:
                    train_loss = log_loss(y_train, model_rule.predict_step_step(x_train))
                    test_loss = log_loss(y_test, model_rule.predict_step_step(x_test))

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_test_loss = test_loss
                    best_model_rule = copy.deepcopy(model_rule)
                    best_complexity_c1, best_complexity_c2, best_complexity_c3 = model_rule.complexity()
                    total_loss_sel = total_loss
                    best_elapsed_time = end_time_one_iteration - start_time_one_iteration

            df_nd.loc[df_ind, 'rep no'] = split_ind+1
            df_nd.loc[df_ind,'max rule no'] = max_num_rules
            df_nd.loc[df_ind,'test loss'] = best_test_loss_nd
            df_nd.loc[df_ind,'train loss'] = best_train_loss_nd
            df_nd.loc[df_ind,'c1'] = best_complexity_c1_nd
            df_nd.loc[df_ind,'c2'] = best_complexity_c2_nd
            df_nd.loc[df_ind,'c3'] = best_complexity_c3_nd
            df_nd.loc[df_ind,'l1 lambda'] = l[0]
            df_nd.loc[df_ind,'w lambda'] = l[1]
            df_nd.loc[df_ind, 'comp time (sec)'] = best_elapsed_time

            df.loc[df_ind, 'rep no'] = split_ind+1
            df.loc[df_ind,'max rule no'] = max_num_rules
            df.loc[df_ind,'test loss'] = best_test_loss
            df.loc[df_ind,'train loss'] = best_train_loss
            df.loc[df_ind,'c1'] = best_complexity_c1
            df.loc[df_ind,'c2'] = best_complexity_c2
            df.loc[df_ind,'c3'] = best_complexity_c3
            df.loc[df_ind,'l1 lambda'] = l[0]
            df.loc[df_ind,'w lambda'] = l[1]
            df.loc[df_ind, 'comp time (sec)'] = best_elapsed_time

            all_rule_models[split_ind, l_ind] = copy.deepcopy(best_model_rule)
            all_total_loss[split_ind, l_ind] = total_loss_sel

            df_ind += 1
        split_ind += 1
        
    end_time = time.time()
    elapsed_time.loc[0,'time (sec)'] = end_time - start_time
    elapsed_time.loc[0,'dataset name'] = dataset_name
    df_nd.to_excel('nondis_FJ_'+dataset_name+'.xlsx', sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('nondis_FJ_'+dataset_name+'.xlsx', engine='openpyxl', mode='a') as writer:
        elapsed_time.to_excel(writer, sheet_name='CalcTime', index=False)
    df.to_excel('FJ_'+dataset_name+'.xlsx', sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('FJ_'+dataset_name+'.xlsx', engine='openpyxl', mode='a') as writer:
        elapsed_time.to_excel(writer, sheet_name='CalcTime', index=False)


    return df, df_nd, elapsed_time, all_rule_models, all_total_loss