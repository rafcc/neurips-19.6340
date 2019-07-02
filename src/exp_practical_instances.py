import numpy as np
import model
import trainer
import subfunction
import data
from itertools import combinations
import copy
import visualize
import random


def exp_practical_instances(trn_data,test_data):
    TRN_DATA = trn_data
    TEST_DATA = test_data
    for NUM_SAMPLE_BORGES in [50, 100]:
        DATA_NAME = TRN_DATA
        for DEGREE in [2,3]:
            DATA_DIR = "../data"
            RESULT_DIR = "../results_practical_instances/D"+str(DEGREE)+"_N"+str(NUM_SAMPLE_BORGES)
            DIMENSION_SIMPLEX = 3
            DIMENSION_SPACE = 3
            def sampling_data_and_param(d,p,n,seed):
                random.seed(seed)
                s = [i for i in range(d.shape[0])]
                s_=random.sample(s,n)
                print(len(s_))
                return(d[s_,:],p[s_,:])

            if DEGREE == 2:
                if DIMENSION_SIMPLEX==3:
                    NUM_SAMPLE_INDUCTIVE = [int(round(NUM_SAMPLE_BORGES*(1-0.739)/3)),
                                            int(round(NUM_SAMPLE_BORGES*0.739/3))]
                if DIMENSION_SIMPLEX==9:
                    NUM_SAMPLE_INDUCTIVE = [int(round(NUM_SAMPLE_BORGES*(1-0.758)/3)),
                                            int(round(NUM_SAMPLE_BORGES*0.758/3))]
                if NUM_SAMPLE_BORGES == 6:
                    NUM_SAMPLE_INDUCTIVE = [1,1]
            if DEGREE == 3:
                NUM_SAMPLE_INDUCTIVE = [int(round(NUM_SAMPLE_BORGES*(1-0.587-0.314)/3)),
                                        int(round(NUM_SAMPLE_BORGES*0.587/3)),
                                        int(round(NUM_SAMPLE_BORGES*0.314))]

            SEED_LIST = [i for i in range(20)]

            for SEED in SEED_LIST:
                np.random.seed(SEED)
                objective_function_indices_list = [i for i in range(DIMENSION_SIMPLEX)]
                subproblem_indices_list = []
                for i in range(1,len(objective_function_indices_list)+1):
                    for c in combinations(objective_function_indices_list, i):
                        subproblem_indices_list.append(c)

                bezier_simplex = model.BezierSimplex(dimSpace=DIMENSION_SPACE,
                                                 dimSimplex=DIMENSION_SIMPLEX,
                                                 degree = DEGREE)
                borges_pastva_trainer = trainer.BorgesPastvaTrainer(dimSpace=DIMENSION_SPACE,
                                                    dimSimplex=DIMENSION_SIMPLEX,
                                                 degree = DEGREE)
                monomial_degree_list = [i for i in subfunction.BezierIndex(dim=DIMENSION_SIMPLEX,
                                        deg=DEGREE)]


                data_all = {}
                param_all = {}
                for e in subproblem_indices_list:
                    if len(e) <= DEGREE or len(e) == DIMENSION_SIMPLEX:
                        string = '_'.join([str(i+1) for i in e])
                        tmp = data.Dataset(DATA_DIR+'/'+DATA_NAME+',f_'+string)
                        data_all[e] = tmp.values

                        tmp = data.Dataset(DATA_DIR+'/'+DATA_NAME+',w_'+string)
                        param_all[e] = tmp.values

                # sampling
                data_trn_borges = {}
                param_trn_borges = {}
                e_borges = tuple([i for i in range(DIMENSION_SIMPLEX)])
                data_trn_borges[e_borges], param_trn_borges[e_borges] = sampling_data_and_param(d=data_all[e_borges],p=param_all[e_borges],n=NUM_SAMPLE_BORGES,seed=SEED)

                data_trn_inductive = {}
                param_trn_inductive = {}
                for e in data_all:
                    if len(e) <= DEGREE:
                        data_trn_inductive[e],param_trn_inductive[e] = sampling_data_and_param(d=data_all[e],
                        p=param_all[e],
                        n=NUM_SAMPLE_INDUCTIVE[len(e)-1],
                        seed=SEED+sum(e))
                print(param_trn_borges[e_borges][0:10])
                print(data_trn_borges[e_borges][0:10])
                print(param_trn_borges[e_borges].shape,data_trn_borges[e_borges].shape,)
                print(param_trn_inductive[(0,)].shape,data_trn_inductive[(0,)].shape,)
                print(param_trn_inductive[(0,1)].shape,data_trn_inductive[(0,1)].shape,)
                if DEGREE == 3:
                    print(param_trn_inductive[(0,1)].shape,data_trn_inductive[(0,1)].shape,)
                # borges learning
                control_point_borges = borges_pastva_trainer.update_control_point(t_mat = param_trn_borges[e_borges],
                                                                                  data=data_trn_borges[e_borges],
                                                                                  c = {},
                                                                                  indices_all = monomial_degree_list,
                                                                                  indices_fix = [])
                print("borges")
                for key in control_point_borges:
                    print(key,control_point_borges[key])

                # inductive learning
                freeze_multiple_degree_set = set()
                control_point_inductive =  {}
                for dim in range(1,DIMENSION_SIMPLEX+1):
                    for index in data_trn_inductive:
                        if len(index) == dim:
                            target_multiple_degree_set  = subfunction.extract_multiple_degree(degree_list=monomial_degree_list,index_list=index)
                            target_multiple_degree_list = list(target_multiple_degree_set)
                            freeze_multiple_degree_list = list(freeze_multiple_degree_set.intersection(target_multiple_degree_set))
                            a = borges_pastva_trainer.update_control_point(t_mat=param_trn_inductive[index],
                                                                           data=data_trn_inductive[index],
                                                                           c = control_point_inductive,
                                                                           indices_all = target_multiple_degree_list,
                                                                           indices_fix = freeze_multiple_degree_list)
                            control_point_inductive.update(a)
                            freeze_multiple_degree_set = freeze_multiple_degree_set.union(target_multiple_degree_set)
                            if len(freeze_multiple_degree_set) == len(monomial_degree_list):
                                break
                print("inductive")
                for key in control_point_inductive:
                    print(key,control_point_inductive[key])

                # evaluation
                data_tst = data.Dataset(DATA_DIR+'/'+TEST_DATA+',f_'+'_'.join([str(i+1) for i in e_borges])).values
                param_tst = data.Dataset(DATA_DIR+'/'+TEST_DATA+',w_'+'_'.join([str(i+1) for i in e_borges])).values
                pred_borges = bezier_simplex.generate_points(c=control_point_borges,tt=param_tst)
                pred_inductive = bezier_simplex.generate_points(c=control_point_inductive,tt=param_tst)

                random_l2risk_borges = subfunction.calculate_l2_expected_error(true=data_tst,
                                                                              pred=pred_borges)
                random_l2risk_inductive = subfunction.calculate_l2_expected_error(true=data_tst,
                                                                        pred=pred_inductive)
                print(random_l2risk_borges,random_l2risk_inductive)

                results = {}
                results['random_l2risk'] = {}
                results['random_l2risk']['borges'] = float(random_l2risk_borges)
                results['random_l2risk']['inductive'] = float(random_l2risk_inductive)
                resultdir = RESULT_DIR + '/'+DATA_NAME+'/'+str(SEED)
                subfunction.create_directory(resultdir)
                subfunction.write_result(result=results,fname=resultdir+'/output.yml')

                if SEED == 0:
                    visualize.plot_estimated_pairplot(d1=data_tst,
                                                  d2=pred_borges,
                                                  d3=data_trn_borges[e_borges],
                                                  output_name=resultdir+'/borges.png')
                    visualize.plot_estimated_pairplot(d1=data_tst,
                                                  d2=pred_inductive,
                                                  d3=np.r_[data_trn_inductive[(0,)],
                                                           data_trn_inductive[(1,)],
                                                           data_trn_inductive[(2,)],
                                                           data_trn_inductive[(0,1)],
                                                           data_trn_inductive[(1,2)],
                                                           data_trn_inductive[(0,2)],
                                                           ],
                                                  output_name=resultdir+'/inductive.png')
if __name__ == '__main__':
    exp_practical_instances(trn_data="Birthwt6.csv,n_1000,r_1e+00,e_1e-01,m_0e+00,s_42,l_1e+00,t_1e-07,i_10000",
                            test_data="Birthwt6.csv,n_1000,r_1e+00,e_1e-01,m_0e+00,s_43,l_1e+00,t_1e-07,i_10000")
    exp_practical_instances(trn_data="MED,dim_4,obj_3,convexity_2.0,noise_0.01",
                            test_data="MED,dim_4,obj_3,convexity_2.0,noise_0.01_test")
