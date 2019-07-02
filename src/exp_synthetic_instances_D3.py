import main
import generate_yaml
import math
from multiprocessing import Process
import time


import math
def my_round(x, d=0):
    p = 10 ** d
    return float(math.floor((x * p) + math.copysign(0.5, x)))/p

def convert_params_to_string(dimension_simplex,
                             dimension_space,
                             degree,
                             N,
                             simplextype,
                             num_sample_train,sigma):
    s = simplextype+"_"
    s = s+"M."+str(dimension_simplex)
    s = s+"_L."+str(dimension_space)
    s = s+"_D."+str(degree)
    s = s+"_N."+str(N)
    s = s+"_sigma."+str(sigma)
    return(s)

def exp3d(n,dimspace,dimsimplex, opt_ratio_flag,borges_flag):

    #N_list = [1000]
    DEGREE = 3
    SIGMA = 0.1

    if opt_ratio_flag==1:
        OPT_RATIO_DIC = {3:[1-0.587-0.314,0.587,0.314],
                         4:[1-0.453-0.481,0.453,0.481],
                         5:[1-0.386-0.547,0.386,0.547],
                         6:[1-0.365-0.577,0.365,0.577],
                         7:[1-0.359-0.589,0.359,0.589],
                         8:[1-0.336-0.623,0.336,0.623]}
    else:
        OPT_RATIO_DIC = {3:[1-0.333-0.333,0.333,0.333],
                         4:[1-0.333-0.333,0.333,0.333],
                         5:[1-0.333-0.333,0.333,0.333],
                         6:[1-0.333-0.333,0.333,0.333],
                         7:[1-0.333-0.333,0.333,0.333],
                         8:[1-0.333-0.333,0.333,0.333]}

    TYPE_LIST =["linear"]#,"squareroot"]#,"linear"]
    NUM_TRIAL = 20

    N = n
    DIM_SIMPLEX = dimsimplex
    DIM_SPACE = dimspace
    RATIO = OPT_RATIO_DIC[DIM_SIMPLEX]
    for TYPE in TYPE_LIST:
        training_sample_list = [int(my_round(N*RATIO[0]/DIM_SIMPLEX)),
                                int(my_round(N*RATIO[1]/(DIM_SIMPLEX*(DIM_SIMPLEX-1)/2))),
                                int(my_round(N*RATIO[2]/(DIM_SIMPLEX*(DIM_SIMPLEX-1)*(DIM_SIMPLEX-2)/6)))]
        training_sample_all =  sum([int(my_round(N*RATIO[0])),
                                int(my_round(N*RATIO[1])),
                                int(my_round(N*RATIO[2]))])

        s = convert_params_to_string(dimension_simplex=DIM_SIMPLEX,
                                     dimension_space=DIM_SPACE,
                                     degree=DEGREE,
                                     N=N,
                                     num_sample_train=training_sample_list,
                                     simplextype = TYPE,
                                     sigma=SIGMA)
        if opt_ratio_flag==1:
            yaml_dir = "../settings/"+s
        else:
             yaml_dir = "../settings_inductive_naive/"+s
        generate_yaml.generate_instance_yml(sigma=SIGMA,
                                            degree=DEGREE,
                                            dimension_simplex=DIM_SIMPLEX,
                                            dimension_space=DIM_SPACE,
                                            num_sample_train=training_sample_list,
                                            num_sample_train_all=N,
                                            simplextype=TYPE,
                                            num_trial=NUM_TRIAL,
                                            result_dir=yaml_dir)
        # run experiment
        if opt_ratio_flag==1:
            results_dir = "../results_synthetic_instances_borges_inductive_optimal/"+s
        else:
            results_dir = "../results_synthetic_instances_inductive_nonoptimal/"+s        seed_list = [i for i in range(NUM_TRIAL)]
        jobs = []
        for seed in seed_list:
            jobs.append(Process(target=main.main, args=(yaml_dir+"/"+str(seed)+".yml",
                                                        results_dir+"/"+str(seed),borges_flag)))
        print(jobs)
        print("multiprocessing start")
        start_ = time.time()
        for j in jobs[0:10]:
            j.start()
        for j in jobs[0:10]:
            j.join()
        for j in jobs[10:20]:
            j.start()
        for j in jobs[10:20]:
            j.join()
        elapsed_time = time.time()-start_
        print("multiprocessing finished, elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
    exp3d(n=1000,dimspace=25,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=25,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)
    exp3d(n=1000,dimspace=50,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=50,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=3,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=3,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=4,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=4,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=5,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=5,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=6,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=6,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=7,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=7,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=250,dimspace=100,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=250,dimspace=100,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=500,dimspace=100,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=500,dimspace=100,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=1000,dimspace=100,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=1000,dimspace=100,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)

    exp3d(n=2000,dimspace=100,dimsimplex=8,opt_ratio_flag=0,borges_flag=0)
    exp3d(n=2000,dimspace=100,dimsimplex=8,opt_ratio_flag=1,borges_flag=1)
