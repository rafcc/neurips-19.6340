import main
import generate_yaml
import math

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

#N_list = [1000]
DEGREE = 3
SIGMA = 0.1

OPT_RATIO_DIC = {3:[1-0.587-0.314,0.587,0.314],
                 4:[1-0.453-0.481,0.453,0.481],
                 5:[1-0.386-0.547,0.386,0.547],
                 6:[1-0.365-0.577,0.365,0.577],
                 7:[1-0.359-0.589,0.359,0.589],
                 8:[1-0.336-0.623,0.336,0.623]}

N_list = [1000, 2500,5000,7500,10000]
DIM_LIST = [3  4, ,5 ,6,7]
TYPE_LIST =["linear","squareroot"]#,"linear"]

for dim in DIM_LIST:
    DIM_SIMPLEX = dim
    DIM_SPACE = dim
    RATIO = OPT_RATIO_DIC[DIM_SIMPLEX]

    for N in N_list:
        for TYPE in TYPE_LIST:
            training_sample_list = [math.ceil(N*RATIO[0]/DIM_SIMPLEX),
                                    math.ceil(N*RATIO[1]/(DIM_SPACE*(DIM_SIMPLEX-1)/2)),
                                    math.ceil(N*RATIO[2]/(DIM_SPACE*(DIM_SIMPLEX-1)*(DIM_SIMPLEX-2)/6))]
            training_sample_all =  sum([math.ceil(N*RATIO[0]),
                                    math.ceil(N*RATIO[1]),
                                    math.ceil(N*RATIO[2])])

            s = convert_params_to_string(dimension_simplex=DIM_SIMPLEX,
                                         dimension_space=DIM_SPACE,
                                         degree=DEGREE,
                                         N=N,
                                         num_sample_train=training_sample_list,
                                         simplextype = TYPE,
                                         sigma=SIGMA)
            yaml_dir = "../settings/"+s

            generate_yaml.generate_instance_yml(sigma=SIGMA,
                                                degree=DEGREE,
                                                dimension_simplex=DIM_SIMPLEX,
                                                dimension_space=DIM_SPACE,
                                                num_sample_train=training_sample_list,
                                                num_sample_train_all=N,
                                                simplextype=TYPE,
                                                result_dir=yaml_dir)
            """
            results_dir = "../results/"+t+"_"+s
            seed_list = [i for i in range(10)]
            for seed in seed_list:
                main.main(ymlfilename=yaml_dir+"/"+str(seed)+".yml",
                          resultdir=results_dir+"/"+str(seed),
                          simplextype=t)
            """
