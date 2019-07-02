import numpy as np  # type: ignore

class UniformSampling:
    def __init__(self,dimension):
        self.dimension = dimension # dimension of simplex
    # generate training points and evaluation points
    def simplex(self,num_sample,seed,dimension):
        '''
        num_sample: number of sample points
        dimension: dimension of simplex
        seed:
        return n\times d 2d ndarray
        '''
        #print(seed)
        np.random.seed(seed)
        x = np.random.uniform(0,1,[num_sample,dimension+1])
        x[:,0] = 0
        x[:,dimension] = 1
        x = np.sort(x,axis=1)
        z = np.zeros([num_sample, dimension])
        for col in range(dimension):
            z[:,col] = x[:,col+1] - x[:,col]
        return(z)
    def subsimplex(self, num_sample, seed, indices):
        z = self.simplex(dimension=len(indices),num_sample=num_sample,seed=seed)
        m = np.zeros([num_sample, self.dimension])
        for i in range(len(indices)):
            col = indices[i]
            m[:,col] = z[:,i]
        return(m)

class GridSampling:
    def __init__(self,dimension):
        self.dimension = dimension # dimension of simplex
    # generate training points and evaluation points
    def simplex(self,num_grid):
        t_list = np.linspace(0, 1, num_grid)
        tmp = np.array(np.meshgrid(*[t_list for i in range(self.dimension-1)]))
        m = np.zeros([tmp[0].ravel().shape[0], self.dimension])
        for i in range(self.dimension-1):
            m[:,i] = tmp[i].ravel()
        m[:,self.dimension-1] = 1 - np.sum(m,axis=1)
        return(m[m[:,-1]>=0,:])
