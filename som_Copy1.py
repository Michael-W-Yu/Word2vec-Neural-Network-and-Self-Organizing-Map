'''som.py
2D self-organizing map
CS343: Neural Networks
YOUR NAMES HERE
Project 5: Word Embeddings and SOMs
'''
import numpy as np


def lin2sub(ind, the_shape):
    '''Utility function that takes a linear index and converts it to subscripts.
    No changes necessary here.
    Parameters:
    ----------
    ind: int. Linear index to convert to subscripts
    the_shape: tuple. Shape of the ndarray from which `ind` was taken.
    Returns:
    ----------
    tuple of subscripts
    Example: ind=2, the_shape=(2,2) -> return (1, 0).
        i.e. [[_, _], [->SUBSCRIPT OF THIS ELEMENT<-, _]]
    '''
    return np.unravel_index(ind, the_shape)


class SOM:
    '''A 2D self-organzing map (SOM) neural network.
    '''
    def __init__(self, map_sz, n_features, max_iter, init_lr=0.2, init_sigma=10.0, verbose=False):
        '''Creates a new SOM with random weights in range [-1, 1]
        Parameters:
        ----------
        map_sz: int. Number of units in each dimension of the SOM.
            e.g. map_sz=9 -> the SOM will have 9x9=81 units arranged in a 9x9 grid
            n_features: int. Number of features in a SINGLE data sample feature vector.
        max_iter: int. Number of training iterations to do
        init_lr: float. INITIAL learning rate during learning. This will decay with time
            (iteration number). The effective learning rate will only equal this if t=0.
        init_sigma: float. INITIAL standard deviation of Gaussian neighborhood in which units
            cooperate during learning. This will decay with time (iteration number).
            The effective learning rate will only equal this if t=0.
        verbose: boolean. Whether to print out debug information at various stages of the algorithm.
            NOTE: if verbose=False, nothing should print out when running methods.
        TODO:
        - Initialize weights (self.wts) to standard normal random values (mu=0, sigma=1)
            Shape=(map_sz, map_sz, n_features).
            Weights should be normalized so that the L^2 norm (Euclidean distance) of EACH som
            unit's weight vector is 1.0
        - Initialize self.bmu_neighborhood_x and self.bmu_neighborhood_y to EACH be a 2D grid of
        (x,y) index values (i.e. x,y positions in the 2D grid), respectively, in the range 0,...,map_sz-1.
        shape of self.bmu_neighborhood_x: (map_sz, map_sz)
        shape of self.bmu_neighborhood_y: (map_sz, map_sz)
        Together, cooresponding values at each position in each array is an ordered pair of SOM unit
        (x,y) positions.
        '''
        self.n_features = n_features
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.init_sigma = init_sigma
        self.verbose = verbose

        self.wts = np.random.normal(loc=0, scale=1, size=(map_sz, map_sz, n_features))
        self.wts = self.wts/(np.sqrt(np.sum(self.wts*self.wts, axis = 2, keepdims =  True)))

        self.bmu_neighborhood_x, self.bmu_neighborhood_y = np.meshgrid(np.linspace(0,map_sz-1,map_sz, dtype = np.dtype(np.int16)), np.linspace(0,map_sz-1,map_sz, dtype = np.dtype(np.int16)))


        pass

    def get_wts(self):
        '''Returns a COPY of the weight vector.
        No changes necessary here.
        '''
        return self.wts.copy()

    def compute_decayed_param(self, t, param):
        '''Takes a hyperparameter (e.g. lr, sigma) and applies a time-dependent decay function.
        Parameters:
        ----------
        t: int. Current (training) time step.
        param: float. Parameter (e.g. lr, sigma) whose value we will decay.
        Returns:
        ----------
        float. The decayed parameter at time t
        TODO:
        - See notebook for decay equation to implement
        '''
        for i in range(len(param)):
            param[i] = param[i] * np.exp(-t/(self.max_iter/2))
        return param

    def gaussian(self, bmu_rc, sigma, lr):
        '''Generates a normalized 2D Gaussian, weighted by the the current learning rate, centered
        on `bmu_rc`.
        Parameters:
        ----------
        bmu_rc: tuple. x,y coordinates in the SOM grid of current best-matching unit (BMU).
            NOTE: bmu_rc is arranged row,col, which is y,x.
        sigma: float. Standard deviation of the Gaussian at the current training iteration.
            The parameter passed in is already decayed.
        lr: float. Learning rate at the current training iteration.
            The parameter passed in is already decayed.
        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). 2D Gaussian, weighted by the the current learning rate.
        TODO:
        - Evaluate a Gaussian on a 2D grid with shape=(map_sz, map_sz) centered on `bmu_rc`.
        - Normalize so that the maximum value in the kernel is `lr`
        '''
        l2 = np.square(self.bmu_neighborhood_x - bmu_rc[1])+np.square(self.bmu_neighborhood_y - bmu_rc[0])
        gauss = lr*np.exp(-l2/(2*np.square(sigma)))
        return gauss

    def fit(self, train_data):
        '''Main training method
        Parameters:
        ----------
        train_data: ndarray. shape=(N, n_features) for N data samples.
        TODO:
        - Shuffle a COPY of the data samples (don't modify the original data passed in).
        - On each training iteration, select a data vector.
            - Compute the BMU, then update the weights of the BMU and its neighbors.
        NOTE: If self.max_iter > N, and the current iter > N, cycle back around and do another
        pass thru each training sample.
        '''

        data = train_data.copy()
        # np.random.shuffle(data)
        # np.random.shuffle(data)

        if self.verbose:
            print(f'Starting training...')

        for t in range(self.max_iter):
           # print(t)
            np.random.shuffle(data)
            index = t%(data.shape[0])
            bmu = self.get_bmu(data[index])
            self.update_wts(t, data[index], bmu)


        if self.verbose:
            print(f'Finished training.')

    def get_bmu(self, input_vector):
        '''Compute the best matching unit (BMU) given an input data vector.
        Uses Euclidean distance (L2 norm) as the distance metric.
        Parameters:
        ----------
        input_vector: ndarray. shape=(n_features,). One data sample vector.
        Returns:
        ----------
        tuple of (x,y) position of the BMU in the SOM grid.
        TODO:
        - Find the unit with the closest weights to the data vector. Return its subscripted position.
        '''
        min_l2 = None
        min_x = 0
        min_y = 0
        for x in range(self.bmu_neighborhood_x.shape[0]):
            for y in range(self.bmu_neighborhood_y.shape[1]):
                l2 = np.sum(np.square(self.wts[x][y]-input_vector))
                if min_l2 == None or min_l2 > l2:
                    min_l2 = l2
                    min_x = x
                    min_y = y
        return (min_x, min_y)


    def update_wts(self, t, input_vector, bmu_rc):
        '''Applies the SOM update rule to change the BMU (and neighboring units') weights,
        bringing them all closer to the data vector (cooperative learning).
        Parameters:
        ----------
        t: int. Current training iteration.
        input_vector: ndarray. shape=(n_features,). One data sample.
        bmu_rc: tuple. BMU (x,y) position in the SOM grid.
        Returns:
        ----------
        None
        TODO:
        - Decay the learning rate and Gaussian neighborhood standard deviation parameters.
        - Apply the SOM weight update rule. See notebook for equation.
        '''
        param = self.compute_decayed_param(t, [self.init_lr, self.init_sigma])
        sz = self.bmu_neighborhood_x.shape[0]

        for x in range(sz):
            for y in range(sz):
                self.wts[x][y] = self.wts[x][y] + self.gaussian(bmu_rc, param[1], param[0])[x][y]*(input_vector-self.wts[x][y])

    def error(self, data):
        '''Computes the quantization error: total error incurred by approximating all data vectors
        with the weight vector of the BMU.
        Parameters:
        ----------
        data: ndarray. shape=(N, n_features) for N data samples.
        Returns:
        ----------
        float. Average error over N data vectors
        TODO:
        - Progressively average the Euclidean distance between each data vector
        and the BMU weight vector.
        '''
        sum = 0
        for N in range(data.shape[0]):
            (x,y) = self.get_bmu(data[N])
            sum += np.sqrt(np.sum(np.square(data[N]-self.wts[x][y])))
        return sum/N

    def u_matrix(self):
        '''Compute U-matrix, the distance each SOM unit wt and that of its 8 local neighbors.
        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). Total Euclidan distance between each SOM unit
            and its 8 neighbors.
        TODO:
        - Compute the U-matrix
        - Normalize it so that the dynamic range of values span [0, 1]
        '''
        sz = self.bmu_neighborhood_x.shape[0]
        u_matrix = np.zeros((sz,sz))
        diff = [-1, 0, 1]
        for x in range(sz):
            for y in range(sz):
                sum = 0
                for dx in diff:
                    for dy in diff:
                        if not(dx == 0 and dy == 0):
                            x_neighbor = x + dx
                            y_neighbor = y + dy
                            if x_neighbor >=0 and x_neighbor < sz and y_neighbor >=0 and y_neighbor < sz:
                                sum += np.sqrt(np.sum(np.square(self.wts[x][y]-self.wts[x_neighbor][y_neighbor])))
                u_matrix[x][y] = sum
        return u_matrix/np.max(u_matrix)
        #return (u_matrix-np.min(u_matrix))/(np.max(u_matrix)-np.min(u_matrix))


    def get_nearest_wts(self, data):
        '''Find the nearest SOM wt vector to each of data sample vectors.
        Parameters:
        ----------
        data: ndarray. shape=(N, n_features) for N data samples.
        Returns:
        ----------
        ndarray. shape=(N, n_features). The most similar weight vector for each data sample vector.
        TODO:
        - Compute and return the array of closest wts vectors to each of the input vectors.
        '''
        near_wts = np.zeros(data.shape)
        for i in range(data.shape[0]):
            (x,y) = self.get_bmu(data[i])
            near_wts[i] = self.wts[x][y]
        return near_wts