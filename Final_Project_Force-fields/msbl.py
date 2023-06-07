import numpy as np

def drift_x(x,y):
    """"double well potential"""
    return (x-x**3) -(x*y**2) -y

def drift_y(x,y):
    """"double well potential"""
    return (y-y*(x**2)) - y**3 + x

def diff_x(x,y):
    """"double well potential"""
    return 1 + y**2

def diff_y(x,y):
    """"double well potential"""
    return 1 + x**2

def pot_x(x,y):
    return 0.5*x**2 - 0.25*x**4 - (0.5*x**2)*y**2 -y*x

def pot_y(x,y):
    return 0.5*y**2 - 0.25*y**4 - (0.5*y**2)*x**2 -y*x
    
    
def trayectories(pos_init, time_step, noise, iterations, sets, example, L = 4, delta_u = 5, k = 3, gamma = 1, T=1, kb=1):
###create trayectories using a predefined drift and diffussion
    positions =[pos_init]
    time = 0
    time_list = [time]
    for j in range (0, sets,1):
        pos = pos_init  
        for i in range (1,iterations,1):
            if example == "double_well":
                update =  ((4*delta_u*pos/(L**2) - (4*delta_u*pos**3)/L**4))*time_step + np.sqrt(2*kb*T/gamma)*noise[i]
            if example == "harmonic_oscillator":
                update = ((-k/gamma)*pos)*time_step + np.sqrt(2*kb*T/gamma)*noise[i]
            pos = pos + update
            time = time + time_step
            positions.append(pos)
            time_list.append(time)
    
    return np.asarray(positions), np.asarray(time_list)

def trayectories_2d(pos_init, time_step, noise, iterations):
    pos_x = pos_init[0]
    pos_y = pos_init[1]
    positions_x = [pos_x]
    positions_y = [pos_y]
    time = 0
    time_list = [time]
    noise_x = noise[0]
    noise_y = noise[1]

    for i in range(1, iterations):
        update_x = drift_x(pos_x, pos_y)*time_step + np.sqrt(diff_x(pos_x, pos_y))*noise_x[i]
        update_y = drift_y(pos_x, pos_y)*time_step + np.sqrt(diff_y(pos_x, pos_y))*noise_y[i]
        pos_x = pos_x + update_x
        pos_y = pos_y + update_y
        time = time + time_step
        positions_x.append(pos_x)
        positions_y.append(pos_y)
        time_list.append(time)
    
    return np.asarray(positions_x), np.asarray(positions_y), np.asarray(time_list)

"""function for constructing trayectories with the obtained equations"""

def plotting_results(pos, array):
    cont = -1
    x= pos
    val= np.zeros((len(x), len(array)))

    for i in range (0, len(array)):
        cont = cont +1
        if array[i]!= 0:
            for k in range(0, len(x)):
                val[k,i] = array[i]*x[k]**cont
    
    return(val)

""""create libraries"""

#Generate a library for computing drift and difussion term ( 1, x, x**2, x**3x, ...)
def library2d(results_x,results_y, length):
    library = np.ones((1,len(results_x)))
    index_i =[]
    index_j =[]
    for i in range (0, length):
        for j in range(0,length):
            library_column = (results_x**i)*(results_y**j)
            library_column = library_column.reshape(1, len(results_x))
            library = np.concatenate((library, library_column), axis = 0)
            index_i.append(i)
            index_j.append(j)
    
    index = np.asarray(index_i), np.asarray(index_j)
    return library[1:], index

def library(results, length):
    library = np.ones((1,len(results)))
    for i in range (1, length):
        library_column = results**i
        library_column = library_column.reshape(1, len(results))
        library = np.concatenate((library, library_column), axis = 0)
       
    return library



#Compute expectation value for the drift
def output_vectors_dr(lib, dt):
    output = np.diff(lib)/dt
    return(np.asarray(output))

#Compute expectation value for the diffusion
def output_vectors_di(lib, dt, phi, dri):
    output = np.diff(lib)
    output = np.asarray(output)
    output = output.reshape(len(output))
    output = ((output- np.dot(phi,dri)*dt)**2)/dt
    return(output)

"""backbone"""

def msbl_pythonic(PHI, Y, max_iters):

    #default control parameters
    prune_gama = 1*10**-4 
    epsilon_val = 1*10**-8
    lambda_0 = 1
    Learn_Lambda = 1


    N, M = PHI.shape
    N, L = Y.shape

#generate initial values

    gamma = 0.5*np.ones(M) #generate values of gamma for each conlumn of the library (10)
    keep_list = np.arange(0,M) #create indexes for each column of the library
    m = len(keep_list) #count number of elements in keep_list
    mu = np.zeros((M, L))
    count = 0
    index=[]

    for i in range (0, max_iters):
        if min(gamma) < prune_gama:
            for j in range (0, len(gamma)):
                if gamma[j] > prune_gama:  #take indexes of elements less than 
                    index.append(j)
            gamma = gamma[index]
            PHI = PHI[:, index] #extract related lib elements
            keep_list = keep_list[index] #store indexes 
            m = len(gamma) #compute number of indexes taken
            index=[]

        if count == 1: 
            gamm = gamma
    
        mu_old = mu
        Gamma = np.diag(gamma) #create identity matrix with diagonal equal to gamma elements
        G = np.diag(np.sqrt(gamma))

    

        U, S, V = np.linalg.svd(np.dot(PHI,G), full_matrices = False)


    
        U = U.transpose()
        di = len(S)

        if di > 1:
            diag_S = S
            aa= np.diag(diag_S/(diag_S**2 + lambda_0 + 1e-16 ))

    
        else:
            diag_S = S[0]
            aa= diag_S/(diag_S**2 + lambda_0 + 1e-16 )

   

        Xi = np.dot(np.dot(np.dot(G,np.transpose(V)), aa ), U)
        mu = np.dot(Xi,Y)
    

        #update hyperparameters
        gamma_old = gamma
        mu2_bar = (abs(mu)**2)/L
        mu2_bar = mu2_bar.reshape(1, len(mu2_bar))


        multi = sum(np.multiply(np.transpose(Xi), np.dot(PHI, Gamma)))
    
        Sigma_w_diag = np.real(gamma - multi)
        gamma = Sigma_w_diag + mu2_bar
        gamma = gamma.reshape(len(gamma[0]))

        if Learn_Lambda == 1:
            den = N-m + np.sum(Sigma_w_diag/gamma_old)

            lambda_0 = (np.linalg.norm(Y-np.dot(PHI,mu), ord = "fro")**2/L)/den

        count = count +1

    mu = mu.reshape(len(mu))
    mu_final = np.zeros(M)
    mu_final[keep_list] = mu
    
    return mu_final
    

