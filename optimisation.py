import numpy as np
import scipy
import copy
from random import random
############### CONSTANTS #################
# General
LIMIT = 512
DIMENSIONS = 2
RANDOM_SEED = 0
MAX_NUM_EVAL = 10000

# Simulated Annealing - Initialisation Parameters
TEMP_INIT = 1.0
TEMP_MIN = 0.00001
ALPHA = 0.9

# Genetic Algorithm
POPULATION_SIZE = 300
NUM_CHILD = 4
CHANCE_MUTATION = 0.1
FIT = 100
LUCK = 50
NUM_GEN = 50


def eggholder(x):
    try:
        x_shape = np.shape(x)
        assert len(x_shape)>=2
        assert len(x_shape)<=3
        if len(x_shape)==2:
            try:
                assert x_shape[0]>1
            except:
                print ("x must be nxm numpy array, where n>1")
        else:
            try:
                assert x_shape[1]>1
            except:
                print ("x must be pxnxm numpy array, where n>1")            
    except:
        print ("x must be nxm or pxnxm numpy array")
    
    if len(x_shape)==2:
        # placeholders
        #funct = 0
        x1 = x[:-1]
        x2 = x[1:]
    else:
        #funct = np.zeros((np.shape(x)[0],1))
        x1 = x[:,:-1,:]
        x2 = x[:,1:,:]
        
    term1 = np.abs(x2+0.5*x1+47)
    term2 = -(x2+47)*np.sin(np.sqrt(term1))
    term3 = np.abs(x1-x2-47)
    term4 = -x1*np.sin(np.sqrt(term3))
    #print (np.shape(term2))
    #print (np.shape(funct))
    funct = term2+term4
    
    if len(x_shape)==2:
        return np.sum(funct)
    else:
        return np.sum(funct, axis=1)

#################### Simulated Annealing Functions ########################
def cost(solution):
    return eggholder(solution)

def acceptance_probability(old_cost, new_cost, T):
    ap = np.exp((old_cost-new_cost)/T)
    return ap

def neighbor(solution, lim=LIMIT):
    solution_max = lim
    solution_min = -lim
    jump_range = solution_max-solution_min
    new_solution = np.random.rand(*np.shape(solution))
    new_solution = solution + new_solution*(jump_range*2) - jump_range
    new_solution = np.clip(new_solution, solution_min, solution_max)
    return new_solution

def anneal(dim=DIMENSIONS, lim=LIMIT, rseed=RANDOM_SEED, 
           T_init = TEMP_INIT, T_min=TEMP_MIN, alpha=ALPHA, 
           max_num_eval=MAX_NUM_EVAL, disp=True):
    np.random.seed(rseed)
    solution = np.random.rand(*(dim,1))*2*lim - lim
    solution = np.clip(solution, -lim, lim)
    old_cost = cost(solution)
    T = T_init
    num_eval = 1
    while T > T_min and num_eval<max_num_eval:
        i = 1
        while i <= 100:
            new_solution = neighbor(solution)
            new_cost = cost(new_solution)
            num_eval = num_eval+1
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                solution = new_solution
                old_cost = new_cost
            i += 1
        T = T*alpha
    if disp:
        if num_eval >= max_num_eval:
            print ("Max number of evaluations reached: "+str(num_eval))
        if T <= T_min:
            print ("Minimum Temperature reached:       "+str(T))
    return solution, old_cost

############# Genetic Algorithm ###############

def genetic_fit(population):
    #print (eggholder2(population))
    return eggholder(population)

def select_parents(fitness, fit, luck):
    N = fit # Fittest
    M = luck  # Lucky few
    sorted_fit = np.argsort(fitness, axis=0)
    fittest = sorted_fit[:N]
    lucky = np.array([np.random.choice(sorted_fit[N:,0], M, replace=False)]).T
    parents_id = np.append(fittest, lucky, axis=0)
    return parents_id

def create_children(parents, num_child, fit, luck):
    N = fit   # N number of the fittest
    M = luck  # M number of luckiest
    fathers = np.tile(parents[:int((N+M)/2)], [num_child,1,1])
    mothers = np.tile(parents[int(-(N+M)/2):], [num_child,1,1])
    gene_mask = np.random.choice([0, 1], size=np.shape(fathers), p=[0.5,0.5])

    new_population = fathers*gene_mask + mothers*(1-gene_mask)
    return new_population

def mutate_population(new_population, chance_mutation):
    new_generation = copy.copy(new_population)
    mutate_mask = np.random.choice([0, 1], size=np.shape(new_generation), p=[1-chance_mutation, chance_mutation])
    mutation = np.random.rand(*np.shape(new_population))*1024-512
    mutation = np.clip(mutation, -512, 512)
    new_generation[mutate_mask==1] = mutation[mutate_mask==1]
    return new_generation

def create_generation(old_generation, num_child, chance_mutation, fit, luck):
    fitness = genetic_fit(old_generation)
    parents_id = select_parents(fitness, fit, luck)
    parents = old_generation[parents_id[:,0]]
    new_population = create_children(parents, num_child, fit, luck)
    new_generation = mutate_population(new_population, chance_mutation)
    return new_generation, fitness

def genetic_alg(dim=DIMENSIONS, popsize=POPULATION_SIZE, num_child=NUM_CHILD, 
                chance_mutation=CHANCE_MUTATION, fit=FIT, luck=LUCK, num_gen=NUM_GEN, 
                rseed=RANDOM_SEED, log_cost=False):
    # Initialise population
    np.random.seed(rseed)
    old_generation = np.random.rand(*(popsize, dim, 1))*1024-512
    old_generation = np.clip(old_generation, -512, 512)
    gen = 0
    if log_cost:
        mean_cost_store = np.zeros((num_gen,1))
        mini_cost_store = np.zeros((num_gen,1))
    # Create 50 new generations
    while gen<num_gen:
        new_generation, old_fitness = create_generation(old_generation, num_child, chance_mutation, fit, luck)
        old_generation = new_generation
        
        if log_cost:
            mean_cost_store[gen] = np.mean(old_fitness)
            mini_cost_store[gen] = np.min(old_fitness)
        
        gen = gen+1
    if log_cost:
        return old_generation, mean_cost_store, mini_cost_store
    else:
        return old_generation