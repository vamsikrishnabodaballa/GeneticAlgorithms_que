'''
@author: piyush singh
'''
import array as ar
import numpy as np
import random
def obj_fn(X):
    # Without constraints Minimize
    # f(x1, x2) = (x1^2 + x2 -11)^2 + (x1 + x2^2 - 7)^2
    # Solution (3,2)
    # f = 0
    
    f = (1.5 - X[0] - X[0]*X[1])**2 + (2.25 - X[0] + X[0]*(X[1]**2))**2 + (2.625 - X[0] + X[0]*((X[1])**3))**2       #Part1
    # f =  ((X[0]**2) -X[1] -11)**2 + (X[0] + (X[1]**2) - 7 )**2       #Part2
    obj_val = f
    return obj_val

    # With constraints Minimize
    # f(x1, x2) = 1/3 * (x1 + 1)^3 + x2
    # subject to
    # g1(x1, x2) = 1 - x1 <= 0
    # g2(x1, x2) = -x2 <= 0
    # solution(1, 0)
    # f = 8/3 = 2.667
    
    # f = ((1/3) * ((X[0] + 1)**3)) + X[1]
    # g1 = 1 - X[0]
    # if g1 <= 0:
    #     g1 = 0
    # g2 = -X[1]  #<=0
    # if g2 <= 0:
    #     g2 = 0
         
    # obj_val = f +(10*g1) + (10*g2)
    # return obj_val

def fit_fun(obj_val):
    fit_val = 1/(1+((1.5)**obj_val))
    # fit_val = 1/(1+exp(obj_val))
    # fit_val = 1/(1+obj_val)
    return fit_val

def pop_string(string_length):
    X = [0]*string_length
    
    for i in range(len(X)):
        val = random.random()
        if val <= 0.5:
            X[i] = 1
    return X
    
def decode_individual(X):
    val = 0
    for i in range(len(X)):
        val = val + X[i]*(2**(len(X)-i-1))
    return val

def chromosome_pop_2_individual_pop(pop, string_lengths, upper_bounds, lower_bounds):
    individual_pop = np.zeros((pop_size, len(string_lengths)))
    
    for i in range(0,pop_size):
        start_ind = 0
        for j in range(0,len(string_lengths)):
            chromosome = pop[i]
            end_ind = string_lengths[j] + start_ind
            chromosome_segnment = chromosome[start_ind : end_ind]
            variable_val = decode_individual(chromosome_segnment)
            nb = string_lengths[j]
            
            actual = lower_bounds[j] + (((upper_bounds[j] - lower_bounds[j])*variable_val)/((2**nb)-1))
##########            print ('actualllllll=' ,actual)
            individual_pop[i,j] = actual
##########            print('chromosome_segment=', chromosome_segnment, 'j=', j, 'start_ind=', start_ind, 'end_ind=', end_ind)
##########            print('string=', str(pop[i]), 'individual=', individual_pop)
            start_ind = end_ind
    return individual_pop

def create_mating_pool(pop, string_lengths,cum_sum):
    mating_pool = np.zeros((pop_size, sum(string_lengths)))
    # needle = [0.1,0.3, 0.2, 0.5, 0.9]
    for i in range (0, pop_size):
        needle = random.random()
        for j in range(0, pop_size):
            if needle<=cum_sum[0]:
                mating_pool[i] = pop[0]
#############                print('in j=1', j)
            if needle>=cum_sum[j] and needle<=cum_sum[j+1]:
                mating_pool[i] = pop[j+1]
#############                print('in j<pop_size:i=', j)
    return mating_pool
                
def crossover_in_mating_pool(mating_pool, pop_size, string_lengths, crossover_prob):
    for i in range(0, int(pop_size),2):
        R = random.random()
        if R <= crossover_prob:
            CP = random.randint(0, sum(string_lengths)-1)
            org_ch_segment = mating_pool[i,CP]
            mating_pool[i,CP] = mating_pool[i+1, CP]
            mating_pool[i+1,CP] = org_ch_segment
    return mating_pool

def mutation_in_mating_pool(mating_pool, pop_size, string_lengths, mutation_prob):
    for i in range(0, int(pop_size),1):
        R = random.random()
        if R <= mutation_prob:
            CP = random.randint(0, sum(string_lengths)-1)
            mating_pool[i,CP] = random.randint(0,1)
    return mating_pool                
    
# ----------------------
# Program
# ----------------------

#OBJECTIVE AND FITNESS FUNCTION VALUES
# X = ar.array('d',[1,5])
# obj_fn_val = obj_fn(X)
# fit_fn_val = fit_fun(obj_fn_val)
# print('obj_fn_val', obj_fn_val, 'fit_fn_val', fit_fn_val)

#GENERATE POPULATION STRING FOR THE GIVEN STRING LENGTHS
# no_of_variables = 2
# string_lengths = [5,6]
# M = sum(string_lengths)
# string = pop_string(M)
# print(string)

#CONVERT POPULATION STRING TO DECIMAL
# ind_test = decode_individual(string)
# print(ind_test)

#GENERATE POPULATION OF GIVEN SIZE
# pop_size = 4
# N = pop_size
# M = sum(string_lengths) 
# pop = [[0]*M]*N

# for i in range(0, pop_size):
#     pop[i] = pop_string(M)

# print(pop)

#To find individual corresponding to each chromosome
# individual = [[0]*no_of_variables]*N
# for i in range(0, pop_size):
#     individual[i] = decode_individual(pop[i])

#SURVIVAL OF FITTEST
no_of_variables = 2
string_lengths = [10,10]
upper_bounds = [5,2]        #Part 1
lower_bounds = [-5,-2]       #Part1
# upper_bounds = [4,4]       #Part2
# lower_bounds = [-4,-4]       #Part2
pop_size = 100
N = pop_size
M = sum(string_lengths)
crossover_prob = 0.8
mutation_prob = 0.1

##
###################print(M,N)
pop = [[0]*M]*N
individual = [[0]*no_of_variables]*N

## Set of all individual_pop (population). Each individual is composed of several variables
ind_obj_val = np.zeros(pop_size)
ind_fit_val = np.zeros(pop_size)
ind_selection_prob = np.zeros(pop_size)
cum_sum = np.zeros(pop_size)
##GENERATING POPULATION OF INDIVIDUALS IN CHROMOSOME FORM
for i in range(0, pop_size):
    pop[i] = pop_string(sum(string_lengths))
# pop = [[1,1,1,0,0,0,1,1,0,0],[0,1,0,1,0,1,0,1,0,1],[0,0,0,1,1,1,0,0,1,1],[1,1,0,0,1,1,0,0,0,1],[1,0,1,0,1,1,0,1,0,1]]
#WHILE LOOP STARTS HERE
max_generations = 1000
Generation_count = 0
while(Generation_count < max_generations):

    individual_pop = chromosome_pop_2_individual_pop(pop, string_lengths, upper_bounds, lower_bounds)
##################    print('after function call', individual_pop)
    
    for i in range(0, pop_size):
        ind_obj_val[i] = obj_fn(individual_pop[i,:])
        ind_fit_val[i] = fit_fun(ind_obj_val[i])
    
    # print('ind_obj_val= ',ind_obj_val, 'ind_fit_val = ', ind_fit_val)
    best_val = np.min(ind_obj_val, axis = 0)
    best_val_index = np.where(best_val == np.amin(best_val))
    best_individual = individual_pop[best_val_index[0]]
    
    
    #MATING POOL
    sum_fitness = np.sum(ind_fit_val)
    cum_sum = np.zeros(pop_size)
    ind_selection_prob = np.zeros(pop_size)
    for i in range(0, pop_size):
        ind_selection_prob[i] = ind_fit_val[i]/sum_fitness
        cum_sum[i] = np.sum(ind_selection_prob)
    
    # #CREATING MATING POOL
    print('ind_selection_prob', ind_selection_prob)
    mating_pool = create_mating_pool(pop, string_lengths, cum_sum)
    print('create mating pool', mating_pool)
    # print(cum_sum)
    # print('selected individuals in mating pool after selection')
    # print(mating_pool[4])
    # print(mating_pool[5])
    # print(mating_pool[6])
    # print(mating_pool[7])
    
    #CROSSOVER IN MATING POOL
    mating_pool = crossover_in_mating_pool(mating_pool, pop_size, string_lengths, crossover_prob)
    # print('selected individuals in mating pool after crossover')
    # print(mating_pool[4])
    # print(mating_pool[5])
    # print(mating_pool[6])
    # print(mating_pool[7])
    
    #MUTATION IN MATING POOL
    mating_pool = mutation_in_mating_pool(mating_pool, pop_size, string_lengths, mutation_prob)
    # print('selected individuals in mating pool after mutation')
    # print(mating_pool[4])
    # print(mating_pool[5])
    # print(mating_pool[6])
    # print(mating_pool[7])
    
    # mating_pool = create_mating_pool(pop, string_lengths, cum_sum)
    # mating_pool = crossover_in_mating_pool(mating_pool, pop_size, string_lengths, crossover_prob)
    # mating_pool = mutation_in_mating_pool(mating_pool, pop_size, string_lengths, mutation_prob)
    pop = mating_pool
    Generation_count = Generation_count + 1
    #WHILE LOOP ENDS

print('individual_pop', individual_pop)
print('best_val', best_val)
print('best_val_index' ,best_val_index)
print('best_individual', best_individual)
 












