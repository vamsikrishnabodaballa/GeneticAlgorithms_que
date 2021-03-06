# GeneticAlgorithms_que

Above python codes are the solutions for two problems solved using genetic algorithms.

Normally, for solving an unknown or a new optimization problem, at the outset, we do not have any idea of GA parameters, such as population size and number of generations (for which the problem has to run). Following procedure may be used for solving an unknown problem. The GA programs are run multiple number of times, by varying the population size and number of generations. These variations are done in multiples of 10. For example you may vary population size as; 20, 200, 2000 and so on. 
When the output (solution to the problem) of the program starts giving almost the same or similar results, the corresponding combination is used for solving the optimization problem finally. 
With the parameters determined as above, the GA program is run afresh for 5 (or more) number of times to ensure that the program returns the same value of the objective function in most of the cases, if not all. 

(a) Minimize f(x,y) = (1.5 – x – x*y)^2 + (2.25 – x + x*(y^2) )^2 + (2.625 – x + x*(y^3) )^2 
    Subject to following bound constraints; 
      # x >= -5; 
      # x <= 5; 
      # y >= -2; 
      # y <= 2; 
    The accuracy must be more than two decimal places.
