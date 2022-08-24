import numpy as np
import random
import copy



def add_random_edge(individual, d, w, char):

    # Choose which layer to connect to (prev, curr, next)
    layer = random.sample([-1, 0, 1], 1)[0]

    # If in first/last layer then can't connect to prev/next, resp.
    if layer == -1 and d == 0:
        layer = 1
    elif layer == 1 and d == individual['depth'] - 1:
        layer = -1

    # If connecting to prev or next layer, sample the node to connect to
    if layer == -1 or layer == 1:
        node = random.randint(0, individual['width'] - 1)
        individual['fa'][d][w][char] = [layer, node]

    # Otherwise add self loop
    else:
        individual['fa'][d][w][char] = [0, w]
    return individual



def initialize_fa(min_depth, max_init_depth, width):
    '''
    Initializes a single finite automata
    '''
    alphabet = ['a', 'b']

    individual = {'depth' : random.randint(min_depth, max_init_depth), 'width' : width}
    individual['start'] = random.randint(0, individual['width'] - 1)
    individual['fa'] = [[dict() for _ in range(individual['width'])] for _ in range(individual['depth'])]

    for d in range(individual['depth']):
        for w in range(individual['width']):
            for char in alphabet:
                individual = add_random_edge(individual, d, w, char)
    
    return individual



def initialize_pop(pop_size, min_depth, max_init_depth, width):
    '''
    Initialize the full population
    '''
    population = []
    for _ in range(pop_size):
        population.append(initialize_fa(min_depth, max_init_depth, width))
    return population



def evaluate_fa(fa, dataset):
    '''
    Evaluates a single FA on the full dataset
    One point for each word that passes
    '''
    fitness = 0
    for word in dataset:

        d = 0
        w = fa['start']
        for char in word:
            diff, w = fa['fa'][d][w][char]
            d += diff

        # If we ended in the final layer than the test case passes
        if d == fa['depth'] - 1:
            fitness += 1

    return fitness



def evaluate_fitness(population, pos_data, neg_data):
    '''
    Get fitness for each individual in the population
    '''
    fitness = []
    for individual in population:
        fitness.append(evaluate_fa(individual, pos_data) - evaluate_fa(individual, neg_data))
    return fitness



def return_active(fa):
    '''
    Returns a list of all effective nodes
    Equivalently, removes intron
    '''
    d, w = 0, fa['start']
    nodes = []
    queue = [[d, w]]

    while len(queue) != 0:

        curr = queue[0]
        queue = queue[1:]   # Dequeues

        if curr in nodes:
            continue
        else:
            d, w = curr
            nodes.append([d, w])
            node = fa['fa'][d][w]

            for char in node.keys():
                diff, new_w = node[char]
                new_d = d + diff
                if [new_d, new_w] not in queue and [new_d, new_w] not in nodes:
                    queue.append([new_d, new_w])

    return nodes



def micro_mutate(individual, micro_mutate_rate):
    '''
    For every edge in the individual, randomly changes which node it's ingoing to
    '''
    for d, row in enumerate(individual['fa']):
        for w, node in enumerate(row):
            for char in node.keys():
                if np.random.binomial(1, 1 - micro_mutate_rate):
                    individual = add_random_edge(individual, d, w, char)                    

    return individual



def macro_mutate(individual, macro_mutate_rate, min_depth, max_depth):
    '''
    For each individual, there's a chance that either a row will be removed or randomly inserted
    '''
    if np.random.binomial(1, 1 - macro_mutate_rate):

        add, remove = False, False
        if individual['depth'] == min_depth:
            add = True
        elif individual['depth'] == max_depth:
            remove = True
        
        # Will always add if individual is the minimum depth, will never add if it's maximum depth
        if (random.randint(0, 1) or add) and not remove:
            row = random.randint(1, individual['depth'] - 1)
            add = True
        else:
            row = random.randint(1, individual['depth'] - 2)
            remove = True

        if add:
            # Add in a copy of the first row (edges will be rewritten, just need all characters)
            individual['fa'].insert(row, copy.deepcopy(individual['fa'][0]))
            for w, node in enumerate(individual['fa'][row]):
                for char in node.keys():
                    individual = add_random_edge(individual, row, w, char)
            individual['depth'] += 1

        elif remove:
            individual['fa'].pop(row)
            individual['depth'] -= 1
        
    return individual



def crossover(parent1, parent2, crossover_rate, max_cross_dist, min_len, max_len):
    '''
    Performs one-point crossover between two individuals (based on depth)
    '''
    # No crossover with probability (1 - crossover_rate), return parents if case
    if np.random.binomial(1, 1 - crossover_rate):
        return parent1, parent2

    # Randomly generate crossover points, making sure they're valid and no more than max_cross_dist indices apart
    upper = min(len(parent1['fa']) - 1, len(parent2['fa']) - 1)
    cross_point1 = random.randint(1, upper)

    lower = max(1, cross_point1 - max_cross_dist)
    upper = min(len(parent2['fa']) - 1, cross_point1 + max_cross_dist)
    cross_point2 = random.randint(lower, upper)

    # Concatenate parents to create offspring
    offspring1 = copy.deepcopy(parent1)
    offspring1['fa'] = offspring1['fa'][:cross_point1] + copy.deepcopy(parent2['fa'][cross_point2:])
    offspring1['depth'] = len(offspring1['fa'])

    offspring2 = copy.deepcopy(parent2)
    offspring2['fa'] = offspring2['fa'][:cross_point2] + copy.deepcopy(parent1['fa'][cross_point1:])
    offspring2['depth'] = len(offspring2['fa'])

    # If the offspring lengths are invlaid return the parent instead
    if len(offspring1['fa']) < min_len or len(offspring1['fa']) > max_len:
        offspring1 = copy.deepcopy(parent1)
    if len(offspring2['fa']) < min_len or len(offspring2['fa']) > max_len:
        offspring2 = copy.deepcopy(parent2)

    return offspring1, offspring2



def tournament(population, fitness, args, pos_data, neg_data, best_individual):
    '''
    Tournament parent selection, generates offspring and returns new population, fitnesses, and the best individual
    '''
    # Sample two tournaments, get fitnesses
    parents = random.sample(range(len(population)), 2 * args['tournament_size'])
    tourn1, tourn2 = parents[:args['tournament_size']], parents[args['tournament_size']:]
    tourn1_fit, tourn2_fit = [fitness[ind] for ind in tourn1], [fitness[ind] for ind in tourn2]

    tourn1_winner = population[tourn1[np.argmax(tourn1_fit)]]
    tourn2_winner = population[tourn2[np.argmax(tourn2_fit)]]

    # Losers can be from either tournament
    loser1_ind, loser2_ind = np.argsort(tourn1_fit + tourn2_fit)[[0, 1]]

    # Perform crossover, then micro and macro mutations
    offspring1, offspring2 = crossover(tourn1_winner, tourn2_winner, args['crossover_rate'], 
                                              args['max_cross_dist'], args['min_depth'], args['max_depth'])

    offspring1 = micro_mutate(offspring1, args['micro_mutate_rate'])
    offspring2 = micro_mutate(offspring2, args['micro_mutate_rate'])

    offspring1 = macro_mutate(offspring1, args['macro_mutate_rate'], args['min_depth'], args['max_depth'])
    offspring2 = macro_mutate(offspring2, args['macro_mutate_rate'], args['min_depth'], args['max_depth'])

    population[loser1_ind], population[loser2_ind] = offspring1, offspring2

    o1_fitness = evaluate_fa(offspring1, pos_data) - evaluate_fa(offspring1, neg_data)
    o2_fitness = evaluate_fa(offspring2, pos_data) - evaluate_fa(offspring2, neg_data)

    if best_individual is not None:
        best_fitness = evaluate_fa(best_individual, pos_data) - evaluate_fa(best_individual, neg_data)
    else:
        best_fitness = -10000

    fitness[loser1_ind], fitness[loser2_ind] = o1_fitness, o2_fitness

    # Check if new best individual
    # If the fitness is the same as the best individual, take the smallest
    if o1_fitness >= best_fitness:
        if o1_fitness == best_fitness and len(return_active(offspring1)) < len(return_active(best_individual)):
            best_individual = offspring1
        elif o1_fitness > best_fitness:
            best_individual = offspring1

    if o2_fitness >= best_fitness:
        if o2_fitness == best_fitness and len(return_active(offspring2)) < len(return_active(best_individual)):
            best_individual = offspring2
        elif o2_fitness > best_fitness:
            best_individual = offspring2

    return population, fitness, best_individual



def print_fa(fa):
    '''
    Prints FA after removing intron
    '''
    to_print = return_active(fa)
    print("Start: (0, %d)" % (fa['start']))
    for i, row in enumerate(fa['fa']):
        for j, node in enumerate(row):
            if [i, j] in to_print:
                print("Node (%d, %d):" % (i, j), end='')
                for key, item in node.items():
                    print("  %s -> (%d, %d)" % (key, i + item[0], item[1]), end='')
                print()