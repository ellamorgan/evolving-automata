import data
import ea
import numpy as np
import time


def train_ea(args):

    # Generate valid and invalid examples
    pos_data = data.generate_dataset(args['pos_reg'], args['n_data'], args['max_repeat'], args['train_split'])
    neg_data = data.generate_dataset(args['neg_reg'], args['n_data'], args['max_repeat'], args['train_split'])

    population = ea.initialize_pop(args['pop_size'], args['min_depth'], args['max_init_depth'], args['width'])
    fitness = ea.evaluate_fitness(population, pos_data['train'], neg_data['train'])

    best_individual = None

    # Training loop
    for i in range(args['ea_epochs']):

        population, fitness, best_individual = ea.tournament(population, fitness, args, pos_data['train'], 
                                                             neg_data['train'], best_individual)

        if (i + 1) % args['print_every'] == 0:
            print("Epoch: %d" % (i + 1))

            best_fitness = ea.evaluate_fa(best_individual, pos_data['train']) - ea.evaluate_fa(best_individual, neg_data['train'])
            print("Fitness: %d" % (best_fitness))
            print("Len: %d" % (len(pos_data['train'])))
            print("Best training accuracy: %.1f%%" % (100 * best_fitness / len(pos_data['train'])))
            ea.print_fa(best_individual)
            print()
    
    best_fitness = ea.evaluate_fa(best_individual, pos_data['test']) - ea.evaluate_fa(best_individual, neg_data['test'])
    final_fitness = 100 * best_fitness / len(pos_data['test'])
    print("Final fitness of EA on test set: %.1f%%" % (final_fitness))
    return final_fitness


if __name__ == '__main__':

    # Load in arguments from configs.conf
    args = data.load_args('configs.conf')

    ea_accuracies = []
    lstm_accuracies = []

    print("Start training EAs")

    for i in range(args['n_runs']):
        start = time.time()
        ea_accuracies.append(train_ea(args))
        end = time.time()
        print("EA run %d of %d complete, time taken: %.1f\n" % (i + 1, args['n_runs'], end - start))

    if args['train_lstm']:
        import lstm
        for i in range(args['n_runs']):
            start = time.time()
            lstm_accuracies.append(lstm.train(args))
            end = time.time()
            print("LSTM run %d of %d complete, time taken: %.1f\n" % (i + 1, args['n_runs'], end - start))


    print()
    print("EA Results")
    print("Min accuracy: %.1f \nMax accuracy: %.1f" % (min(ea_accuracies), max(ea_accuracies)))
    print("Average accuracy: %.1f \nStandard deviation: %.1f" % (np.mean(ea_accuracies), np.std(ea_accuracies)))

    if len(lstm_accuracies) > 0:
        print()
        print("LSTM Results")
        print("Min accuracy: %.1f \nMax accuracy: %.1f" % (min(lstm_accuracies), max(lstm_accuracies)))
        print("Average accuracy: %.1f \nStandard deviation: %.1f" % (np.mean(lstm_accuracies), np.std(lstm_accuracies)))
