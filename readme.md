Requires pytorch==1.9.1 (current version) for LSTM. To run without this requirement, set train_lstm=0 in configs.conf.

To run:
```
python3 main.py
```

Hyperparamters can be changed in configs.conf

### main.py
Main entry of program.

* train_ea()  
Training loop for the evolutionary algorithm.

### ea.py
Contains all methods (aside from the training loop) for the evolutionary algorithm. The methods are described in the order they appear in the file.

* add_random_edge()  
Randomly generates a single edge. Used in initializing the finite automata and both micro and macro mutations.

* initalize_fa()  
Initializes a single finite automata.

* initialize_pop()  
Initializes the population, calling initialize_fa() to generate individuals.

* evaluate_fa()  
Evaluates the fitness of a single individual.

* evaluate_fitness()  
Evaluates the fitness of a whole population, calling evaluate_fa(). This method is passed a dataset for positive examples and one for negative. It calls evaluate_fa() for both the positive and negative examples, subtracting the score for the negative examples from the score for the positive.

* return_active()  
This is essentially an intron removing method. It returns a list of effective nodes, i.e. nodes that are accessible for some string from the start node.

* micro_mutate()  
Randomly reassigns each edge with mucro_mutate_rate probability.

* macro_mutate()  
Will either add a random row or delete an existing row in an individual with a macro_mutation_rate probability.

* crossover()  
Performs crossover between two individuals. Splits both individuals at a random point along the depth dimension and reattaches them.

* tournament()  
Performs a single tournament. Draws two groups of size tournament_size without replacement. The best individuals in each tournament will produce offspring, which replace the two worst individuals in the two tournaments combined.

* print_fa()  
Prints the effective nodes of an individual by calling return_active() and only printing those.

### lstm.py
Contains the LSTM model and training loop.

* class Model()  
A standard PyTorch model. Consists of an LSTM layer then a linear layer which outputs the final prediction.

* train()  
Training loop for LSTM, trains model then prints results.

### data.py
Contains methods for generating the training instances, formatting the datasets, and parsing the configuration file.

* load_args()  
Loads the configuration arguments present in configs.conf into a dictionary so that they can be easily passed around.

* generate_word()  
Randomly generates a single word given a regular expression. The argument max_repeats controls the maximum number of times a Kleene star * can repeat something. Will first resolve the repeats, parsing them from right to left and resulting in a regular expression without any \*'s. Will then parse the +'s from left to right, randomly sampling from the options to produce the final generated word.

* generate_dataset()  
Builds a dataset of a specified size given a regular expression by repeatedly calling generate_word(). Uses a set to ensure there are no duplicates, terminates the generation process once the set has the specified number of words. Returns the data split into training and testing sets.

* generate_dataset_nn()  
Generates the data for the LSTM. Loads the data into PyTorch dataloaders for training and testing to be compatible with the LSTM implementation.
