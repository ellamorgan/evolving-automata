import torch
from torch.utils.data import DataLoader

import numpy as np
import random
import configparser
import re



def load_args(config_path):
    """
    Parses command line arguments
    Returns a dictionary with command line and config file configs
    Gives priority to command line, can overwrite config file configs
    """
    config = configparser.ConfigParser()
    with open(config_path) as f:
        config.read_file(f)
    configs = dict(config['main'])

    float_rg = "(([+-]?([0-9]*)?[.][0-9]+)|([0-9]+[e][-][0-9]+))"
    f = lambda x : float(x) if re.match(float_rg, x) else int(x) if x.isnumeric() else x
    configs = {k : f(v) for k, v in configs.items()}
    
    print("Configs used:")
    for k, v in configs.items():
        print(k + ": " + str(v))
    print()
    return configs



def generate_word(reg, max_repeat):
    '''
    Sampling based method returns a randomly generated
    word in a target language
    '''
    # Remove spaces
    reg = reg.replace(" ", "")

    # Resolve *'s - from right to left
    i = len(reg) - 1
    while i >= 0:

        if reg[i] == '*':

            # Single character has a *
            if reg[i - 1] != ')':
                add_len = random.randint(0, max_repeat)
                reg = reg[:i - 1] + reg[i - 1] * add_len + reg[i + 1:]
                i -= 1
            
            # Section wrapped in brackets has a *
            else:
                or_ = False
                brackets = 1
                j = 2
                while brackets != 0:
                    if reg[i - j] == ')':
                        brackets += 1
                    elif reg[i - j] == '(':
                        brackets -= 1
                    elif reg[i - j] == '+' and brackets == 1:
                        or_ = True
                    j += 1
                j -= 1
                
                if or_:
                    rep = reg[i - j : i]
                else:
                    rep = reg[i - j + 1 : i - 1]

                add = rep * random.randint(0, max_repeat)
                reg = reg[:i - j] + add + reg[i + 1:]
                i += len(add) - j
        
        i -= 1

    # Tracks unclosed brackets
    last_bracket = []

    # Resolve +'s - resolve in the order that they're closed
    # All brackets correspond to a series of + operations
    i = 0
    while i < len(reg):
        if reg[i] == '(':
            last_bracket.append(i)

        if reg[i] == ')':
            start = last_bracket.pop()
            choices = reg[start + 1 : i].split('+')
            choice = random.sample(choices, 1)[0]
            reg = reg[:start] + choice + reg[i + 1 :]
            i = start + len(choice) - 1
        
        i += 1
    
    return reg



def generate_dataset(reg, size, max_repeat, train_split=None):
    '''
    Generated dataset given a regular expression and size
    Removes duplicates
    '''
    data = set()
    while len(data) < size:
        data.add(generate_word(reg, max_repeat))
    data = list(data)

    if train_split is None:
        return data

    split_ind = int(len(data) * train_split)
    train_data = data[:split_ind]
    test_data = data[split_ind:]

    return {'train' : train_data, 'test' : test_data}



def generate_dataset_nn(pos_reg, neg_reg, size, batch, train_split, max_repeat, usecuda):
    '''
    Generates the dataset for the LSTM
    '''
    dataset = generate_dataset(pos_reg, size, max_repeat) + generate_dataset(neg_reg, size, max_repeat)

    numeric_data = []
    max_len = len(max(dataset, key=len))

    for word in dataset:
        numeric_word = np.array(list(map(ord, word)), dtype=int) - 97
        padded_word = np.concatenate((numeric_word, -1 * np.ones((max_len - len(word)), dtype=int)), axis=0)
        numeric_data.append(padded_word)
    
    labels = np.concatenate((np.ones((size, 1), dtype=int), np.zeros((size, 1), dtype=int)), axis=0)
    numeric_data = np.concatenate((labels, numeric_data), axis=1)
    numeric_data = np.expand_dims(numeric_data, axis=-1)

    numeric_data = torch.FloatTensor(numeric_data)

    train_ind = int(len(numeric_data) * train_split)

    train_dataloader = DataLoader(numeric_data[:train_ind], batch_size=batch, shuffle=True, pin_memory=usecuda)
    test_dataloader = DataLoader(numeric_data[train_ind:], batch_size=batch, shuffle=True, pin_memory=usecuda)

    return {'train' : train_dataloader, 'test' : test_dataloader}