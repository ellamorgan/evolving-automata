import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import data


class Model(nn.Module):

    def __init__(self, hdim):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hdim, batch_first=True)
        self.linear = nn.Linear(in_features=hdim, out_features=1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x, (_, _) = self.lstm(x)
        x = self.sigmoid(self.linear(x[:, -1, :]))
        return x



def train(args):

    # Trains on GPU if available
    usecuda = torch.cuda.is_available()
    device = torch.device("cuda") if usecuda else torch.device("cpu")

    # Generate and load in data
    loaders = data.generate_dataset_nn(args['pos_reg'], args['neg_reg'], args['n_data'], 
                                       args['batch_size'], args['train_split'], args['max_repeat'], 
                                       usecuda)

    # Initialize model, Adam optimizer, and Binary Cross Entropy loss
    model = Model(args['h_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = nn.BCELoss()

    # Train model, iterating over the training data (epoch) times
    for _ in range(args['lstm_epochs']):

        train_loss = 0

        for batch in loaders['train']:

            input = batch[:, 1:, :].to(device)
            target = batch[:, 0, :].to(device)

            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    # Test accuracy of trained model on the unseen testing set
    with torch.no_grad():

        final_results = []
        final_target = []

        model.eval()
        test_loss = 0

        for batch in loaders['test']:

            input = batch[:, 1:, :].to(device)
            target = batch[:, 0, :].to(device)

            out = model(input)
            loss = criterion(out, target)
            final_results.append(out.cpu())
            final_target.append(target.cpu())
            test_loss += loss.item()
        
        test_loss /= (len(loaders['test']) * args['batch_size'])
    
    # Calculate and print results
    final_results = np.concatenate(final_results).flatten()
    final_results[final_results >= 0.5] = 1
    final_results[final_results < 0.5] = 0
    final_target = np.concatenate(final_target).flatten()
    final_result = 100 * int(np.sum(np.absolute(final_results - final_target))) / len(final_results)
    print("Final error of LSTM on test set: %.1f%%" % (final_result))

    return final_result