import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators

class Network(layers.BaseNetwork):
    
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    
    #Update the network 

    def __init__(self, data_layer, parameters):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here

        #get the parameters
        hidden_units = parameters["hidden_units"] 
        hidden_layers = parameters["hidden_layers"]
        
        #define blank modules
        self.MY_MODULE_LIST = layers.ModuleList()
        
        for i in range(hidden_layers):

            if i == 0 :
                l = layers.Linear(data_layer, hidden_units[i])
                b = layers.Bias(l)

            else :
                l = layers.Linear(b, hidden_units[i])
                b = layers.Bias(l)

            self.MY_MODULE_LIST.append(l)
            self.MY_MODULE_LIST.append(b)
        
        #Remove the last element
        last = self.MY_MODULE_LIST[-1]
        del self.MY_MODULE_LIST[-1]

        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.MY_LAST_LAYER = last
        self.set_output_layer(self.MY_LAST_LAYER)


class Trainer:
    def __init__(self, lr):
        self.lr = lr
#         pass
    
    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4 and mnist:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers. 
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        # hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        # hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        self.parameters = parameters
        
        #TODO: construct your network here
        self.network = Network(data_layer, self.parameters)
        return self.network
    
    def setup(self, training_data):
        x, y = training_data
        
        #TODO: define input data layer
        self.data_layer = layers.Data(x)

        #define
        self.parameters = {'hidden_units' : [1], 'hidden_layers' : 1}
        
        #TODO: construct the network. you don't have to use define_network.
        self.network = Network(self.data_layer, self.parameters)
        
        #TODO: use the appropriate loss function here
        self.loss_layer = layers.SquareLoss(self.network.MY_LAST_LAYER, y)
        
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        modules  = self.network.get_modules_with_parameters() ## list
#         lr = 1e-2
        self.optim = layers.SGDSolver(self.lr, modules)
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function

        #get the loss value
        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optim.step()
        return loss
    
    def get_num_iters_on_public_test(self):
        #TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 30000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        
        for iter in range(num_iter) :
            loss = self.train_step()
            train_losses.append(loss)
        # you have to return train_losses for the function
        return train_losses
    
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        
        #define the dict
        data_dict = data_generators.data_1a()
        
        #get the data
        training_data = data_dict['train']
        test_data = data_dict['test']

        #initialize the network
        _, network, _, _ = trainer.setup(training_data)

        #train
        num_iter = trainer.get_num_iters_on_public_test()
        train_losses = trainer.train(num_iter)
        

        #hold-out data
        x_test, y_test = test_data
        test_data_layer = layers.Data(x_test)
        y_pred = network.MY_LAST_LAYER.forward()
        
        #compute squared loss
        test_loss = layers.SquareLoss(network.MY_LAST_LAYER, y_test).forward()

        #save the weights
        params = []
        for m in network.get_modules_with_parameters() :
            params.append(m.W)
#             print (params)

        return np.savez('/Users/schakra3/PS1_828L/loss.npz', train_losses=train_losses, y_pred=y_pred, y_test= y_test, params = params, test_loss=test_loss)


        
    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

if __name__ == "__main__":
    main()
