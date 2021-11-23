import torch
import os

# this file is to save and load torch.nn parameters

DIR = './NN/'

def make_dir(file_dir):
     isExists=os.path.exists(file_dir)
     if not isExists:
         os.makedirs(file_dir)

def get_full_dir(nn_type, name):
    make_dir(DIR)
    full_dir = DIR + nn_type + '_' + name + '.pkl'
    return full_dir

def save_nn_frame(net, nn_type='policy', name='nn_name'):
    full_dir = get_full_dir(nn_type, name)
    torch.save(net, full_dir)
    print('save whole nn frame in ', full_dir)

def load_nn_frame(nn_type='policy', name='nn_name'):
    full_dir = get_full_dir(nn_type, name)
    net = torch.load(full_dir)
    print('load whole nn frame in ', full_dir)
    return net

def save_nn_param(net, nn_type='policy', name='nn_name'):
    full_dir = get_full_dir(nn_type, name)
    torch.save(net.state_dict(), full_dir)
    print('save nn parameters in ', full_dir)

def load_nn_param(net, nn_type='policy', name='nn_name'):
    full_dir = get_full_dir(nn_type, name)
    state_dict = torch.load(full_dir)
    net.load_state_dict(state_dict)
    print('load nn parameters in ', full_dir)


def test():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    nn_type = 'q_value'
    name = 'ac'
    save_nn_frame(net1, nn_type, name)
    net2 = load_nn_frame(nn_type, name)

    save_nn_param(net2, nn_type, name)
    load_nn_param(net2, nn_type, name)


if __name__ == '__main__':
    test()