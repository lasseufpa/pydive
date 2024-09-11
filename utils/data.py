import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms 
from sklearn.preprocessing import LabelEncoder

import numpy as np
import os

import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_nist(batch_size=3, test_size=0.3):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 64
    num_classes = 10

    return trainloader, testloader, input_size, num_classes

def load_mnist(data_path, batch_size,):
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    # Download and load the training data
    trainset = datasets.MNIST(data_path, download=True, train=True, transform=transform)
    valset = datasets.MNIST(data_path, download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

    X, y = next(iter(trainloader))
    input_size = X.size(2)*X.size(3)
    num_classes = len(torch.unique(y))

    return trainloader, valloader, input_size, num_classes

def load_mnistII(data_path, batch_size):
    trainloader = torch.utils.data.DataLoader(
                                            torchvision.datasets.MNIST(data_path, train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                            batch_size=batch_size, shuffle=True)

    valloader = torch.utils.data.DataLoader(
                                            torchvision.datasets.MNIST(data_path, train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                            batch_size=batch_size, shuffle=True)
    X, y = next(iter(trainloader))
    input_size = X.size(2)*X.size(3)
    num_classes = len(torch.unique(y))

    return trainloader, valloader, input_size, num_classes

def load_FMNIST(data_path, batch_size):
    training_data = datasets.FashionMNIST(
                                        root=data_path,
                                        train=True,
                                        download=True,
                                        transform=ToTensor()
                                    )

    test_data = datasets.FashionMNIST(
                                    root=data_path,
                                    train=False,
                                    download=True,
                                    transform=ToTensor()
                                )

    trainloader = DataLoader(training_data, batch_size=batch_size)
    valloader = DataLoader(test_data, batch_size=batch_size)

    X, y = next(iter(trainloader))
    input_size = X.size(2)*X.size(3)
    num_classes = len(torch.unique(y))

    return trainloader, valloader, input_size, num_classes

def load_iris():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Leia o arquivo CSV usando o caminho do diretório atual
    df = pd.read_csv(os.path.join(dir_path, 'iris.csv'))
    df['Species'].unique()
    df['Species'] = df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
    df.drop(['Id'],axis=1,inplace=True)
    X = df.drop(["Species"],axis=1).values
    y = df["Species"].values
    batch_size = 1

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    #X_val = torch.FloatTensor(X_val)
    #y_val = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    #val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 4
    num_classes = 3

    return trainloader, testloader, input_size, num_classes

def get_in_out_size(loader):
    X, y = next(iter(loader))
    input_size = X.size(3)
    print(input_size)


class CSVDataset(Dataset):
    """Data"""

    # load the dataset
    def __init__(self, path, mask_path, device):
        # load the csv file as a dataframe
        data = np.genfromtxt(path, delimiter=",")
        mask = np.genfromtxt(mask_path, delimiter=",")


        # store the inputs and outputs
        self.x_train = data[:, :-1]
        self.y_train = data[:, -1]

        # ensure input data is floats
        self.x_train = torch.as_tensor(self.x_train).float().to(device)

        # Load the mask  
        self.mask = torch.from_numpy(mask).to(device)

        self.x_train = torch.cat((self.x_train, self.mask.unsqueeze(-1)), dim=-1)

        # label encode target and ensure the values are floats
        self.y_train = LabelEncoder().fit_transform(self.y_train)
        self.y_train = torch.as_tensor(self.y_train).to(device)
        self.shape = self.x_train.shape
        self.classes = torch.unique(self.y_train)
        self.n_classes = torch.numel(self.classes)

    # number of rows in the dataset
    def __len__(self):
        return len(self.x_train)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x_train[idx], self.y_train[idx]]