import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import yaml
import pprint
import numpy as np

class RNNCoreContainer(nn.Module):
    def __init__(self, true_rnn_core):
        super().__init__()
        for para in true_rnn_core.named_parameters():
            self.register_parameter(para[0], para[1])

class Container(nn.Module):
    def __init__(self, true_rnn):
        super().__init__()
        for module in true_rnn.named_modules():
            name = module[0]
            if name == "linear_layer":
                self.add_module(name, module[1])
            elif name == "gru":
                self.gru = RNNCoreContainer(module[1])

class RNNPredictor(nn.Module):
    
  def __init__(self, input_size, output_classes, hidden_state_size, core="GRU"):
    print("core: " + core)
    super().__init__()
    self.hidden_state_size = hidden_state_size
    if core == "RNN":
        self.gru = nn.RNN(input_size=input_size, hidden_size=hidden_state_size, batch_first=True)
    elif core == "GRU":
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_state_size, batch_first=True)
    self.output_classes = output_classes
    self.output_layer_size = sum(output_classes)
    self.linear_layer = nn.Linear(hidden_state_size, self.output_layer_size)

  def forward(self, inputs):
    hidden_state = torch.zeros(1, inputs.shape[0], self.hidden_state_size)
    gru_outputs, _ = self.gru(inputs, hidden_state)
    logits = self.linear_layer(gru_outputs)
    probs = []
    count = 0
    for i, num_of_outputs in enumerate(self.output_classes):
        probs.append(torch.nn.functional.softmax(logits[:,:,count:count+num_of_outputs], dim=2))
        count += num_of_outputs
    return probs

  @torch.jit.export
  def recurrentForward(self, hidden_state, inputs):
    gru_outputs, hidden_state = self.gru(inputs, hidden_state)
    logits = self.linear_layer(gru_outputs)
    probs = []
    count = 0
    for i, num_of_outputs in enumerate(self.output_classes):
        probs.append(torch.nn.functional.softmax(logits[:,:,count:count+num_of_outputs], dim=2))
        count += num_of_outputs
    return probs, hidden_state

class Dataset(torch.utils.data.Dataset):
  def __init__(self, inputs, outputs):
    self.inputs = torch.IntTensor(inputs).type(torch.FloatTensor)
    self.outputs = torch.IntTensor(outputs).type(torch.LongTensor)
  def __len__(self):
    return len(self.outputs)
  def __getitem__(self, idx):
    return self.inputs[idx], self.outputs[idx]

# generate data for training influence predictor
def generate_data(config_path, data_folder_path):
  command = './run scripts/generateInfluenceLearningData.sh {} {}'.format(config_path, data_folder_path)
  print(command)
  print(os.system(command))

def train_influence_predictor(
    config_path,
    generate_new_data=False,
    batch_size = 128,
    lr = 0.001,
    weight_decay=5e-4,
    num_epochs = 1000,
    data_path=None,
    core="GRU",
    save_model=True
):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    the_path = config_path.split(".yaml")[0].replace("configs", "models")
    if data_path is None:
      data_path = the_path
    
    if generate_new_data is True or os.path.exists(data_path) is False:
        generate_data(config_path, data_path)
    
    hidden_state_size = config["AgentComponent"][config["General"]["IDOfAgentToControl"]]["Simulator"]["InfluencePredictor"]["numberOfHiddenStates"]
    core = config["AgentComponent"][config["General"]["IDOfAgentToControl"]]["Simulator"]["InfluencePredictor"]["Type"]
    # read inputs and outputs from files
    inputs = torch.jit.load(os.path.join(data_path,"inputs.pt"))._parameters['0']
    outputs = torch.jit.load(os.path.join(data_path,"outputs.pt"))._parameters['0']
    print("inputs:", inputs.shape)
    print("outputs:", outputs.shape)
    print("data loaded.")

    # split raw_inputs and output into training_inputs, training_outputs, testing_inputs, testing_outputs
    split_ratio = 0.8
    full_dataset_size = len(inputs)
    training_set_size = int(full_dataset_size * split_ratio)
    testing_set_size = full_dataset_size - training_set_size
    training_inputs = inputs[:training_set_size]
    training_outputs = outputs[:training_set_size]
    testing_inputs = inputs[training_set_size:]
    testing_outputs = outputs[training_set_size:]
    input_size = inputs.size()[-1]
    output_size = outputs.size()[-1]
    output_classes = []
    for i in range(output_size):
        _max = torch.max(outputs[:,:,i]).item()
        _min = torch.min(outputs[:,:,i]).item()
        _num = _max-_min+1
        output_classes.append(_num)
    print("output classes:")
    pprint.pprint(output_classes)
    print("training set and testing set are split.")

    training_dataset = Dataset(training_inputs, training_outputs)
    testing_dataset = Dataset(testing_inputs, testing_outputs)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=testing_set_size,shuffle=True, drop_last=False)
    print("dataset constructed")

    # initialize the influence predictor
    predictor = RNNPredictor(input_size, output_classes, hidden_state_size, core=core)
    print("influence predictor initialized.")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    def evaluate(the_predictor, dataloader):
        with torch.no_grad():
            loss = 0
            for batch_inputs, batch_targets in testing_dataloader:
                predictions = the_predictor(batch_inputs)
                for i, num_classes in enumerate(output_classes):
                    the_predictions = predictions[i].view(-1, num_classes)
                    the_targets = batch_targets[:,:,i].view(-1)
                    loss += loss_function(the_predictions, the_targets).item()
            return loss

    epoch_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        gradient_step = 0
        for batch_inputs, batch_targets in training_dataloader:
            predictor.zero_grad()
            predictions = predictor(batch_inputs)
            loss = 0
            for i, num_classes in enumerate(output_classes):
                the_predictions = predictions[i].view(-1, num_classes)
                the_targets = batch_targets[:,:,i].view(-1)
                loss += loss_function(the_predictions, the_targets)
            loss.backward()     
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.item() * batch_inputs.size()[0] / training_set_size
            gradient_step += 1
        test_loss = evaluate(predictor, testing_dataloader)
        test_losses.append(test_loss)
        if epoch % 100 == 0:
          print("epoch_loss {}: {}".format(epoch, epoch_loss))
          print("test_loss {}: {}".format(epoch, test_loss))
        epoch_losses.append(epoch_loss)

    plt.title("training loss")
    plt.plot(epoch_losses)
    plt.show()
    plt.title("testing loss")
    plt.plot(test_losses)
    plt.show()

    if os.path.exists(the_path) is not True:
      print(the_path)
      os.makedirs(the_path)

    # save the statistics
    with open(os.path.join(the_path, "training_losses.npy"), "wb") as f:
        np.save(f, epoch_losses)
    with open(os.path.join(the_path, "testing_losses.npy"), "wb") as f:
        np.save(f, test_losses)

    if save_model:    
      # save the model
      model_path = os.path.join(the_path, "model.pt")

      if core == "GRU":
        script_model = torch.jit.script(predictor)
      else:
        script_model = torch.jit.script(Container(predictor))
        print("transformed")
      print(script_model)
      torch.jit.save(script_model, open(model_path, "wb"))
      print("model saved at", model_path)

    return predictor