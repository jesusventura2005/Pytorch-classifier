import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Capas convolucionales
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv_layers.append(nn.Conv2d(1, hidden_size[0], kernel_size=3, padding=1))
 
        for i in range(len(hidden_size)-1):
            self.conv_layers.append(nn.Conv2d(hidden_size[i], hidden_size[i+1], kernel_size=3, padding=1))
        
        num_pooling_layers = len(hidden_size)
        final_size = 28 // (2 ** num_pooling_layers)
        self.fc_input_size = hidden_size[-1] * final_size * final_size
        
        # Capas fully connected
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.fc_input_size, 128))
        self.fc_layers.append(nn.Linear(128, output_size))
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.pool(F.relu(conv_layer(x)))
            x = self.dropout(x)
        
        # Aplanar para las capas fully connected
        x = x.view(-1, self.fc_input_size)
        
        # Aplicar capas fully connected
        for i, fc_layer in enumerate(self.fc_layers[:-1]):
            x = F.relu(fc_layer(x))
            x = self.dropout(x)
        
        # Capa de salida
        x = self.fc_layers[-1](x)
        return F.log_softmax(x, dim=1)
    
    
    
        
        