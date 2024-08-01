import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLAgent(nn.Module):
    def __init__(self, image_shape, metadata_shape, num_actions, lstm_hidden_size=512):
        super(RLAgent, self).__init__()
        
        # CNN for image processing
        self.conv1 = nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully connected layers for metadata
        self.fc_meta = nn.Linear(metadata_shape, 128)
        
        # Determine the size of the flattened CNN output
        self.flatten_size = self._get_conv_output(image_shape)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.flatten_size + 128, hidden_size=lstm_hidden_size, batch_first=True)
        
        # Fully connected layers for action selection
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
    
    def _get_conv_output(self, shape):
        # Helper function to determine the size of the CNN output
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, image, metadata, hidden_state=None):
        # Image processing
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Metadata processing
        meta = F.relu(self.fc_meta(metadata))
        
        # Feature fusion
        combined = torch.cat((x, meta), dim=1).unsqueeze(0)  # Add sequence dimension
        
        # LSTM
        lstm_out, hidden_state = self.lstm(combined, hidden_state)
        lstm_out = lstm_out.squeeze(0)  # Remove sequence dimension
        
        # Decision making
        x = F.relu(self.fc1(lstm_out))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, hidden_state
    

def train():
    pass
    # for episode in range(num_episodes):
    #     state = env.reset()
    #     hidden_state = None  # Initialize LSTM hidden state
    #     done = False
        
    #     while not done:
    #         image, metadata = preprocess_state(state)
    #         action_probs, hidden_state = agent(image, metadata, hidden_state)
    #         action = select_action(action_probs)
            
    #         next_state, reward, done, info = env.step(action)
            
    #         # Store experience in replay buffer
    #         replay_buffer.add(state, action, reward, next_state, done, hidden_state)
            
    #         # Sample mini-batch from replay buffer
    #         batch = replay_buffer.sample(batch_size)
            
    #         # Compute loss and update agent
    #         loss = compute_loss(agent, batch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         state = next_state


# Initialize agent
image_shape = (3, 84, 84)  # Example image shape (channels, height, width)
metadata_shape = 10  # Example metadata shape
num_actions = 4  # Example number of actions
agent = RLAgent(image_shape, metadata_shape, num_actions)

# Example forward pass
image = torch.randn((1, *image_shape))  # Batch size of 1
metadata = torch.randn((1, metadata_shape))
output, hidden_state = agent(image, metadata)
print(output)



