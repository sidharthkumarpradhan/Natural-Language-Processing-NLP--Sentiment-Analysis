# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Iterator
from sentiment_analysis_model import SentimentAnalysisModel

# Initialize model, optimizer, and criterion
model = SentimentAnalysisModel(len(TEXT.vocab), 100, 1, 256)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Move model to device
model = model.to(device)
criterion = criterion.to(device)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

# Number of epochs for training
N_EPOCHS = 5

# Train the model
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'saved_model.pt')
