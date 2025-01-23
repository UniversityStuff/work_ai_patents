def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out['patents'][data['patents'].train_mask], data['patents'].y[data['patents'].train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out['patents'].argmax(dim=1)
    test_correct = pred[data['patents'].test_mask] == data['patents'].y[data['patents'].test_mask]
    test_acc = int(test_correct.sum()) / int(data['patents'].test_mask.sum())
    return test_acc

class ModelTrainer:
    def __init__(self, model, data, optimizer, criterion):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train_model(self, epochs):
        losses = []
        
        for epoch in range(1, epochs + 1):
            loss = train(self.model, self.data, self.optimizer, self.criterion)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                
    def test_model(self):
        test_acc = test(self.model, self.data)
        print(f'***** Evaluating the test dataset *****')
        print(f'Test Accuracy: {test_acc:.4f}')