# Linear regression by backpropagation by SGD and loss function of MSE.

# x_train = [1, 2, 3, 4, 5]
# y_train = [2, 3.8, 6.1, 7.9, 10.11]
# x_test = [6, 7] 
# y_test = [12, 14]

import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3], [4], [5]]) # 1x5 tensor
y_train = torch.FloatTensor([[2], [3.8], [6.1], [7.9], [10.11]]) # 1x5 tensor

x_test = torch.FloatTensor([ [6], [7] ])
y_test = torch.FloatTensor([ [12], [14] ])

W = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W,b], lr = 0.01) 
# torch.optim으로 optimizer object를 선언. 이것은 현재 상태를 저장하고, 
# computed gradient를 이용해 parameter를 업데이트 함.

# 이걸 처음에는 좀 0.01처럼 큰걸로 했다가, 나중에는 0.001로 정밀하게 할 수는 없을까?

 
nb_epochs = 1000
for epochs in range(1, nb_epochs+1):
    hypothesis = x_train * W + b # the output of the model
    cost = torch.mean( (hypothesis - y_train) ** 2 ) # the loss of output for target

    optimizer.zero_grad() # 
    cost.backward() # 
    optimizer.step() # 
    print('epoch: ', epochs , ' / cost: ', cost)

print('------trainning ends------')

print('trained W, b : ', W, b) # it should be 2, 0

test_hypothesis = x_test * W + b
test_MSE = torch.mean( (test_hypothesis - y_test) ** 2 )

print('as a result of test, the MSE with test dataset: ', test_MSE)
