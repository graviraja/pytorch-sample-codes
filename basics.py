import torch
import numpy as np

# ################################################################### #
# Implementing a 2 layer network with 1 hidden layer using only numpy #
# ################################################################### #

N = 64          # batch size
D_in = 1000     # input dimension
H = 100         # hidden dimension
D_out = 10      # output dimension

# create a random input of shape [batch_size, input_dim]
x = np.random.randn(N, D_in)
# create a random output of shape [batch_size, output_dim]
y = np.random.randn(N, D_out)

# randomly initialize the weights of the network
# w1 of shape [input_dim, hidden_dim]
w1 = np.random.randn(D_in, H)
# w2 of shape [hidden_dim, output_dim]
w2 = np.random.randn(H, D_out)

# learning rate
learning_rate = 1e-6
print('--------------------------------')
print("Training the network using numpy")
print('--------------------------------')

for epoch in range(500):
    # forward pass
    i2h = x.dot(w1)
    relu_activation = np.maximum(i2h, 0)
    y_pred = relu_activation.dot(w2)

    # loss
    loss = np.square(y_pred - y).sum()
    if epoch % 50 == 0:
        print(f"epoch : {epoch}, loss : {loss}")

    # backpropagation
    # compute the gradients of w1, w2 w.r.t loss
    # dl/dw2 = dl/dy * dy/dw2
    #        = 2*(y_pred - y) * (relu_activation)
    grad_y_pred = 2.0 * (y_pred - y)
    # transpose on relu_activation since w2 is of shape [h, h_out]
    grad_w2 = relu_activation.T.dot(grad_y_pred)

    # dl/dw1 = dl/dy * dy/drelu * drelu/dw1
    #        = 2*(y_pred - y) * (w2) * relu(x)
    grad_h_relu = grad_y_pred.dot(w2.T)  # shape is [N, h]
    grad_h = grad_h_relu.copy()
    grad_h[i2h < 0] = 0   # relu(dl/dy * dy/drelu)
    grad_w1 = x.T.dot(grad_h)  # shape is [h_in, h]

    # update the weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# ################################################################################ #
# Implementing the above 2 layer network with 1 hidden layer using pytorch tensors #
# ################################################################################ #

print('-----------------------------------------')
print("Training the network using pytorch tensor")
print('-----------------------------------------')

# define the default dtype and device to use
dtype = torch.float
device = torch.device("cpu")       # can use gpu also with torch.device("cuda:0")

# create a random input of shape [batch_size, input_dim]
x = torch.randn(N, D_in, device=device, dtype=dtype)
# create a random output of shape [batch_size, output_dim]
y = torch.randn(N, D_out, device=device, dtype=dtype)

# randomly initialize the weights of the network
# w1 of shape [input_dim, hidden_dim]
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 of shape [hidden_dim, output_dim]
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

for epoch in range(500):
    # forward pass
    i2h = x.mm(w1)
    relu_activation = i2h.clamp(min=0)
    y_pred = relu_activation.mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum().item()
    if epoch % 50 == 0:
        print(f"epoch : {epoch}, loss : {loss}")

    # backpropagation
    # compute the gradients of w1, w2 w.r.t loss
    # dl/dw2 = dl/dy * dy/dw2
    #        = 2*(y_pred - y) * (relu_activation)
    grad_y_pred = 2.0 * (y_pred - y)
    # transpose on relu_activation since w2 is of shape [h, h_out]
    grad_w2 = relu_activation.t().mm(grad_y_pred)

    # dl/dw1 = dl/dy * dy/drelu * drelu/dw1
    #        = 2*(y_pred - y) * (w2) * relu(x)
    grad_h_relu = grad_y_pred.mm(w2.t())  # shape is [N, h]
    grad_h = grad_h_relu.clone()
    grad_h[i2h < 0] = 0   # relu(dl/dy * dy/drelu)
    grad_w1 = x.t().mm(grad_h)  # shape is [h_in, h]

    # update the weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# ################################################################################# #
# Implementing the above 2 layer network with 1 hidden layer using pytorch autograd #
# ################################################################################# #

print('-------------------------------------------')
print("Training the network using pytorch autograd")
print('-------------------------------------------')

# create a random input of shape [batch_size, input_dim]
x = torch.randn(N, D_in, device=device, dtype=dtype)
# create a random output of shape [batch_size, output_dim]
y = torch.randn(N, D_out, device=device, dtype=dtype)

# randomly initialize the weights of the network
# setting the requires_grad = True indicates that we want to compute gradients w.r.t
# tensors during the backward propagation

# w1 of shape [input_dim, hidden_dim]
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 of shape [hidden_dim, output_dim]
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

for epoch in range(500):
    # forward pass
    i2h = x.mm(w1)
    relu_activation = i2h.clamp(min=0)
    y_pred = relu_activation.mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum()
    if epoch % 50 == 0:
        print(f"epoch : {epoch}, loss : {loss.item()}")

    # this call will compute the gradients of loss w.r.t all tensors
    # with requires_grad = True
    # after the call, w1.grad and w2.grad will be the tensors holding
    # the gradient of the loss w.r.t w1 and w2.
    loss.backward()

    # update the weights
    # weight updation don't need to be tracked, so wrap the updation code
    # in torch.no_grad()
    # updates can also be done with torch.optim.SGD
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero the gradients after updating the weights
        w1.grad.zero_()
        w2.grad.zero_()

# ################################################################################### #
# Implementing the above 2 layer network with 1 hidden layer using pytorch nn package #
# ################################################################################### #
print('--------------------------------------------')
print("Training the network using pytorch nn module")
print('--------------------------------------------')

# create a random input of shape [batch_size, input_dim]
x = torch.randn(N, D_in, device=device, dtype=dtype)
# create a random output of shape [batch_size, output_dim]
y = torch.randn(N, D_out, device=device, dtype=dtype)

# define the model using torch.nn. nn.Sequential is module contains other modules
# and applies them in sequence to produce the output.
# nn.Linear computes output from input using a linear function
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# loss function
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4

for epoch in range(500):
    # forward pass
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    if epoch % 50 == 0:
        print(f"epoch : {epoch}, loss : {loss.item()}")

    # zero the gradients before the backward pass
    model.zero_grad()

    # backward pass
    loss.backward()

    # weight update
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


# ############################################################################## #
# Implementing the above 2 layer network with 1 hidden layer using pytorch optim #
# ############################################################################## #
print('----------------------------------------')
print("Training the network using pytorch optim")
print('----------------------------------------')

# create a random input of shape [batch_size, input_dim]
x = torch.randn(N, D_in, device=device, dtype=dtype)
# create a random output of shape [batch_size, output_dim]
y = torch.randn(N, D_out, device=device, dtype=dtype)

# define the model using torch.nn. nn.Sequential is module contains other modules
# and applies them in sequence to produce the output.
# nn.Linear computes output from input using a linear function
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# loss function
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(500):
    # forward pass
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    if epoch % 50 == 0:
        print(f"epoch : {epoch}, loss : {loss.item()}")

    # zero the gradients before the backward pass
    model.zero_grad()

    # backward pass
    loss.backward()

    # weight update using the optimizer
    optimizer.step()
