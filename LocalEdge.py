# For Test
# Run LocalEdge and TorchServer in the same computer.
import torch
import torch.nn as nn
import torch.optim as optim
import grpc
import edge_server_pb2
import edge_server_pb2_grpc
import numpy as np
import Net

HOST_IP = '127.0.0.1'
PORT= 65121



edge_net = Net.EdgeNet()
optimizer = optim.SGD(edge_net.parameters(), lr=0.01)

# 反向传播
def backward():
    print("Start local backward and updating...")
    edge_output.backward(server_grad)
    print(f"EdgeNet - fc3 gradient: {(edge_net.fc3.weight.grad)[0][:5]}")
    print(f"EdgeNet - fc2 gradient: {(edge_net.fc2.weight.grad)[0][:5]}")
    print(f"EdgeNet - fc1 gradient: {(edge_net.fc1.weight.grad)[0][:5]}")
    optimizer.step()
    print(f"\nEdgeNet parameters updated\n")

# 假设输入数据为 x
def loadInputs():  #创建输入数据
    x = torch.randn(1, 784)
    return x



#正式开始训练:
EPOCH = 1
for epoch in range(EPOCH):
    print(f"Epoch {epoch} start...")
    TEMP_INPUT = loadInputs()
    print(f"Input to EdgeNet: {TEMP_INPUT[0][:5]}")
    edge_output = edge_net(TEMP_INPUT)
    print(f"EdgeNet output: {edge_output[0][:5]}")
    # 将 edge_output 发送到服务器
    channel = grpc.insecure_channel(f'{HOST_IP}:{PORT}')
    print(f"\nCommunicating to Server at {HOST_IP}:{PORT} ...")
    stub = edge_server_pb2_grpc.ServerStub(channel)
    response = stub.ForwardPass(edge_server_pb2.Tensor(data=edge_output.detach().numpy().tobytes()))

    # 接收服务器传回的梯度
    server_grad = torch.from_numpy(np.frombuffer(response.data, dtype=np.float32).reshape(1, 64))
    print(f"Received gradient from server: {server_grad[0][:5]}\n")
    # 本地的反向传播
    backward()
