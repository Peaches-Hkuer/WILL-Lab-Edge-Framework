import torch
import torch.nn as nn
import torch.optim as optim
import grpc
import edge_server_pb2
import edge_server_pb2_grpc
from concurrent import futures
import numpy as np

HOST_IP = '127.0.0.1'
PORT= 65121



class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.fc4(x))
        print(f"ServerNet - fc4 output: {x[0][:5]}")
        x = torch.relu(self.fc5(x))
        print(f"ServerNet - fc5 output: {x[0][:5]}")
        x = torch.relu(self.fc6(x))
        print(f"ServerNet - fc6 output: {x[0][:5]}")
        x = self.fc7(x)
        print(f"ServerNet - fc7 output: {x[0][:5]}")
        return x


server_net = ServerNet()
optimizer = optim.SGD(server_net.parameters(), lr=0.01)

def Load_Labels():
    return torch.tensor([0,0,1,0,1])

TEMP_LABELS = Load_Labels()

class Server(edge_server_pb2_grpc.ServerServicer):
    # 到时候如何获取标签？
    labelCount = 0
    def getLabel(self):
        singleLabel = TEMP_LABELS[self.labelCount ]
        self.labelCount += 1
        return torch.tensor([singleLabel])


    def ForwardPass(self, request, context):
        edge_output = torch.from_numpy(np.frombuffer(request.data, dtype=np.float32).reshape(1, 64))
        edge_output.requires_grad = True
        print(f"Received edge_output from edge device: {edge_output[0][:5]}")
        server_output = server_net(edge_output)
        print(f"ServerNet output: {server_output[0][:5]}")

        # 开始反向传播
        # 假设损失函数为交叉熵损失
        label = self.getLabel()
        loss = nn.CrossEntropyLoss()(server_output,label)
        print(f"Loss: {loss}")
        loss.backward()
        print(f"ServerNet - fc7 gradient: {server_net.fc7.weight.grad[0][:5]}")
        print(f"ServerNet - fc6 gradient: {server_net.fc6.weight.grad[0][:5]}")
        print(f"ServerNet - fc5 gradient: {server_net.fc5.weight.grad[0][:5]}")
        print(f"ServerNet - fc4 gradient: {server_net.fc4.weight.grad[0][:5]}")
        optimizer.step()
        print(f"ServerNet parameters updated")

        # 将梯度传回边缘设备
        edge_grad = edge_output.grad
        print(f"edge_grad: {edge_grad[0][:5]}")
        print(f"Gradient to be sent back to edge device: {edge_grad[0][:5]}")
        return edge_server_pb2.Tensor(data=edge_grad.detach().numpy().tobytes())


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
edge_server_pb2_grpc.add_ServerServicer_to_server(Server(), server)

server.add_insecure_port(f'{HOST_IP}:{PORT}')
server.start()
print(f"Server start at {HOST_IP}:{PORT} ...")

server.wait_for_termination()