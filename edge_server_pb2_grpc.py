# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import edge_server_pb2 as edge__server__pb2


class ServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ForwardPass = channel.unary_unary(
                '/edge_server.Server/ForwardPass',
                request_serializer=edge__server__pb2.Tensor.SerializeToString,
                response_deserializer=edge__server__pb2.Tensor.FromString,
                )


class ServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ForwardPass(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ForwardPass': grpc.unary_unary_rpc_method_handler(
                    servicer.ForwardPass,
                    request_deserializer=edge__server__pb2.Tensor.FromString,
                    response_serializer=edge__server__pb2.Tensor.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'edge_server.Server', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Server(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ForwardPass(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/edge_server.Server/ForwardPass',
            edge__server__pb2.Tensor.SerializeToString,
            edge__server__pb2.Tensor.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
