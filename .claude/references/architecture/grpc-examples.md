# gRPC Code Examples

## Proto3 Schema

```protobuf
syntax = "proto3";
package myservice.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/field_mask.proto";

message User {
  string id = 1;
  string email = 2;
  string display_name = 3;
  UserRole role = 4;
  google.protobuf.Timestamp created_at = 5;
  map<string, string> metadata = 6;
  oneof contact {
    string phone = 7;
    string slack_id = 8;
  }
}

enum UserRole {
  USER_ROLE_UNSPECIFIED = 0;
  USER_ROLE_ADMIN = 1;
  USER_ROLE_MEMBER = 2;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}

message UpdateUserRequest {
  User user = 1;
  google.protobuf.FieldMask update_mask = 2;
}

service UserService {
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
  rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);
  rpc BulkCreateUsers(stream CreateUserRequest) returns (BulkResp);
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

## Python Server

```python
import grpc
from concurrent import futures
from grpc_reflection.v1alpha import reflection
import user_pb2, user_pb2_grpc

class UserServicer(user_pb2_grpc.UserServiceServicer):
    def __init__(self):
        self.users = {}

    def CreateUser(self, request, context):
        user = request.user
        if user.id in self.users:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(f"User {user.id} already exists")
            return user_pb2.CreateUserResponse()
        self.users[user.id] = user
        return user_pb2.CreateUserResponse(user=user)

    def GetUser(self, request, context):
        user = self.users.get(request.id)
        if not user:
            context.abort(grpc.StatusCode.NOT_FOUND, f"User {request.id} not found")
        return user

    def ListUsers(self, request, context):
        page_size = min(request.page_size or 50, 100)
        all_users = list(self.users.values())
        start = int(request.page_token) if request.page_token else 0
        page = all_users[start:start + page_size]
        next_token = str(start + page_size) if start + page_size < len(all_users) else ""
        return user_pb2.ListUsersResponse(users=page, next_page_token=next_token)

    def WatchUsers(self, request, context):
        """Server-streaming: push events while client connected."""
        import time
        while context.is_active():
            yield user_pb2.UserEvent(type="heartbeat")
            time.sleep(5)

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ("grpc.max_receive_message_length", 10 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 30000),
    ])
    user_pb2_grpc.add_UserServiceServicer_to_server(UserServicer(), server)
    SERVICE_NAMES = (user_pb2.DESCRIPTOR.services_by_name["UserService"].full_name,
                     reflection.SERVICE_NAME)
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
```

## Python Client

```python
class UserClient:
    def __init__(self, target="localhost:50051", timeout=5.0):
        self.channel = grpc.insecure_channel(
            target, options=[("grpc.lb_policy_name", "round_robin")])
        self.stub = user_pb2_grpc.UserServiceStub(self.channel)
        self.timeout = timeout

    def get_user(self, user_id):
        try:
            return self.stub.GetUser(
                user_pb2.GetUserRequest(id=user_id), timeout=self.timeout)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def list_all_users(self):
        """Auto-paginate through all users."""
        users, token = [], ""
        while True:
            resp = self.stub.ListUsers(
                user_pb2.ListUsersRequest(page_size=100, page_token=token),
                timeout=self.timeout)
            users.extend(resp.users)
            if not resp.next_page_token:
                break
            token = resp.next_page_token
        return users
```

## Server Auth Interceptor

```python
class AuthInterceptor(grpc.ServerInterceptor):
    def __init__(self, public_methods=None):
        self.public_methods = public_methods or set()

    def intercept_service(self, continuation, handler_call_details):
        if handler_call_details.method in self.public_methods:
            return continuation(handler_call_details)
        metadata = dict(handler_call_details.invocation_metadata)
        if not metadata.get("authorization", "").startswith("Bearer "):
            return grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "No token"))
        return continuation(handler_call_details)
```

## Client Retry Interceptor

```python
class RetryInterceptor(grpc.UnaryUnaryClientInterceptor):
    RETRYABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}

    def __init__(self, max_retries=3, base_delay=0.1):
        self.max_retries, self.base_delay = max_retries, base_delay

    def intercept_unary_unary(self, continuation, call_details, request):
        import time
        for attempt in range(self.max_retries + 1):
            response = continuation(call_details, request)
            if response.code() not in self.RETRYABLE or attempt == self.max_retries:
                return response
            time.sleep(self.base_delay * (2 ** attempt))
        return response
```

## Deadlines and Health Checking

```python
# Client: timeout
try:
    response = stub.GetUser(request, timeout=5.0)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Timed out")

# Server: propagate remaining deadline to downstream
def GetUser(self, request, context):
    remaining = context.time_remaining()
    if remaining < 0.5:
        context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Insufficient time")
    return other_stub.Lookup(req, timeout=remaining - 0.1)

# Health checking
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

health_servicer = health.HealthServicer()
health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
health_servicer.set("myservice.v1.UserService", health_pb2.HealthCheckResponse.SERVING)
```
