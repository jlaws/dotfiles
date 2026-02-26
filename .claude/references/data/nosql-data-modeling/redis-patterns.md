# Redis Data Structure Patterns

Extended code examples for Redis hashes, sorted sets, streams, rate limiting, and caching.

## Core Patterns

```python
import redis

r = redis.Redis(decode_responses=True)

# Hash: user profile (better than serialized JSON -- update fields individually)
r.hset("user:456", mapping={"name": "Alice", "email": "alice@example.com", "login_count": "0"})
r.hincrby("user:456", "login_count", 1)
profile = r.hgetall("user:456")

# Sorted set: leaderboard
r.zadd("leaderboard:weekly", {"alice": 2500, "bob": 1800, "carol": 3200})
top_10 = r.zrevrange("leaderboard:weekly", 0, 9, withscores=True)
alice_rank = r.zrevrank("leaderboard:weekly", "alice")  # 0-indexed

# Sorted set: rate limiting (sliding window)
import time

def is_rate_limited(user_id: str, limit: int = 100, window_s: int = 60) -> bool:
    key = f"rate:{user_id}"
    now = time.time()
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, now - window_s)  # Remove old entries
    pipe.zadd(key, {f"{now}": now})                  # Add current request
    pipe.zcard(key)                                   # Count in window
    pipe.expire(key, window_s)                        # TTL cleanup
    _, _, count, _ = pipe.execute()
    return count > limit

# Stream: event log
r.xadd("events:orders", {"type": "created", "order_id": "ord_123", "user_id": "usr_456"})
# Read latest events
events = r.xrevrange("events:orders", count=10)

# Cache with TTL
r.setex("cache:product:789", 300, '{"name": "Widget", "price": 9.99}')  # 5min TTL
```
