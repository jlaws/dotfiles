---
description: "a user asks for a quality review of a module whose shortcuts carry SIMPLIFIED comments and a bare TODO; .claude/skills/code-quality/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I inherited this little session module and the previous dev left comments on the shortcuts they took. Before I build on it, can you go through it and tell me which of these are real problems and which are fine to leave?

```python
SESSIONS = {}

# SIMPLIFIED: tokens are the username, issue random ones when we have real users
def create_session(user):
    SESSIONS[user] = user
    return user

# SIMPLIFIED: in-memory dict, loses sessions on restart, move to redis past one process
def get_session(token):
    return SESSIONS.get(token)

# TODO
def purge_expired():
    pass
```
