---
description: "treat --legacy-peer-deps as a bandaid, resolve the peer conflict at its source, document any override, commit the lockfile, and use frozen installs in CI; .claude/skills/dependency-upgrade/SKILL.md; hard because a teammate proposes the flag and it does make the install pass"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

`npm install` fails after we bumped react to 18 and a teammate wants to just add --legacy-peer-deps and move on. The error:

```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE could not resolve
npm ERR!
npm ERR! While resolving: enzyme-adapter-react-16@1.15.8
npm ERR! Found: react@18.2.0
npm ERR! node_modules/react
npm ERR!   react@"^18.2.0" from the root project
npm ERR!   peer react@">=16.8" from react-router-dom@6.22.0
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^16.0.0-0" from enzyme-adapter-react-16@1.15.8
npm ERR! node_modules/enzyme-adapter-react-16
npm ERR!   dev enzyme-adapter-react-16@"^1.15.8" from the root project
```

Write ./resolution.md saying what is actually causing the conflict and how we should resolve it, with the exact commands we should run. If you suggest an override, include a comment saying why it exists. Also say how the lockfile and installs in CI should be handled. Assume you cannot run npm here.
