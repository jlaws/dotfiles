---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "for large browser uploads to S3, send multipart uploads with presigned URLs valid for 1 hour or more, and abort incomplete multipart uploads after a set number of days; .claude/references/cloud/file-storage-patterns.md; hard because a short presigned expiry reads as the secure default, and abandoned multipart parts are billed without showing in the bucket listing"
---

/kb:j-cloud redesign the video upload path described in ./docs/uploads.md (the current code is ./api/presign.py). Save the design to ./upload-design.md, covering how the file is sent, how the browser gets permission to upload, how long that permission stays valid, and what happens to the storage left by uploads that never finish.
