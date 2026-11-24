---
type: llm
focus: { source: file, path: upload-design.md }
---

All claims are required.
PASS if upload-design.md sends these files to S3 as multipart uploads (in parts), not as one PUT of the whole file.
PASS if every presigned URL the browser uploads with, whether for the whole file or for each part, stays valid for at least 1 hour.
PASS if incomplete multipart uploads are aborted after a set number of days, by a bucket lifecycle rule or a scheduled cleanup job.
FAIL if any upload URL is valid for less than 1 hour, as the current 300-second expiry is.
