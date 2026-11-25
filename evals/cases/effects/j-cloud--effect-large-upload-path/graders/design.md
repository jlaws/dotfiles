---
type: llm
focus: { source: file, path: upload-design.md }
---

All claims are required.
PASS if upload-design.md sends these files to S3 as multipart uploads (in parts), not as one PUT of the whole file.
PASS if no presigned URL can expire while a slow upload still needs it: a URL that covers the whole file stays valid for at least 1 hour, and part URLs either stay valid for at least 1 hour or are signed on demand, so an expired part URL is replaced instead of failing the upload.
PASS if incomplete multipart uploads are aborted after a set number of days, by a bucket lifecycle rule or a scheduled cleanup job.
FAIL if upload-design.md sends the whole file through one presigned URL valid for less than 1 hour (such as the current ExpiresIn=300 in api/presign.py), or signs every part URL up front with an expiry under 1 hour and no way to get a fresh one.
