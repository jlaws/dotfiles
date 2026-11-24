set -euo pipefail
mkdir -p docs api
cat > docs/uploads.md <<'MD'
# Inspection video uploads

- Field inspectors upload inspection videos from laptops, in the browser. Files are 300 MB to 4 GB.
- The browser uploads straight to the S3 bucket `inspections-raw` (us-east-1) with a presigned URL
  from our API. The API never handles the file bytes.
- Inspectors often close the laptop lid or lose the connection partway through an upload, and
  start again later.
- The bucket has no lifecycle rules today.
MD
cat > api/presign.py <<'PY'
import boto3

s3 = boto3.client("s3")


def presign_upload(key: str, content_type: str) -> str:
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": "inspections-raw", "Key": key, "ContentType": content_type},
        ExpiresIn=300,
    )
PY
