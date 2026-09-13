# Terraform Module Library

## Critical Rules

- **Always** run `terraform plan` and review the diff before `terraform apply` -- unreviewed applies destroy infrastructure
- **Never** store state files locally or in Git -- use remote backends (S3, GCS) with locking enabled
- **Always** pin provider and module versions to exact versions -- unpinned upgrades break infrastructure silently
- **Never** hardcode secrets in `.tf` files -- use variables with `sensitive = true` and inject from a secret store

## Module Structure

```
module-name/
├── main.tf          # Main resources
├── variables.tf     # Input variables
├── outputs.tf       # Output values
├── versions.tf      # Provider versions
├── examples/
│   └── complete/
│       ├── main.tf
│       └── variables.tf
└── tests/
    └── module_test.go
```

## AWS VPC Module Example

**main.tf:**
```hcl
resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = var.enable_dns_hostnames
  enable_dns_support   = var.enable_dns_support

  tags = merge({ Name = var.name }, var.tags)
}

resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge({
    Name = "${var.name}-private-${count.index + 1}"
    Tier = "private"
  }, var.tags)
}

resource "aws_internet_gateway" "main" {
  count  = var.create_internet_gateway ? 1 : 0
  vpc_id = aws_vpc.main.id

  tags = merge({ Name = "${var.name}-igw" }, var.tags)
}
```

**variables.tf:**
```hcl
variable "name" {
  description = "Name of the VPC"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for VPC"
  type        = string
  validation {
    condition     = can(regex("^([0-9]{1,3}\\.){3}[0-9]{1,3}/[0-9]{1,2}$", var.cidr_block))
    error_message = "CIDR block must be valid IPv4 CIDR notation."
  }
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
```

**outputs.tf:**
```hcl
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}
```

## Module Composition

```hcl
module "vpc" {
  source = "../../modules/aws/vpc"

  name               = "production"
  cidr_block         = "10.0.0.0/16"
  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

module "rds" {
  source = "../../modules/aws/rds"

  identifier     = "production-db"
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.large"
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids
}
```

## State Management

### Remote Backend (S3)

```hcl
terraform {
  backend "s3" {
    bucket         = "company-terraform-state"
    key            = "environments/production/vpc/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

- **Always** enable encryption + DynamoDB locking
- Key path: `environments/{env}/{module}/terraform.tfstate`
- Never share state between environments

### State Operations

```bash
# Import existing resource
terraform import aws_vpc.main vpc-abc123

# Move resource between modules
terraform state mv module.old.aws_vpc.main module.new.aws_vpc.main

# Remove from state (keep in cloud)
terraform state rm aws_vpc.main
```

## Variable Patterns

### Environment Files

```hcl
# environments/production.tfvars
environment = "production"
vpc_cidr    = "10.0.0.0/16"
min_size    = 3
max_size    = 10
```

```bash
terraform plan -var-file=environments/production.tfvars
```

### Sensitive Variables

```hcl
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

Never in `.tfvars`. Pass via: `TF_VAR_db_password`, `-var`, or secret manager data source.

## Gotchas

- **`count` vs `for_each`**: Use `for_each` with maps/sets. `count` causes index-shift issues when removing middle elements.
- **Provider version drift**: Pin exact versions in `versions.tf`. `~>` allows patch drift that breaks things.
- **Destroy ordering**: Terraform sometimes can't determine correct destroy order. Use `depends_on` for implicit dependencies.
- **State file secrets**: State contains all resource attributes in plaintext. Encrypt backend + restrict access.
- **Large state files**: Split into smaller root modules. One monolith state = slow plans + blast radius.

## Cross-References

- **devops:kubernetes-configuration** -- K8s manifest generation, security policies
- **devops:pipeline-design** -- CI/CD pipeline for Terraform (plan in PR, apply on merge)
