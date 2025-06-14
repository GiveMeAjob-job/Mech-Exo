# Terraform Configuration for Mech-Exo Cold Backup Infrastructure
# Creates second region EC2 + EBS for disaster recovery

terraform {
  required_version = ">= 0.14"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "region" {
  description = "Backup region for disaster recovery"
  type        = string
  default     = "us-west-2"
}

variable "primary_region" {
  description = "Primary region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for backup"
  type        = string
  default     = "t3.medium"
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "backup"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mech-exo"
}

# Configure AWS provider for backup region
provider "aws" {
  alias  = "backup"
  region = var.region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      Region      = var.region
      Purpose     = "disaster-recovery"
      Terraform   = "true"
    }
  }
}

# Data sources
data "aws_availability_zones" "backup" {
  provider = aws.backup
  state    = "available"
}

data "aws_ami" "ubuntu" {
  provider    = aws.backup
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# S3 bucket for backup storage
resource "aws_s3_bucket" "backup" {
  provider = aws.backup
  bucket   = "${var.project_name}-backup-${var.region}-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "backup" {
  provider = aws.backup
  bucket   = aws_s3_bucket.backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "backup" {
  provider = aws.backup
  bucket   = aws_s3_bucket.backup.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backup" {
  provider = aws.backup
  bucket   = aws_s3_bucket.backup.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# VPC for backup infrastructure
resource "aws_vpc" "backup" {
  provider             = aws.backup
  cidr_block           = "10.1.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-backup-vpc"
  }
}

resource "aws_internet_gateway" "backup" {
  provider = aws.backup
  vpc_id   = aws_vpc.backup.id

  tags = {
    Name = "${var.project_name}-backup-igw"
  }
}

resource "aws_subnet" "backup" {
  provider                = aws.backup
  vpc_id                  = aws_vpc.backup.id
  cidr_block              = "10.1.1.0/24"
  availability_zone       = data.aws_availability_zones.backup.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-backup-subnet"
  }
}

resource "aws_route_table" "backup" {
  provider = aws.backup
  vpc_id   = aws_vpc.backup.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.backup.id
  }

  tags = {
    Name = "${var.project_name}-backup-rt"
  }
}

resource "aws_route_table_association" "backup" {
  provider       = aws.backup
  subnet_id      = aws_subnet.backup.id
  route_table_id = aws_route_table.backup.id
}

# Security group for backup instance
resource "aws_security_group" "backup" {
  provider    = aws.backup
  name        = "${var.project_name}-backup-sg"
  description = "Security group for backup instance"
  vpc_id      = aws_vpc.backup.id

  # SSH access (restrict to specific IPs in production)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # IB Gateway port
  ingress {
    from_port   = 7497
    to_port     = 7497
    protocol    = "tcp"
    cidr_blocks = ["10.1.0.0/16"]
  }

  # TWS port
  ingress {
    from_port   = 7496
    to_port     = 7496
    protocol    = "tcp"
    cidr_blocks = ["10.1.0.0/16"]
  }

  # Prometheus metrics
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["10.1.0.0/16"]
  }

  # Health check endpoint
  ingress {
    from_port   = 8050
    to_port     = 8050
    protocol    = "tcp"
    cidr_blocks = ["10.1.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-backup-sg"
  }
}

# IAM role for backup instance
resource "aws_iam_role" "backup_instance" {
  provider = aws.backup
  name     = "${var.project_name}-backup-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "backup_instance" {
  provider = aws.backup
  name     = "${var.project_name}-backup-instance-policy"
  role     = aws_iam_role.backup_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.backup.arn,
          "${aws_s3_bucket.backup.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeVolumes",
          "ec2:DescribeSnapshots",
          "ec2:CreateSnapshot",
          "ec2:DeleteSnapshot",
          "ec2:DescribeInstances",
          "ec2:CreateTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "route53:GetHostedZone",
          "route53:ListHostedZones",
          "route53:ChangeResourceRecordSets",
          "route53:GetChange"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "backup_instance" {
  provider = aws.backup
  name     = "${var.project_name}-backup-instance-profile"
  role     = aws_iam_role.backup_instance.name
}

# EBS volume for DuckDB storage
resource "aws_ebs_volume" "backup_data" {
  provider          = aws.backup
  availability_zone = data.aws_availability_zones.backup.names[0]
  size              = 100  # 100GB for DuckDB files
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "${var.project_name}-backup-data"
    Type = "backup-storage"
  }
}

# EC2 instance for backup
resource "aws_instance" "backup" {
  provider                    = aws.backup
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.backup.id
  vpc_security_group_ids      = [aws_security_group.backup.id]
  iam_instance_profile        = aws_iam_instance_profile.backup_instance.name
  associate_public_ip_address = true

  user_data = base64encode(templatefile("${path.module}/backup_user_data.sh", {
    s3_bucket = aws_s3_bucket.backup.bucket
    region    = var.region
  }))

  tags = {
    Name   = "${var.project_name}-backup-instance"
    Type   = "backup"
    Region = var.region
  }

  # Prevent accidental termination
  disable_api_termination = true
}

# Attach EBS volume to instance
resource "aws_volume_attachment" "backup_data" {
  provider    = aws.backup
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.backup_data.id
  instance_id = aws_instance.backup.id
}

# Route53 record for backup instance (for DNS failover)
data "aws_route53_zone" "main" {
  count = var.enable_dns_failover ? 1 : 0
  name  = var.domain_name
}

variable "enable_dns_failover" {
  description = "Enable DNS failover setup"
  type        = bool
  default     = false
}

variable "domain_name" {
  description = "Domain name for DNS failover"
  type        = string
  default     = "mech-exo.com"
}

resource "aws_route53_record" "backup" {
  count           = var.enable_dns_failover ? 1 : 0
  provider        = aws.backup
  zone_id         = data.aws_route53_zone.main[0].zone_id
  name            = "backup.${var.domain_name}"
  type            = "A"
  ttl             = "60"
  records         = [aws_instance.backup.public_ip]
  allow_overwrite = true
}

# CloudWatch alarms for backup instance
resource "aws_cloudwatch_metric_alarm" "backup_instance_status" {
  provider            = aws.backup
  alarm_name          = "${var.project_name}-backup-instance-status"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = "60"
  statistic           = "Maximum"
  threshold           = "0"
  alarm_description   = "This metric monitors backup instance status"

  dimensions = {
    InstanceId = aws_instance.backup.id
  }
}

# Outputs
output "backup_instance_id" {
  description = "ID of the backup instance"
  value       = aws_instance.backup.id
}

output "backup_instance_public_ip" {
  description = "Public IP of the backup instance"
  value       = aws_instance.backup.public_ip
}

output "backup_s3_bucket" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.backup.bucket
}

output "backup_region" {
  description = "Backup region"
  value       = var.region
}

output "backup_dns_record" {
  description = "DNS record for backup instance"
  value       = var.enable_dns_failover ? aws_route53_record.backup[0].fqdn : "Not configured"
}

output "backup_data_volume_id" {
  description = "EBS volume ID for backup data"
  value       = aws_ebs_volume.backup_data.id
}