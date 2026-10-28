#!/usr/bin/env bash
# Stop the ECS service and remove the cluster so it stops billing.
set -euo pipefail
AWS_REGION="${AWS_REGION:-us-east-2}"
aws ecs update-service --cluster deepsentinel --service deepsentinel-api --desired-count 0 --region "$AWS_REGION" >/dev/null
aws ecs delete-service --cluster deepsentinel --service deepsentinel-api --force --region "$AWS_REGION" >/dev/null
aws ecs delete-cluster --cluster deepsentinel --region "$AWS_REGION" >/dev/null
echo "Service and cluster deleted. The ECR image is kept for redeploys."
