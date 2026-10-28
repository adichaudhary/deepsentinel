#!/usr/bin/env bash
# Build the API image, push it to ECR, and run it as a public Fargate service on ECS.
# Requires: Docker, AWS CLI v2 configured with credentials (aws configure).
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-2}"
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO=deepsentinel
CLUSTER=deepsentinel
SERVICE=deepsentinel-api
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
cd "$(dirname "$0")/.."

echo "==> ECR repository"
aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" >/dev/null

echo "==> Build and push image"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$REPO:latest" .
docker tag "$REPO:latest" "$REGISTRY/$REPO:latest"
docker push "$REGISTRY/$REPO:latest"

echo "==> Task execution role"
if ! aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1; then
  aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 10
fi

echo "==> Log group and task definition"
aws logs create-log-group --log-group-name /ecs/deepsentinel --region "$AWS_REGION" 2>/dev/null || true
TASKDEF="$(mktemp)"
envsubst < deploy/ecs-task-definition.json > "$TASKDEF"
aws ecs register-task-definition --cli-input-json "file://$TASKDEF" --region "$AWS_REGION" >/dev/null

echo "==> Cluster and networking"
aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION" >/dev/null
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$AWS_REGION")"
SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[].SubnetId' --output text --region "$AWS_REGION" | tr '\t' ',')"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=deepsentinel-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$AWS_REGION")"
if [ "$SG_ID" = "None" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name deepsentinel-sg --description "DeepSentinel API" \
    --vpc-id "$VPC_ID" --query GroupId --output text --region "$AWS_REGION")"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$AWS_REGION" >/dev/null
fi

echo "==> Service"
NETWORK="awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"
if aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$AWS_REGION" \
     --query 'services[?status==`ACTIVE`].serviceName' --output text | grep -q "$SERVICE"; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --task-definition deepsentinel \
    --force-new-deployment --region "$AWS_REGION" >/dev/null
else
  aws ecs create-service --cluster "$CLUSTER" --service-name "$SERVICE" --task-definition deepsentinel \
    --desired-count 1 --launch-type FARGATE --network-configuration "$NETWORK" --region "$AWS_REGION" >/dev/null
fi

echo "==> Waiting for the service to become stable (a few minutes)"
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE" --region "$AWS_REGION"
TASK_ARN="$(aws ecs list-tasks --cluster "$CLUSTER" --service-name "$SERVICE" --query 'taskArns[0]' --output text --region "$AWS_REGION")"
ENI="$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$AWS_REGION" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)"
PUBLIC_IP="$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" --region "$AWS_REGION" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)"

echo
echo "API:  http://${PUBLIC_IP}:8000/health"
echo "Docs: http://${PUBLIC_IP}:8000/docs"
