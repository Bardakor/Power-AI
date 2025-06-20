#!/bin/bash

# Power AI GCP Deployment Script
echo "🚀 Power AI - GCP Deployment Script"
echo "===================================="

# Set project configuration
PROJECT_ID="power-ai-463509"
REGION="us-central1"
SERVICE_NAME="power-ai"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Authenticate and set project
echo -e "${BLUE}🔐 Setting up authentication...${NC}"
gcloud auth login --account=basquiat.liam@gmail.com
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required GCP APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo -e "${BLUE}🏗️ Building and deploying with Cloud Build...${NC}"
gcloud builds submit --config=cloudbuild.yaml .

# Get the service URL
echo -e "${GREEN}✅ Deployment completed!${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

if [ ! -z "$SERVICE_URL" ]; then
    echo -e "${GREEN}🌐 Your Power AI application is available at:${NC}"
    echo -e "${YELLOW}$SERVICE_URL${NC}"
else
    echo -e "${RED}❌ Could not retrieve service URL${NC}"
fi

echo -e "${BLUE}📊 To view logs:${NC}"
echo "gcloud logs read $SERVICE_NAME --region=$REGION"

echo -e "${BLUE}🔧 To update the deployment:${NC}"
echo "./deploy.sh" 