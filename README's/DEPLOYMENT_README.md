# 🚀 Power AI - GCP Deployment Guide

This guide will help you deploy your Power AI application to Google Cloud Platform (GCP) using Docker and Cloud Run.

## 📋 Prerequisites

1. **Google Cloud Account**: Your account `basquiat.liam@gmail.com` with the required permissions
2. **GCP Project**: `power-ai-463509` (already configured)
3. **Google Cloud SDK**: Install from https://cloud.google.com/sdk/docs/install

## 🔧 Quick Setup

### 1. Install Google Cloud CLI

```bash
# macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login --account=basquiat.liam@gmail.com
gcloud config set project power-ai-463509
```

### 3. Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 🚀 Deployment Options

### Option 1: One-Click Deployment (Recommended)

Simply run the deployment script:

```bash
./deploy.sh
```

This script will:
- ✅ Check dependencies
- 🔐 Set up authentication
- 🔧 Enable required APIs
- 🏗️ Build Docker image
- 🚀 Deploy to Cloud Run
- 🌐 Provide the application URL

### Option 2: Manual Deployment

#### Step 1: Build and Deploy with Cloud Build

```bash
gcloud builds submit --config=cloudbuild.yaml .
```

#### Step 2: Get Service URL

```bash
gcloud run services describe power-ai --region=us-central1 --format="value(status.url)"
```

### Option 3: Local Docker Testing

Before deploying, you can test locally:

```bash
# Build the Docker image
docker build -t power-ai .

# Run locally
docker run -p 8080:8080 power-ai

# Access at: http://localhost:8080
```

## 📊 Application Features

Your deployed Power AI application includes:

- **📈 Real-time Power Monitoring**: Live dashboards for UPS systems, PDUs, and energy meters
- **🤖 ML Predictions**: Advanced machine learning models for power forecasting
- **⚡ Power Quality Analysis**: Electrical engineering analysis and anomaly detection
- **📅 Historical Analysis**: Time series analysis with customizable date ranges
- **🚨 Anomaly Detection**: Advanced MLOps pipeline for system health monitoring

## 📁 Data Included

The following CSV data will be deployed to the cloud:

- `outputs/csv_data/leituras311024_2031-231224_0730/leituras.csv` (405MB)
- `outputs/csv_data/leituras301224_1343_270225_0830/leituras.csv` (249MB)

## 🔧 Configuration

### Environment Variables

The application uses these environment variables:

- `PORT`: Server port (default: 8080)
- `PYTHONPATH`: Python path for imports

### Resource Allocation

- **CPU**: 2 cores
- **Memory**: 2GB RAM
- **Timeout**: 60 minutes
- **Scaling**: 1-10 instances based on load

## 📊 Monitoring & Logs

### View Application Logs

```bash
gcloud logs read power-ai --region=us-central1
```

### Monitor Performance

```bash
gcloud run services describe power-ai --region=us-central1
```

### Update Deployment

To update your application, simply run:

```bash
./deploy.sh
```

## 🛠️ Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   gcloud auth login --account=basquiat.liam@gmail.com
   ```

2. **Permission Denied**
   - Ensure your account has the required IAM roles:
     - Cloud Build Editor
     - Cloud Run Admin
     - Service Account User

3. **Large File Upload Issues**
   - The deployment includes 650MB+ of CSV data
   - Cloud Build has a 10GB disk limit (configured)
   - If issues persist, consider using Cloud Storage for large files

4. **Memory Issues**
   - Application configured with 2GB RAM
   - Increase if needed in `cloudbuild.yaml`

### Debug Commands

```bash
# Check build status
gcloud builds list --limit=5

# Check service status  
gcloud run services list

# View detailed logs
gcloud logs read power-ai --region=us-central1 --limit=50
```

## 🌐 Access Your Application

After successful deployment, your Power AI application will be available at:

```
https://power-ai-[random-hash]-uc.a.run.app
```

The exact URL will be displayed after deployment completion.

## 📱 Application Usage

1. **Select Dataset**: Choose from available power monitoring datasets
2. **Set Time Range**: Select from 24H, 7D, 30D, or All Data
3. **Run Analysis**: Click "Run MLOps Analysis" for advanced ML insights
4. **Explore Tabs**:
   - 📈 Real-time Monitoring
   - 🤖 ML Predictions  
   - ⚡ Power Quality
   - 📅 Historical Analysis
   - 🚨 Anomalies

## 🔒 Security

- Application runs with minimal privileges
- All data is processed in-memory
- No persistent storage of sensitive data
- Automatic scaling based on demand

## 💰 Cost Optimization

- Automatic scaling (1-10 instances)
- Pay-per-request pricing
- Optimized Docker image
- Efficient resource usage

---

**Need Help?** Contact support or check the GCP console at:
https://console.cloud.google.com/run?project=power-ai-463509 