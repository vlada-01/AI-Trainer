# AI-Trainer

**Transferring from platfrom to developer use-case in progress - Not usable right now**

A configuration-driven AI training system that orchestrates the full machine learning lifecycle: dataset preparation, model training, evaluation, post-processing, experiment tracking, and containerized execution with GPU support.

The goal of this project is to simulate a **production-grade MLOps system** where ML experiments are fully reproducible and defined via structured configuration files, without hardcoded pipelines.

## Requirements

To run this project locally:

- Docker  
- Docker Compose  
- NVIDIA GPU  
- NVIDIA Drivers  
- NVIDIA Container Toolkit  

**How to Start the System** - docker compose up -d --build

---

## Key Features

- Configuration-driven ML pipelines (Pydantic-based schemas)
- Dataset preparation (train/val/test splits, transforms, metadata logging)
- Model training & fine-tuning with PyTorch (CPU & GPU support)
- Post-processing pipeline (calibration, thresholding)
- Experiment tracking & model registry using MLflow
- Asynchronous job execution via FastAPI
- Containerized execution with Docker & Docker Compose
- GPU acceleration via NVIDIA Container Toolkit
- Error analysis tables, confusion matrix & metrics reporting
- Reproducible experiments (configs = experiments)

---

## Architecture Overview

The system is designed as a modular, configuration-driven MLOps platform where each component has a clear responsibility and can be extended independently.

- **FastAPI Backend**  
  Acts as the main entry point for users. It exposes endpoints to trigger dataset preparation, model training, evaluation, and post-processing jobs. All requests are validated against structured configuration schemas.

- **Configuration Layer (Pydantic Schemas)**  
  All experiments are defined via configuration files (datasets, models, training, post-processing). This makes experiments fully reproducible and removes hardcoded logic from the training pipeline.

- **Training & Evaluation Engine (PyTorch)**  
  Responsible for model creation, fine-tuning, training loops, and evaluation. The engine supports both CPU and GPU execution and can be extended with new models and training strategies.

- **Post-Processing Pipeline**  
  Applies optional steps such as calibration and thresholding on top of raw model predictions. Each post-processor is modular and can be chained.

- **Experiment Tracking (MLflow)**  
  All runs, metrics, artifacts, datasets, and trained models are logged to MLflow. This enables experiment comparison, reproducibility, and model versioning.

- **Containerized Runtime (Docker + NVIDIA Container Toolkit)**  
  The entire platform runs inside Docker containers, ensuring consistent environments across machines. GPU acceleration is enabled via NVIDIA Container Toolkit, allowing training to scale from CPU to GPU without code changes.

This architecture enables a clean separation of concerns between orchestration, configuration, training logic, and infrastructure, closely mirroring production-grade ML systems.



**Project Status:** This system is under active development. Some features may be incomplete, and certain edge cases may not yet be fully handled. Stability and test coverage are continuously improving.
