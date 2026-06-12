# Research-to-Production ML Platform

A local-first MLOps platform for validating PyTorch model packages, registering model versions, deploying candidate models to staging, running smoke tests, and promoting approved models to production.

## V1 Scope

- Validate PyTorch model packages
- Register model metadata locally
- Serve models through FastAPI
- Build Docker containers
- Deploy to local Kubernetes using kind
- Separate staging and production namespaces
