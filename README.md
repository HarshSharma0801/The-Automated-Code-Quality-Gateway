# 🚀 The Automated Code Quality Gateway

## 📋 Scenario

As a DevOps engineer in a fast-growing startup, you're tasked with preventing poorly formatted or buggy code from ever being merged into the main codebase. You need to automate code style enforcement and linting for a Python codebase.

## 🎯 Core Learning Objectives

- CI fundamentals
- Event-driven automation
- Code quality control
- Basic Git workflow integration

## 🛠️ Tech Stack & Rationale

| Technology                                                                                                             | Purpose              | Rationale                                                             |
| ---------------------------------------------------------------------------------------------------------------------- | -------------------- | --------------------------------------------------------------------- |
| ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)                                    | Version Control      | Industry standard for version control and collaboration               |
| ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)                           | Repository Hosting   | Manages code and hosts the repository                                 |
| ![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white) | CI/CD Pipeline       | Tight integration with GitHub, simplest way to create CI/CD pipelines |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)                           | Programming Language | Versatile and popular language, perfect for learning these concepts   |
| ![Black](https://img.shields.io/badge/Black-000000?style=flat&logo=python&logoColor=white)                             | Code Formatter       | Opinionated formatter ensuring uniform code style                     |
| ![Flake8](https://img.shields.io/badge/Flake8-3776AB?style=flat&logo=python&logoColor=white)                           | Linter               | Checks for errors and bad practices                                   |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)                        | Web Framework        | Modern async framework for concurrent API demonstrations              |

## 📋 Implementation Steps

### Step 1: Application Setup

Create FastAPI application with concurrent API calls to JSONPlaceholder

<img width="860" height="597" alt="Screenshot 2025-07-22 at 10 09 05 PM" src="https://github.com/user-attachments/assets/9d6ccd31-aceb-4e36-9518-87c4a4028b73" />


### Step 2: Configure Code Quality Tools

Set up Black, Flake8 with configuration files

<img width="983" height="459" alt="Screenshot 2025-07-22 at 10 00 59 PM" src="https://github.com/user-attachments/assets/a65b742e-fcdd-41ed-ac57-c82696684288" />


### Step 3: Create GitHub Actions Workflow

Implement automated quality checking on push/PR events

<img width="860" height="658" alt="Screenshot 2025-07-22 at 10 15 58 PM" src="https://github.com/user-attachments/assets/49f27a90-df8e-4faa-a1da-cb9f85fda3b2" />


### Step 4: Add Main Branch Protection rule for Quality Checks

Add qulaity checks on main branch so that it gets merged only when the github workflow passes 

<img width="860" height="597" alt="Screenshot 2025-07-22 at 10 12 28 PM" src="https://github.com/user-attachments/assets/0267e2d4-1ff7-4880-86e6-dae6938383bf" />


### Step 4: Push to main or make demo PR

<img width="860" height="597" alt="Screenshot 2025-07-22 at 10 13 59 PM" src="https://github.com/user-attachments/assets/5264b737-e3a0-45f9-bb45-83892848df97" />


### Step 5: Verify Checks in Actions Tab

Start the FastAPI server and test concurrent endpoints

<img width="860" height="597" alt="Screenshot 2025-07-22 at 10 14 18 PM" src="https://github.com/user-attachments/assets/06028d53-4803-40cb-8feb-8664985c0df0" />


## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quality checks
black --check . && flake8 . 

# Start the application
python main.py
```



