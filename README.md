## Overview

Welcome to the `overhead-object-detection` Repository. This repository is designed for high-performance object detection tasks on overhead imagery, such as satellite images, using state-of-the-art machine learning models. The repository supports various datasets, foundation models, custom models, and advanced fine-tuning techniques, including PEFT (Parameter-Efficient Fine-Tuning) and FSDP (Fully Sharded Data Parallel).

This repository is structured to facilitate ease of use, scalability, and maintainability, ensuring that both individual developers and teams can efficiently contribute and leverage its capabilities.

## Repository structure

```
rare-objects-ultimatum/
├── .github/
│   ├── issue_template/                 # Folder for issue templates to standardize issue reporting
│   │ ├── bug_report.md                 # Template for reporting bugs
│   │ ├── feature_request.md            # Template for requesting new features
│   │ └── custom_template.md            # Template for other types of issues
│   ├── pull_request_template.md        # Template to standardize pull request submissions
│   └── workflows/                      # GitHub Actions workflows for CI/CD and automation
│     ├── ci.yml                        # Continuous Integration pipeline configuration
│     ├── cd.yml                        # Continuous Deployment pipeline configuration
│     ├── security.yml                  # Workflow for running security scans
│     └── codeql-analysis.yml           # Workflow for CodeQL analysis for code security
├── .vscode/
│   ├── extensions.json                 # Recommended extensions for the project
│   ├── launch.json                     # Debugger configurations for running and testing the project
│   └── settings.json                   # Workspace-specific settings, such as formatting rules and linter configurations
├── assets                              # Contains static resources such as images, diagrams, and example files
│   ├── diagrams                        # Diagrams representing architecture, workflows, or other visual documentation
│   └── examples                        # Example datasets, configuration files, or other illustrative resources
├── automation                          # Scripts to automate various tasks such as deployment and data ingestion
│   ├── auto_deploy.py                  # Script to automate the deployment of models or services
│   ├── data_ingestion_pipeline.py      # Script to automate data ingestion and preprocessing pipelines
│   └── model_build_pipeline.py         # Script to automate the building and training of models
├── ci_cd                               # CI/CD pipeline configurations for different tools
│   ├── github_actions.yml              # CI/CD configuration for GitHub Actions
│   ├── jenkins_pipeline.groovy         # CI/CD pipeline configuration for Jenkins
│   └── security_scan.yml               # Configuration for automated security scanning during CI/CD
├── configs                             # Configuration files for different aspects of the project
│   ├── datasets                        # Dataset-specific configurations
│   │   ├── README.md                   # Documentation for dataset configuration files
│   │   ├── spacenet.yml                # Configuration for SpaceNet dataset
│   │   └── xview.yml                   # Configuration for xView dataset
│   ├── deployment                      # Deployment-specific configurations
│   │   ├── docker_deploy.yml           # Docker deployment configuration
│   │   └── kubernetes_deploy.yml       # Kubernetes deployment configuration
│   ├── environments                    # Environment-specific configurations
│   │   ├── development.yml             # Configuration for the development environment
│   │   ├── production.yml              # Configuration for the production environment
│   │   └── staging.yml                 # Configuration for the staging environment
│   ├── models                          # Model configuration files
│   │   ├── custom_model.yml            # Configuration for custom models
│   │   └── foundation_model.yml        # Configuration for foundation models
│   ├── monitoring                      # Monitoring and logging configurations
│   │   ├── grafana                     # Grafana dashboard configurations
│   │   └── prometheus.yml              # Prometheus monitoring configuration
│   ├── README.md                       # Documentation for the configs directory
│   └── training                        # Training-specific configurations
│       ├── default_training.yml        # Default training configuration
│       └── peft_training.yml           # Configuration for PEFT-based training
├── data                                # Directory for handling and processing data
│   ├── annotations                     # Scripts for handling and converting annotations
│   │   ├── convert_to_coco.py          # Convert annotations to COCO format
│   │   └── convert_to_voc.py           # Convert annotations to Pascal VOC format
│   ├── augmentation                    # Data augmentation scripts
│   │   ├── geometric.py                # Geometric transformations for data augmentation
│   │   └── photometric.py              # Photometric transformations for data augmentation
│   ├── governance                      # Data governance and versioning
│   │   └── data_governance.md          # Guidelines for data handling, compliance, and auditing
│   ├── preprocessors                   # Scripts for preprocessing data
│   │   └── image_converter.py          # Convert images to required formats
│   │   └── metadata_converter.py       # Convert metadata between different formats
│   ├── processed                       # Processed datasets (output of preprocessing)
│   ├── raw                             # Raw, unprocessed datasets
│   └── validation                      # Scripts for validating data and annotations
│       ├── geojson_validator.py        # Validate GEOJSON files against schema standards
│       ├── quality_checks.py           # Perform quality checks on datasets
│       └── schema_validation.py        # Validate data against predefined schemas
├── deployment                          # Deployment-related configurations and scripts
│   ├── docker-compose.yml              # Docker Compose configuration for local development and testing
│   ├── Dockerfile                      # Dockerfile for building the project container
│   └── k8s                             # Kubernetes deployment configurations
│       ├── deployment.yaml             # Kubernetes Deployment resource definition
│       ├── ingress.yaml                # Kubernetes Ingress resource definition
│       └── service.yaml                # Kubernetes Service resource definition
├── docker                              # Additional Docker-related configurations
│   ├── docker-compose.yml              # Another Docker Compose file, possibly for different use cases
│   └── Dockerfile                      # Dockerfile for building specific containers
├── docs                                # Project documentation
│   ├── api                             # API documentation (auto-generated or manually created)
│   ├── architecture.md                 # Documentation on the system architecture
│   ├── changelog.md                    # Changelog for tracking project updates and changes
│   ├── deployment.md                   # Documentation on how to deploy the project
│   ├── development.md                  # Development guidelines for contributors
│   ├── faq.md                          # Frequently asked questions about the project
│   ├── guides                          # User and developer guides
│   │   ├── deployment.md               # Step-by-step deployment guide
│   │   └── repo_tutorial.md            # Tutorial on how to use and navigate the repository
│   └── usage.md                        # Documentation on how to use the project or specific tools
├── experiments                         # Experiment results and logs
│   └── exp_<timestamp>                 # Folder for a specific experiment, identified by timestamp
│       ├── artifacts                   # Artifacts generated during the experiment (e.g., models, visualizations)
│       ├── checkpoints                 # Model checkpoints saved during training
│       ├── logs                        # Logs generated during the experiment
│       └── metrics                     # Metrics collected during the experiment
├── LICENSE                             # License file for the project
├── mlflow                              # MLflow configuration for tracking experiments
│   └── mlflow_config.yml               # MLflow setup and configuration file
├── models                              # Directory for model-related code and configurations
│   ├── adapters                        # Adapter implementations for model fine-tuning
│   │   ├── lora_adapter.py             # Implementation of LoRA adapter
│   │   ├── peft_adapter.py             # Implementation of PEFT adapter
│   │   └── qlora_adapter.py            # Implementation of Quantized LoRA adapter
│   ├── custom                          # Custom models specific to the project
│   │   └── custom_model.py             # Code for a custom model implementation
│   ├── finetuning                      # Scripts and utilities for fine-tuning models
│   │   ├── hyperparameter_search.py    # Script for hyperparameter tuning
│   │   ├── __init__.py                 # Initialize the finetuning module
│   │   ├── peft_evaluation.py          # Evaluation of models fine-tuned with PEFT
│   │   ├── peft_training.py            # Training script for PEFT models
│   │   └── peft_utils.py               # Utility functions for PEFT fine-tuning
│   ├── foundation                      # Foundation models used as the base for fine-tuning
│   │   └── foundation_model.py         # Implementation of the foundation model
│   ├── heads                           # Modular heads for different tasks (e.g., classification, detection)
│   │   ├── classification_head.py      # Head for classification tasks
│   │   ├── detection_head.py           # Head for object detection tasks
│   │   └── segmentation_head.py        # Head for segmentation tasks
│   └── model_registry.py               # Registry for managing and tracking models
├── notebooks                           # Jupyter notebooks for exploration and experimentation
│   ├── data_exploration.ipynb          # Notebook for exploring datasets
│   ├── experiment_notebook.ipynb       # Notebook for running and documenting experiments
│   ├── model_prototyping.ipynb         # Notebook for prototyping new models
│   └── visualization.ipynb             # Notebook for visualizing data and results
├── pyproject.toml                      # Project metadata and build system configuration
├── README.md                           # Main README file for the project, providing an overview and instructions
├── requirements.txt                    # List of Python dependencies for the project
├── scripts                             # Utility scripts for various tasks
│   ├── clean_data.py                   # Script to clean and preprocess raw data
│   ├── deploy_model.py                 # Script to deploy models
│   ├── evaluate_model.py               # Script to evaluate model performance
│   ├── export_model.py                 # Script to export models for deployment
│   ├── generate_reports.py             # Script to generate evaluation and performance reports
│   ├── inference.py                    # Script to run inference using trained models
│   ├── monitor_model.py                # Script to monitor model performance in production
│   ├── optimize_hparams.py             # Script for hyperparameter optimization
│   ├── preprocess_data.py              # Script to preprocess data for training
│   ├── setup_environment.py            # Script to set up the development environment
│   └── train_model.py                  # Main script for training models
├── security                            # Security-related policies and guidelines
│   ├── encryption_guidelines.md        # Guidelines for encryption and data protection
│   └── security_policies.md            # Security policies and procedures for the project
├── setup.py                            # Setup script for installing the project as a package
├── src                                 # Source code directory for core functionality
│   ├── data                            # Core data handling and processing modules
│   │   ├── augmentation.py             # Data augmentation utilities
│   │   ├── dataloader.py               # Custom data loading utilities
│   │   ├── dataset_factory.py          # Factory for creating dataset objects
│   │   └── __init__.py                 # Initialize the data module
│   ├── evaluation                      # Modules for evaluating model performance
│   │   ├── evaluator.py                # Evaluation logic and utilities
│   │   ├── __init__.py                 # Initialize the evaluation module
│   │   └── metrics.py                  # Custom metrics for evaluation
│   ├── models                          # Core model-building and handling modules
│   │   ├── __init__.py                 # Initialize the models module
│   │   ├── model_builder.py            # Utilities for building models from configurations
│   │   ├── model_registry.py           # Registry for managing model versions and configurations
│   │   └── peft_finetune.py            # PEFT-specific fine-tuning utilities
│   ├── training                        # Training loops, loss functions, and optimizers
│   │   ├── __init__.py                 # Initialize the training module
│   │   ├── loss_functions.py           # Custom loss functions
│   │   ├── optimizers.py               # Custom optimizers
│   │   └── trainer.py                  # Main training loop and logic
│   └── utils                           # Utility functions for various tasks
│       ├── config_parser.py            # Utilities for parsing configuration files
│       ├── file_utils.py               # File handling utilities
│       ├── __init__.py                 # Initialize the utils module
│       ├── logging.py                  # Logging utilities
│       ├── metrics.py                  # Metrics calculations and utilities
│       └── system_utils.py             # System-level utilities (e.g., environment checks)
└── tests                               # Unit and integration tests for the project
    ├── test_data.py                    # Tests for data handling modules
    ├── test_deployment.py              # Tests for deployment scripts and configurations
    ├── test_integration.py             # Integration tests for the whole system
    ├── test_models.py                  # Tests for model-related code
    ├── test_scripts.py                 # Tests for utility scripts
    ├── test_training.py                # Tests for training modules
    └── test_utils.py                   # Tests for utility functions
```

## Features

- **Multi-dataset Support**: Easily handle and preprocess datasets such as xView, SpaceNet, and more.
- **Foundation and Custom Models**: Leverage pre-trained models and create custom architectures tailored for overhead object detection.
- **Advanced Fine-tuning**: Utilize PEFT and FSDP for efficient and scalable fine-tuning.
- **Hyperparameter Optimization**: Integrated hyperparameter tuning with detailed tracking and reporting using MLflow.
- **Comprehensive Deployment**: Docker and Kubernetes configurations for easy model deployment in various environments.
- **Monitoring**: Prometheus and Grafana integration for monitoring deployed models.

## GitLab - Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

cd existing_repo
git remote add origin https://gitlab.dso.xc.nga.mil/rare-objects/rare-objects-ultimatum.git
git branch -M main
git push -uf origin main


## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.dso.xc.nga.mil/rare-objects/rare-objects-ultimatum/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
