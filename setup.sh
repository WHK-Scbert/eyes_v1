#!/bin/bash

# Read the access key and secret access key from the text file
read -r ACCESS_KEY_ID SECRET_ACCESS_KEY REGION < "/path/to/access-keys.txt"

# Set the AWS CLI configuration
aws configure set aws_access_key_id "$ACCESS_KEY"
aws configure set aws_secret_access_key "$SECRET_ACCESS_KEY"

# Optional: Set the default region
aws configure set default.region "$REGION"
