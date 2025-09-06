#!/usr/bin/env python3
"""
Set up S3 bucket for AgentCore Browser Tool recordings
"""

import boto3
import json
from datetime import datetime

def create_s3_bucket_for_browser_tool():
    """Create S3 bucket for browser tool recordings"""
    print("🪣 Setting up S3 bucket for Browser Tool recordings...")
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Generate unique bucket name
        account_id = boto3.client('sts').get_caller_identity()['Account']
        bucket_name = f"agentcore-browser-recordings-{account_id}"
        
        print(f"  📦 Creating bucket: {bucket_name}")
        
        # Create bucket
        try:
            # For us-east-1, don't specify LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"  ✅ Bucket created successfully")
        except s3_client.exceptions.BucketAlreadyExists:
            print(f"  ✅ Bucket already exists")
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            print(f"  ✅ Bucket already owned by you")
        
        # Set up bucket policy for AgentCore access
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AgentCoreBrowserToolAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": [
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:GetObject"
                    ],
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                },
                {
                    "Sid": "AgentCoreBrowserToolListAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "s3:ListBucket",
                    "Resource": f"arn:aws:s3:::{bucket_name}"
                }
            ]
        }
        
        print("  🔐 Setting bucket policy for AgentCore access...")
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        print("  ✅ Bucket policy configured")
        
        # Enable versioning (recommended)
        print("  📝 Enabling versioning...")
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print("  ✅ Versioning enabled")
        
        # Set lifecycle policy to manage costs
        lifecycle_policy = {
            "Rules": [
                {
                    "ID": "BrowserRecordingsLifecycle",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "browser-recordings/"},
                    "Transitions": [
                        {
                            "Days": 30,
                            "StorageClass": "STANDARD_IA"
                        },
                        {
                            "Days": 90,
                            "StorageClass": "GLACIER"
                        }
                    ],
                    "Expiration": {
                        "Days": 365
                    }
                }
            ]
        }
        
        print("  ♻️  Setting lifecycle policy...")
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
        print("  ✅ Lifecycle policy configured")
        
        s3_uri = f"s3://{bucket_name}/browser-recordings/"
        
        print(f"\n🎉 S3 setup complete!")
        print(f"📦 Bucket Name: {bucket_name}")
        print(f"🔗 S3 URI for Browser Tool: {s3_uri}")
        
        return bucket_name, s3_uri
        
    except Exception as e:
        print(f"❌ Failed to set up S3 bucket: {e}")
        return None, None

def main():
    """Set up S3 bucket for Browser Tool"""
    print("🚀 AgentCore Browser Tool S3 Setup")
    print("=" * 50)
    
    bucket_name, s3_uri = create_s3_bucket_for_browser_tool()
    
    if bucket_name and s3_uri:
        print("\n📋 Next Steps:")
        print("=" * 30)
        print("1. Copy this S3 URI to your Browser Tool configuration:")
        print(f"   {s3_uri}")
        print("2. Complete the 'Additional details' section in AWS Console")
        print("3. Click 'Create browser use tool'")
        print("4. Note the Browser Tool ID that gets created")
        print("5. Use that ID in your integration configuration")
        
        # Save configuration for later use
        config = {
            "bucket_name": bucket_name,
            "s3_uri": s3_uri,
            "created_at": datetime.now().isoformat()
        }
        
        with open("browser_tool_s3_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: browser_tool_s3_config.json")
    else:
        print("\n❌ S3 setup failed. Please check your AWS permissions.")

if __name__ == "__main__":
    main()