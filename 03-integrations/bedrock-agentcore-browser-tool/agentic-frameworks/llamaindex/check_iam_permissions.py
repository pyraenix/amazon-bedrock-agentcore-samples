#!/usr/bin/env python3
"""
Check and create IAM permissions for AgentCore Browser Tool
"""

import boto3
import json
from datetime import datetime

def check_current_permissions():
    """Check current user/role permissions"""
    print("🔐 Checking Current IAM Permissions...")
    
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print(f"  ✅ Account ID: {identity['Account']}")
        print(f"  ✅ User/Role ARN: {identity['Arn']}")
        
        # Check if we're using a role or user
        if ':assumed-role/' in identity['Arn']:
            role_name = identity['Arn'].split('/')[-2]
            print(f"  ✅ Using IAM Role: {role_name}")
            return 'role', role_name
        elif ':user/' in identity['Arn']:
            user_name = identity['Arn'].split('/')[-1]
            print(f"  ✅ Using IAM User: {user_name}")
            return 'user', user_name
        else:
            print(f"  ⚠️  Unknown identity type")
            return 'unknown', None
            
    except Exception as e:
        print(f"  ❌ Cannot check identity: {e}")
        return None, None

def create_agentcore_browser_policy():
    """Create IAM policy for AgentCore Browser Tool"""
    print("📋 Creating AgentCore Browser Tool IAM Policy...")
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    bucket_name = f"agentcore-browser-recordings-{account_id}"
    
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockAgentCoreAccess",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock-agentcore:*",
                    "bedrock-agentcore-control:*"
                ],
                "Resource": "*"
            },
            {
                "Sid": "S3BrowserRecordingsAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:PutObjectAcl",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            },
            {
                "Sid": "CloudWatchLogsAccess",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams"
                ],
                "Resource": f"arn:aws:logs:*:{account_id}:log-group:/aws/bedrock/agentcore/*"
            },
            {
                "Sid": "IAMPassRoleForAgentCore",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": f"arn:aws:iam::{account_id}:role/AgentCore*",
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": "bedrock.amazonaws.com"
                    }
                }
            }
        ]
    }
    
    policy_name = "AgentCoreBrowserToolPolicy"
    
    try:
        iam = boto3.client('iam')
        
        # Try to create the policy
        try:
            response = iam.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_document, indent=2),
                Description="Policy for AgentCore Browser Tool access"
            )
            policy_arn = response['Policy']['Arn']
            print(f"  ✅ Created policy: {policy_arn}")
            return policy_arn
            
        except iam.exceptions.EntityAlreadyExistsException:
            # Policy already exists, get its ARN
            policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
            print(f"  ✅ Policy already exists: {policy_arn}")
            
            # Update the policy
            try:
                iam.create_policy_version(
                    PolicyArn=policy_arn,
                    PolicyDocument=json.dumps(policy_document, indent=2),
                    SetAsDefault=True
                )
                print(f"  ✅ Updated policy to latest version")
            except Exception as e:
                print(f"  ⚠️  Could not update policy: {e}")
            
            return policy_arn
            
    except Exception as e:
        print(f"  ❌ Failed to create policy: {e}")
        return None

def create_agentcore_service_role():
    """Create service role for AgentCore Browser Tool"""
    print("👤 Creating AgentCore Service Role...")
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    # Trust policy for the service role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    role_name = "AgentCoreBrowserToolServiceRole"
    
    try:
        iam = boto3.client('iam')
        
        # Create the role
        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Service role for AgentCore Browser Tool"
            )
            role_arn = response['Role']['Arn']
            print(f"  ✅ Created service role: {role_arn}")
            
        except iam.exceptions.EntityAlreadyExistsException:
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            print(f"  ✅ Service role already exists: {role_arn}")
        
        # Attach the policy to the role
        policy_arn = f"arn:aws:iam::{account_id}:policy/AgentCoreBrowserToolPolicy"
        try:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"  ✅ Attached policy to service role")
        except Exception as e:
            print(f"  ⚠️  Could not attach policy: {e}")
        
        return role_arn
        
    except Exception as e:
        print(f"  ❌ Failed to create service role: {e}")
        return None

def attach_policy_to_current_identity(policy_arn, identity_type, identity_name):
    """Attach policy to current user/role"""
    print(f"🔗 Attaching Policy to Current {identity_type.title()}...")
    
    try:
        iam = boto3.client('iam')
        
        if identity_type == 'user':
            iam.attach_user_policy(
                UserName=identity_name,
                PolicyArn=policy_arn
            )
            print(f"  ✅ Attached policy to user: {identity_name}")
            
        elif identity_type == 'role':
            iam.attach_role_policy(
                RoleName=identity_name,
                PolicyArn=policy_arn
            )
            print(f"  ✅ Attached policy to role: {identity_name}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to attach policy: {e}")
        return False

def main():
    """Set up IAM permissions for AgentCore Browser Tool"""
    print("🚀 AgentCore Browser Tool IAM Setup")
    print("=" * 50)
    
    # Check current permissions
    identity_type, identity_name = check_current_permissions()
    
    if not identity_type:
        print("❌ Cannot determine current identity. Exiting.")
        return
    
    print()
    
    # Create the policy
    policy_arn = create_agentcore_browser_policy()
    
    if not policy_arn:
        print("❌ Failed to create policy. Exiting.")
        return
    
    print()
    
    # Create service role
    service_role_arn = create_agentcore_service_role()
    
    print()
    
    # Attach policy to current identity
    if identity_name:
        success = attach_policy_to_current_identity(policy_arn, identity_type, identity_name)
        
        if success:
            print("✅ Policy attached successfully")
        else:
            print("⚠️  Could not attach policy automatically")
    
    print()
    print("📋 Summary:")
    print("=" * 30)
    print(f"✅ Policy ARN: {policy_arn}")
    if service_role_arn:
        print(f"✅ Service Role ARN: {service_role_arn}")
    print(f"✅ Current Identity: {identity_type} - {identity_name}")
    
    print()
    print("🎯 Next Steps:")
    print("1. Wait a few minutes for IAM changes to propagate")
    print("2. Go back to AWS Console and complete Browser Tool creation")
    print("3. Use the S3 URI from the previous step")
    print("4. Test the Browser Tool once created")
    
    # Save configuration
    config = {
        "policy_arn": policy_arn,
        "service_role_arn": service_role_arn,
        "identity_type": identity_type,
        "identity_name": identity_name,
        "created_at": datetime.now().isoformat()
    }
    
    with open("agentcore_iam_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 IAM configuration saved to: agentcore_iam_config.json")

if __name__ == "__main__":
    main()