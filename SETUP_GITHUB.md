# Setting Up GitHub Repository

## Step 1: Initialize Git Repository (if not already done)
```bash
cd /projects/0/prjs1250/my_multimodal_WAD/fun_experiments/SST-WSVADL
git init
```

## Step 2: Add All Files
```bash
git add .
```

## Step 3: Make Initial Commit
```bash
git commit -m "Initial commit: SST-WSVADL project"
```

## Step 4: Create Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `SST-WSVADL` (or your preferred name)
3. Description: "Simplified Two-Stage Video Anomaly Detection with Weak Supervision"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

## Step 5: Connect Local Repository to GitHub
After creating the repository on GitHub, you'll see instructions. Use these commands:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/SST-WSVADL.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/SST-WSVADL.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Create Repository via GitHub CLI (if installed)
```bash
# Install GitHub CLI if needed: https://cli.github.com/
gh repo create SST-WSVADL --public --source=. --remote=origin --push
```

## Troubleshooting
- If you get authentication errors, you may need to set up a Personal Access Token:
  - Go to GitHub Settings → Developer settings → Personal access tokens
  - Generate a token with `repo` permissions
  - Use the token as your password when pushing

- If you need to change the remote URL:
  ```bash
  git remote set-url origin https://github.com/YOUR_USERNAME/SST-WSVADL.git
  ```

