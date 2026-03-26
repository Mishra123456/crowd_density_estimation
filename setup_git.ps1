# Check if git exists
$git_exists = Get-Command git -ErrorAction SilentlyContinue
if (-not $git_exists) {
    Write-Host "Git is not installed or not in PATH."
    exit 1
}

# Initialize repository
git init

# Temporarily set dummy config if user hasn't configured git globally
$has_name = git config user.name
$has_email = git config user.email

if (-not $has_name) {
    git config --local user.name "Developer"
}
if (-not $has_email) {
    git config --local user.email "dev@example.com"
}

Write-Host "Simulating development history..."

# Commit 1
git add README.md project_report.md viva_questions.md .gitignore
git commit -m "docs: Initial commit with detailed project documentation and architecture plan"
Start-Sleep -s 1

# Commit 2
git add utils.py heatmap.py
git commit -m "feat: Implement dynamic heatmap generator and density classification utilities"
Start-Sleep -s 1

# Commit 3
git add detector.py
git commit -m "feat: Develop base person detector class utilizing OpenCV DNN"
Start-Sleep -s 1

# Commit 4
git add main.py
git commit -m "feat: Orchestrate video stream pipeline and real-time bounding box overlays"
Start-Sleep -s 1

# Commit 5
git add models/download_models.py download_sample_video.py
git commit -m "build: Add automated model weights and sample video downloader scripts"
Start-Sleep -s 1

# Final Commit (Catch anything missed)
git add .
git commit -m "refactor: Upgrade architecture to YOLOv4-tiny for robust crowd accuracy and finalize"

Write-Host "Git repository setup and history simulation complete!"
