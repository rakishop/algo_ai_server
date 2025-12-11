@echo off
echo "== Initializing repo =="
echo "# aimodelsearch" > README.md

REM Init git only if not already initialized
IF NOT EXIST ".git" git init

REM Create and switch to main branch from detached HEAD
git checkout -b main-temp
git branch -D main 2>nul
git branch -m main-temp main

REM Stage and commit all changes
git add .
git commit -m "new end point added for news" --allow-empty

REM Add remote
git remote remove origin 2>nul
git remote add origin https://github.com/rakishop/algo_ai_server.git

REM Force push to overwrite remote
git push -f origin main

echo === Commit Complete ===

REM Deploy (this assumes deploy script is correctly set up in package.json)
echo === Deploying the app ===
REM call npm run deploy

echo === Deployment Complete ===
pause
