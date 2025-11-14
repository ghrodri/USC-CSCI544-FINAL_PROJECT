#!/bin/bash
# Start the React frontend
cd frontend || { echo "frontend folder not found"; exit 1; }

if [ ! -d node_modules ] || [ ! -f node_modules/react/package.json ]; then
  echo "node_modules missing or incomplete. Cleaning and reinstalling..."
  rm -rf node_modules package-lock.json
  npm install --legacy-peer-deps react-scripts@5.0.1 typescript@4.9.5 ajv@^6.12.6 ajv-keywords@^3.5.2
else
  echo "Dependencies present. (Run 'rm -rf frontend/node_modules && ./start_frontend.sh' if issues persist.)"
fi

echo ""
echo "Starting React development server..."
echo "Frontend: http://localhost:3000"
echo "----------------------------------------"
exec npx react-scripts start
