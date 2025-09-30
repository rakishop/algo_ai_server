# Amazon Q Development Instructions

## Code Reuse Policy
Before creating any new code:
1. **Check existing codebase** for similar functionality
2. **Reuse existing methods/classes** if available
3. **Extend existing code** rather than duplicate
4. **Create new methods** only if no suitable code exists

## Directory Structure Rules
```
aiadvanceanalysisserver/
├── handlers/           # Message/data handlers
├── websockets/         # WebSocket connections
├── analyzers/          # Data analysis modules
├── notifications/      # Telegram/alert systems
├── utils/             # Utility functions
└── config/            # Configuration files
```

## Coding Standards
- **Minimal code**: Write only essential functionality
- **Reuse first**: Check existing imports, classes, methods
- **Appropriate directory**: Place code in correct folder
- **Single responsibility**: One class/function per task
- **No duplication**: Extend existing rather than recreate

## Before Any Coding Task
1. List existing relevant files
2. Check for reusable components
3. Identify appropriate directory
4. Extend existing or create minimal new code

## File Naming Convention
- `{purpose}_{type}.py` (e.g., `derivative_websocket.py`)
- Use existing naming patterns in project
- Keep names descriptive but concise