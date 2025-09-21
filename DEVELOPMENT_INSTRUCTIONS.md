# Development Instructions

## Coding Guidelines

### Core Principles
1. **Minimize File Creation** - Don't create unnecessary files; reuse existing ones when possible
2. **Function Reusability** - Always reuse existing functions instead of duplicating code
3. **Test Organization** - Create separate `tests/` directory for all test files
4. **Documentation** - Always write and update API documentation
5. **Code Validation** - Always check and run code before committing

### Project Structure
```
historydata/
├── tests/           # All test files go here
├── docs/            # API documentation
├── *.py            # Main application files
└── *.json          # Data files
```

## Setup
```bash
pip install requests pandas scikit-learn numpy
```

## Usage
```bash
python api_tester.py
```

## Components
- `api_tester.py` - Main testing script
- `ml_analyzer.py` - ML analysis module  
- `data_processor.py` - Data processing utilities

## API Endpoints
- Market data: `/api/v1/market/*`
- Derivatives: `/api/v1/derivatives/*` 
- Indices: `/api/v1/indices/*`

## Development Workflow

### Before Writing Code
1. Check if existing functions can be reused
2. Identify minimal code needed for the requirement
3. Plan test file organization in `tests/` directory

### During Development
1. Write minimal, focused code
2. Reuse existing functions and utilities
3. Update API documentation as you code
4. Create tests in separate `tests/` directory

### Before Committing
1. Run all tests: `python -m pytest tests/`
2. Validate code functionality
3. Update documentation
4. Ensure no unnecessary files created

## Quality Assurance
- **Code Reuse**: Always check existing modules before writing new functions
- **Testing**: All test files must be in `tests/` directory
- **Documentation**: Keep `docs/api.md` updated with all endpoints
- **Validation**: Run code before any commit

## Output Files
- `live_api_responses.json` - Raw API responses
- `processed_stocks.json` - Processed stock data with ML features
- `docs/api.md` - API documentation (auto-updated)