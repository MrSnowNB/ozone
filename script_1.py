# Create requirements.txt
requirements = """psutil>=5.9.0
PyYAML>=6.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("Created requirements.txt")

# Create VS Code tasks.json
vscode_tasks = """{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "O3: Test Single Model",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_optimizer.py",
                "${input:modelName}"
            ],
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "O3: Test All Coding Models",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_optimizer.py",
                "qwen3-coder:30b",
                "orieg/gemma3-tools:27b-it-qat"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "O3: Test All RAG Models",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_optimizer.py",
                "liquid-rag:latest"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "O3: Test All Chat Models",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_optimizer.py",
                "qwen2.5:3b-instruct",
                "gemma3:latest"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "O3: Full Test Suite",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_optimizer.py",
                "qwen3-coder:30b",
                "orieg/gemma3-tools:27b-it-qat",
                "liquid-rag:latest",
                "qwen2.5:3b-instruct",
                "gemma3:latest"
            ],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        },
        {
            "label": "O3: Generate Summary Report",
            "type": "shell",
            "command": "python",
            "args": [
                "o3_report_generator.py"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "dependsOn": [],
            "problemMatcher": []
        },
        {
            "label": "O3: Install Dependencies",
            "type": "shell",
            "command": "pip",
            "args": [
                "install",
                "-r",
                "requirements.txt"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        }
    ],
    "inputs": [
        {
            "id": "modelName",
            "description": "Enter the model name to test",
            "default": "qwen3-coder:30b",
            "type": "promptString"
        }
    ]
}"""

# Create .vscode directory and tasks.json
import os
os.makedirs(".vscode", exist_ok=True)
with open(".vscode/tasks.json", "w") as f:
    f.write(vscode_tasks)

print("Created .vscode/tasks.json")

# Create launch.json for debugging
vscode_launch = """{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "O3: Debug Single Model",
            "type": "python",
            "request": "launch",
            "program": "o3_optimizer.py",
            "args": ["qwen2.5:3b-instruct"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {},
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "O3: Debug Report Generator",
            "type": "python",
            "request": "launch",
            "program": "o3_report_generator.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}"""

with open(".vscode/launch.json", "w") as f:
    f.write(vscode_launch)

print("Created .vscode/launch.json")