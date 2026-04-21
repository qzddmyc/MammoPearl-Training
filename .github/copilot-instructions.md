# Repository Copilot Instructions

## Path notation and shell usage

1. The default terminal in this repository is Git Bash.
2. When writing runnable shell examples, use Git Bash / Bash syntax instead of PowerShell syntax.
3. When a multi-line shell command needs explicit continuation, use `\`; do not use PowerShell backticks.
4. In runnable commands, prefer Unix-style forward-slash paths such as `src/data/bounding-box/bbox-test-resnet50.py` and `models/bbox_resnet50.pth`.

## ask_user usage

1. After each completed reply, call the `ask_user` tool to prompt for the next step.
2. Continue doing this after every turn unless the user explicitly indicates they want to stop, end the conversation, or no longer wants follow-up questions.
3. Keep each `ask_user` prompt short and focused on the most natural next action.
4. Prefer multiple-choice options when there are a few clear next steps.
5. Do not ask plain-text follow-up questions in the response body; use the `ask_user` tool instead.

Recommended stop phrases to respect include: "结束", "停止", "不用继续", "先这样", "done", "stop", and similar clear endings.
