# Repository Copilot Instructions

## Language

1. The user communicates in Simplified Chinese. All substantive responses, explanations, analysis, and summaries must be written in Simplified Chinese.
2. Code, command-line examples, file paths, and technical identifiers should remain in their original form (English/ASCII).

## Path notation and shell usage

1. The default terminal in this repository is Git Bash.
2. When writing runnable shell examples, use Git Bash / Bash syntax instead of PowerShell syntax.
3. When a multi-line shell command needs explicit continuation, use `\`; do not use PowerShell backticks.
4. In runnable commands, prefer Unix-style forward-slash paths such as `src/data/bounding-box/my-script.py` and `models/bbox_output.pth`.

## ask_user usage

1. After each completed reply, call the `ask_user` tool to prompt for the next step.
2. A "completed reply" must first include the substantive result the user asked for, such as the answer, conclusion, summary, or completed task outcome.
3. Always provide the requested result in the response body before calling `ask_user`. Do not call `ask_user` as a substitute for the actual reply content.
4. The intended turn pattern is: answer the user's current request -> call `ask_user` -> wait for the user's reply -> answer that new request -> call `ask_user` again.
5. Continue doing this after every turn unless the user explicitly indicates they want to stop, end the conversation, or no longer wants follow-up questions.
6. Keep each `ask_user` prompt short and focused on the most natural next action.
7. Prefer multiple-choice options when there are a few clear next steps.
8. Do not ask plain-text follow-up questions in the response body; use the `ask_user` tool instead.

Recommended stop phrases to respect include: "结束", "停止", "不用继续", "先这样", "done", "stop", and similar clear endings.
