---
layout: basic
---

## Add a keyboard shortcut in vscode

This was a bit tricky (atleast for me) as I could not find the link to ```keybindings.json``` from the keyboard shortcuts tab.

Soo see [this page](https://code.visualstudio.com/docs/getstarted/keybindings)

Basically you can open the ```keybindings.json``` from the Command Palette (Ctrl+Shift+P) by searching for ```Open Keyboard Shortcuts (JSON)```

Then add keyboard shortcuts in the following format (below shortcut cleans imports on Python files)

```
{
        "key": "shift+alt+r",
        "command": "editor.action.codeAction",
        "args": {
            "kind": "source.unusedImports",
        }
}
```


