# Comparison: Hermes memory, Orca, RTK, and Graphify

This comparison uses the checked-in Hermes and Orca implementations and their READMEs, plus the official RTK and Graphify project pages.

| Tool | Context problem | Intervention point | Durable data |
|---|---|---|---|
| markdown-memory-hermes | Replaying old conversation turns | Pre-request context selection; turn-completion recording | Per-session working state and append-only transcript |
| markdown-memory-orca | Reorientation and loss of project decisions across sessions | System-prompt injection plus explicit memory tool in OpenCode | Shared global/project Markdown files |
| RTK | Verbose shell/tool output | Command execution through a CLI proxy | Not a conversation transcript archive |
| Graphify | Re-reading code to understand structure | Project graph creation and on-demand or installed query workflow | Queryable codebase graph and reports |

RTK can reduce command results before they enter the transcript; Graphify can help answer code-structure questions; Hermes limits prior conversation replay; Orca stores distilled project state for later sessions. The layers can complement each other, but savings are not automatically additive.

## Verified distinctions

- Hermes is a Hermes Agent context-engine plugin. When usable state exists, it keeps the current turn and replaces older turns with a digest.
- Orca is currently an OpenCode plugin. It writes shared Markdown state and injects a bounded system-prompt block.
- RTK focuses on reducing token-heavy command output.
- Graphify documents a queryable knowledge graph with local AST-based code parsing and optional model-assisted processing for non-code materials.

## Sources

- [Hermes](https://github.com/ivanbayuf/markdown-memory-hermes)
- [Orca](https://github.com/ivanbayuf/markdown-memory-orca)
- [RTK](https://github.com/rtk-ai/rtk)
- [Graphify](https://github.com/Graphify-Labs/graphify)

