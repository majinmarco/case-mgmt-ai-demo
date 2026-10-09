# Issue tracker: Jira

Issues and specs for this repo live in Jira project **CMAWD** on https://ethreesolutions.atlassian.net. Use the **Atlassian MCP** connector (`mcp__claude_ai_Atlassian_MCP__*`) with `cloudId: "https://ethreesolutions.atlassian.net"`. The older `mcp__claude_ai_Atlassian__*` connector has no access to this site.

## Issue types

- **Epic**: a spec (output of `/to-spec`).
- **Story**: a user-facing slice of a spec; child of its Epic.
- **Task**: technical work (spikes, ingestion, setup); child of its Epic.
- **Bug**: defects.
- **Subtask**: only for splitting a Story or Task.

## Conventions

- **Create an issue**: `createJiraIssue` (project `CMAWD`, issue type, summary, Markdown description). For a child issue, set the parent Epic.
- **Read an issue**: `getJiraIssue` with view `evidence`; `listJiraIssueComments` for comments.
- **List issues**: `searchJiraIssuesUsingJql`, e.g. `project = CMAWD AND labels = ready-for-agent AND statusCategory != Done`.
- **Comment on an issue**: `addOrEditJiraIssueComment`.
- **Apply / remove labels**: `editJiraIssue`. Labels are free-form strings, e.g. `ready-for-agent`.
- **Transition / close**: `discover` the transitions operation, then `transitionJiraIssue`.
- For any other operation, use only names returned by `discover`; never guess an operation name.

## When a skill says "publish to the issue tracker"

Create a CMAWD issue. A spec becomes an **Epic**.

## When a skill says "fetch the relevant ticket"

`getJiraIssue` with the issue key (e.g. `CMAWD-12`), including comments.
