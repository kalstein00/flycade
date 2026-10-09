---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets.

If the user passes a ticket reference, fetch it from the issue tracker and state its title before starting. If the reference is ambiguous, ask.

Call the Skill tool with "tdd" where possible, at pre-agreed seams.

Run typechecking regularly, single test files regularly, and the full test suite once at the end.

Once done, call the Skill tool with "code-review" to review the work.

Commit your work to the current branch.

When implementing a ticket, finish by updating the issue tracker:

- If all acceptance criteria are satisfied, comment with the commit, implementation summary, and verification results, then close the ticket.
- State whether the commit has been pushed.
- If any acceptance criteria remain unmet, leave the ticket open and comment with the remaining work and blockers.
