# Governance

OpenEngram is open to everyone, and every change is reviewed. This page explains who
decides what, which roles exist, and how to grow into a bigger role.

## Founder

**David Abramov** founded OpenEngram and is its lead maintainer. He has the final say on
the project's direction, on who receives a role, and on anything not covered here.
The founder credit in [NOTICE](NOTICE) stays with every copy of the project.

## Roles

| Role | Who it's for | What you can do | How you get it |
|---|---|---|---|
| **Contributor** | Anyone | Open issues, share knowledge, send pull requests, join Discussions | Your first accepted contribution. You're added to [CONTRIBUTORS.md](CONTRIBUTORS.md). |
| **Scientific advisor** | Doctors, scientists, professors | Suggest research questions, review results labelled `needs-expert-review`, guide modules. Listed by name and title on the website. | Apply with the [advisor form](https://github.com/LorDave1/openengram/issues/new?template=04-scientific-advisor.yml). Approved by the founder. |
| **Reviewer** | Regular contributors with good judgement | Review pull requests in an area (e.g. EEG), label and triage issues | Several quality contributions in that area, then apply with the [role form](https://github.com/LorDave1/openengram/issues/new?template=05-role-application.yml). |
| **Module lead** | People who drive one research module | Own the plan, code and results of a module; their approval is required for changes to it (via CODEOWNERS) | A module proposal that was accepted, or a reviewer who has led work in that module. |
| **Core team** | The project's most trusted people | Help set direction, approve releases and roles, represent the project. Listed on the website's Team section. | Invited by the founder after sustained, high-quality work as a module lead, reviewer or advisor. |

Roles describe responsibility, not seniority over people. A role can be stepped down from
at any time, and inactive roles (about 6 months without activity) may move to "emeritus",
keeping the credit.

## How decisions are made

1. **Everyday changes:** a pull request needs the automatic checks to pass and at least one
   approving review from a maintainer or the module lead (see `.github/CODEOWNERS`).
   Nothing is merged without review, including the founder's own larger changes when a
   reviewer is available.
2. **Scientific claims:** a result that a reader could take as medical needs a review from a
   scientific advisor (label `needs-expert-review`) before it appears on the website.
3. **Bigger decisions** (new modules, new datasets, changes to these rules): discussed openly in an
   issue or Discussion for at least a week. The core team aims for consensus; the founder
   decides if there is none.
4. **Roles:** applications are public issues. The founder approves, after hearing the core team.

## Quality and safety

- Everyone follows the [Code of Conduct](CODE_OF_CONDUCT.md), [SAFETY.md](SAFETY.md) and the scientific rules in [CONTRIBUTING.md](CONTRIBUTING.md).
- Maintainers may close contributions that are off-topic, unsafe, low-quality or that make medical claims.
- No one, at any role, may use the project to diagnose, treat or advise a real person.

## Money

Donations (see the website) go to running costs: compute, storage, domain, and possibly
small grants for contributors' research work. Spending is reported publicly once there is
any income. Donating never buys a role, a decision or credit for scientific work.

## The team

The current people in each role are listed in [`docs/team.json`](docs/team.json), which feeds
the Team section of the website. Changes to it go through a reviewed pull request.
