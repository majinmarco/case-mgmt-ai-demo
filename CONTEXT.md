# Permit Review Assistant

An AI assistant inside a case management system that reviews business Permit Requests against Puerto Rico permit regulations and produces a cited Recommendation for a human Reviewer. The current corpus (RC-2023-E and COBROS OA-2022-07) stands in for the regional regulations of the eventual real use case.

## Language

### Parties

**Applicant**:
The person or business that files a Permit Request (owner, lessee, or their authorized representative).
_Avoid_: Legislator, solicitante (in English text), client

**Reviewer**:
The case-management staff member who reads the Recommendation and records the Determination.
_Avoid_: Approver, officer, user

### The case

**Permit Request**:
A single application for a business permit, made of a fixed set of fields that depend on its Request Type.
_Avoid_: Application, permit, case, solicitud (in English text)

**Request Type**:
The kind of permit being requested, which determines the required fields and applicable rules. Currently Permiso de Construcción (PCOC) and Permiso Único.
_Avoid_: Permit type, category

### The regulation

**Reglamento**:
The body of regulatory documents the assistant reviews against; today, the Reglamento Conjunto (RC-2023-E) plus the fee order (COBROS OA-2022-07).
_Avoid_: Documentation, docs, reglamentación, regulations

**Section**:
A citable unit of the Reglamento, identified by its official number (e.g. "Sección 3.7.1.2" or "Anejo A, 6").
_Avoid_: Chunk (that is a storage concept), article

**Clause**:
A lettered paragraph within a Section, the smallest citable unit that states a single requirement (e.g. clause (k) of Sección 2.1.9.2, cited as `RC-2023-E:2.1.9.2(k)`). The Reglamento calls it an *inciso*; Spanish content keeps that word.
_Avoid_: inciso (in English text), subsection, paragraph

### The outcome

**Recommendation**:
The assistant's output for a Permit Request: an Approve / Not approve verdict, an Approval Score, and Notes. Advisory only.
_Avoid_: Decision, result, AI approval

**Approval Score**:
The probability (0–1) that the Permit Request should be approved: the probability of its weakest requirement, since approval needs every requirement met. A score near the cutoff is **Borderline** and flagged for closer review.
_Avoid_: Confidence, rating, grade

**Criterion**:
A single regulatory requirement for a Request Type, curated in advance and tied to the Clauses (or whole Sections) that establish it. Each Criterion is one Requirement. A Criterion may apply only to new requests or only to renewals.
_Avoid_: Rule, check, condition

**Requirement**:
Something a Permit Request must meet, assessed on its own with a probability: either a Criterion, or the remaining applicable Clauses of a Section that applies to the request. The weakest Requirement sets the Approval Score.
_Avoid_: check, requirement check, condition

**Note**:
A one-to-two-sentence statement supporting the Recommendation, citing at least one Section. A Note may be **curable**: it describes a deficiency the Applicant can fix rather than a reason to deny.
_Avoid_: Comment, reason, explanation, finding

**Determination**:
The Reviewer's final action on a Permit Request: accept the Recommendation, override it (with a reason), or issue a Cure Request.
_Avoid_: Decision, verdict, ruling

**Cure Request**:
A Determination asking the Applicant to fix listed deficiencies within the Reglamento's deadline instead of approving or denying (requerimiento de subsanación).
_Avoid_: Rejection, return, correction request

**Fee Check**:
A deterministic verification that the fee declared on a Permit Request matches the COBROS fee schedule for its Request Type and Applicant.
_Avoid_: Payment validation
