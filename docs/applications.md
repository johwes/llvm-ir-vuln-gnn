# Applications and Market Position

This document covers where an IR-level GNN vulnerability detector fits relative to
existing tools, which scenarios it addresses that nothing else does, and where it
does not compete.

---

## What existing tools cover well

**Traditional SAST** (Coverity, Fortify, CodeQL, Semgrep) works at the source level
with identifier names intact. These tools are rule-based or pattern-matched: they look
for `strcpy` near `user_input`, `malloc` without a null check, `strlen` used as an
allocation size. They're effective at known patterns in code where full source access
is available. CodeQL can express multi-file dataflow queries across call chains and has
rule libraries built up over years.

**ML-based SAST** (CodeBERT-based systems, VulDeePecker, ReVeal) also requires source
code. These tools reach 63–65% on the Devign benchmark precisely because they read
token sequences — `user_input`, `buffer`, `n` — and correlate them with vulnerability
patterns learned from billions of lines of code and CVE descriptions.

**Fuzzing** (AFL++, libFuzzer, OSS-Fuzz) finds bugs dynamically but requires the code
to run, harnesses to be written, and hours to days of compute per target.

**Formal verification** (IKOS, KLEE, CBMC) is sound but requires specifications and
scales poorly to large codebases without significant engineering investment.

---

## The specific gap: IR-level, identifier-independent analysis

The gap is **architecture-neutral, language-agnostic, identifier-independent vulnerability
screening that operates natively at the LLVM IR level and runs in under a second per
function**. No existing commercial tool occupies this position.

This matters in three scenarios, all of which are growing:

### 1. Supply chain security

Post-Log4Shell and SolarWinds, organisations face pressure to scan third-party
dependencies at build time. Third-party artifacts often arrive as compiled IR or
bitcode without original source identifiers. Binary analysis tools exist (IDA Pro,
Binary Ninja, Ghidra) but operate far below IR in information density — IR retains
SSA form, explicit types, and call structure that binaries lose.

An IR-level GNN provides structured vulnerability screening at the richest
representation available without source access. Nothing in the current tooling
ecosystem fills this role.

### 2. Language-agnostic pipelines

Any language with an LLVM frontend — C, C++, Rust, Go, Swift, Zig, Kotlin Native —
compiles to the same IR format. A single IR-level model screens all of them without
retraining per language and without language-specific rule sets. As Rust adoption grows
in security-critical code (Linux kernel drivers, Android system components), an IR-level
tool picks up Rust vulnerabilities with zero additional engineering. CodeQL requires
separate queries per language; Semgrep patterns are language-specific.

### 3. Zero-cost CI/CD triage before expensive tools

A GNN scoring pass costs under a second per function. CodeQL on a large codebase takes
10–60 minutes. Formal verification (IKOS) requires hours per function. Fuzzing requires
days. An IR-level pre-screener that narrows 500 functions to 20 high-suspicion candidates
before any expensive tool runs has real economic value — not as a replacement but as a
first-pass filter that reduces the cost of the expensive tier.

This is the direct integration point for SCAR: bitcode is already produced by the
`build-bitcode` Tekton task, the GNN pass adds no dependencies and near-zero compute,
and the shortlist feeds directly into the LLM triage and IKOS formal analysis stages.

---

## Where it does not compete

When full source access and readable identifiers are available, CodeQL and Semgrep are
more accurate, more expressive (multi-file dataflow, cross-function queries), and have
far larger rule libraries. The 5–6 percentage point accuracy gap to CodeBERT (63% vs.
58% on Devign) is real and is explained by IR discarding the identifier vocabulary that
those tools rely on.

The GNN also does not produce findings — it produces scores. Traditional SAST gives a
specific trace: `user_input flows to memcpy at line 47 without bounds check`. That is
directly actionable for a developer. A suspicion score is actionable only for a
downstream system (an LLM, a fuzzer harness generator) that reconstructs the trace
itself.

---

## Honest market position

The IR-level GNN is a complement to LLVM/IKOS-based pipelines, not a replacement for
any existing tool. Its value is specifically where IR is already being produced, the
GNN pass has zero marginal cost, and a ranked shortlist is more useful than an
unranked set.

The addressable market today is narrow: organisations running LLVM-based CI/CD
pipelines that want sub-second vulnerability triage on compiled IR before invoking
expensive static analysis or fuzzing.

The market that could make this relevant at scale is **software supply chain security**:
scanning third-party components at build time without source access, where the
alternative is either no analysis or prohibitively expensive binary analysis. SBOM
requirements (NTIA, Executive Order 14028) are already driving inventory at scale; the
next regulatory pressure will be toward functional analysis of SBOM components, not
just their enumeration.

---

## The long-term case

The combination that does not exist anywhere today:

1. **SCAR at scale** generating a large instruction-level labelled dataset — planted
   bugs with exact IR instruction locations, IKOS-validated true positives, controllable
   CWE balance, minimal-diff CVE/fix pairs.

2. **A GNN trained with node-level supervision** on that dataset: instead of "is this
   function vulnerable?" (graph classification), the model answers "is this instruction
   an unguarded sink?" (node classification). The PDG backward-slice architecture and
   the §23 sink-node readout are already structured correctly for this objective — they
   require instruction-level training labels to realise their potential.

3. **Integration into supply chain scanning pipelines** as a free first pass over
   compiled IR, producing a ranked list of high-suspicion functions and their specific
   suspicious instructions, before any source-dependent tool is invoked.

That pipeline — SCAR-generated training data → node-level GNN → IR-native CI/CD
integration — would be the first toolchain capable of structured, language-agnostic,
identifier-independent vulnerability triage at IR scale. The individual components exist;
the combination does not.
