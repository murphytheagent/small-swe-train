You: 

read these three papers first: - https://arxiv.org/abs/2601.18734 - https://arxiv.org/abs/2601.20802 - https://arxiv.org/abs/2601.19897 summarize each approach covering their methods and main math. Then brainstorm can these algorithms be used to post train a small LM, such as Qwen3-4B, to become a coding agent or research agent

ChatGPT: 

Reasoned for 28m 23s

You: 

Below are the three papers’ core ideas, *what is being optimized*, and the key math, plus a concrete brainstorm for using these algorithms to post-train a small LM (e.g., **Qwen3-4B**) into a **coding agent** or **research agent**.

All three papers share a unifying pattern:

> **Teacher = the same model, but conditioned on extra “privileged” context** (ground-truth solution, environment feedback, or demonstrations).  
> **Student = the model without that privileged context.**  
> **Train on the student’s own rollouts (“on-policy”) by minimizing a divergence between teacher and student next-token distributions.**

---

## 1) Self-Distilled Reasoner: On-Policy Self-Distillation (OPSD) — arXiv:2601.18734

### What problem it targets

Reasoning post-training often uses either:

-   **SFT** on expert traces (off-policy → exposure bias / train–test mismatch), or
    
-   **RLVR** (e.g., GRPO) with **sparse** rewards and expensive sampling.
    

This paper’s idea: if you already have **ground-truth solutions** (possibly with CoT), use them as *privileged information* to create a “wiser” teacher *from the same model* and do **dense token-level supervision** on student rollouts. [arXiv+1](https://www.arxiv.org/pdf/2601.18734)

### Teacher/student construction

Given dataset $\mathcal{S}=\{(x_i, y_i^\star)\}$:

-   Student policy: $p_S(\cdot \mid x) := p_\theta(\cdot \mid x)$
    
-   Teacher policy: $p_T(\cdot \mid x, y^\star) := p_\theta(\cdot \mid x, y^\star)$  
    (same weights; different conditioning/prompt) [arXiv](https://www.arxiv.org/pdf/2601.18734)
    

Rollout is **on-policy**: sample $\hat y \sim p_S(\cdot \mid x)$. At each position $n$, compare teacher vs student next-token distributions conditioned on the same student prefix $\hat y_{<n}$. [arXiv](https://www.arxiv.org/pdf/2601.18734)

### Core objective (main math)

They write the OPSD loss as expectation over data and **student rollouts**, summing a per-token divergence between next-token distributions:

$$
\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{(x,y^\star)\sim \mathcal{S}} \ \mathbb{E}_{\hat y \sim p_S(\cdot\mid x)} \ \sum_{n=1}^{|\hat y|} D\!\left( p_T(\cdot\mid x,y^\star,\hat y_{<n}) \ \|\ p_S(\cdot\mid x,\hat y_{<n}) \right).
$$

[arXiv+1](https://www.arxiv.org/pdf/2601.18734)

A key instantiation is **full-vocabulary logit distillation** with a divergence $D$. They highlight generalized Jensen–Shannon divergence:

$$
\mathrm{JSD}_\beta(p_T\|p_S)= \beta\,\mathrm{KL}(p_T\|m) + (1-\beta)\,\mathrm{KL}(p_S\|m), \quad m=\beta p_T + (1-\beta)p_S.
$$

[arXiv](https://www.arxiv.org/pdf/2601.18734)

They also discuss a **sampled-token / policy-gradient style** variant, defining a token-level “advantage-like” term:

$$
A_n(x,\hat y)=\log p_T(\hat y_n \mid x,y^\star,\hat y_{<n}) - \log p_S(\hat y_n \mid x,\hat y_{<n}).
$$

[arXiv](https://www.arxiv.org/pdf/2601.18734)

### Practical stabilization choice

Important detail: they **fix the teacher policy to the initial policy** (rather than continually updating the teacher to match the student), as an implicit regularizer for stability. [arXiv](https://www.arxiv.org/pdf/2601.18734)

### Why it matters for small models (Qwen3-4B)

They explicitly run OPSD on **Qwen3-4B** and report improvements vs both SFT and GRPO on math reasoning benchmarks, plus large **token-efficiency** gains (OPSD can be *4–8× more token-efficient than GRPO* in their experiment, using shorter generations and fewer rollouts). [arXiv](https://www.arxiv.org/pdf/2601.18734)

---

## 2) Reinforcement Learning via Self-Distillation: SDPO — arXiv:2601.20802

### What problem it targets

RL with verifiable rewards (**RLVR**) has a **credit assignment bottleneck**: often you only get a scalar outcome reward (correct/incorrect), even though in many real settings you can get *textual feedback* that explains failures (runtime errors, judge critiques, tool outputs). [arXiv](https://arxiv.org/pdf/2601.20802)

They formalize this as **Reinforcement Learning with Rich Feedback (RLRF)**:

-   RLVR: scalar reward $r$ is an information bottleneck.
    
-   RLRF: environment provides tokenized feedback $f$, which can encode state observations and “why” something failed. [arXiv](https://arxiv.org/pdf/2601.20802)
    

### Key idea

Use the *same model* as a **self-teacher** by conditioning it on the feedback $f$. Then distill that feedback-informed distribution back into the student policy (without feedback). This turns feedback into **dense logit-level learning signal**. [arXiv+1](https://arxiv.org/pdf/2601.20802)

### Core objective (main math)

Let student be $\pi_\theta(\cdot\mid x)$ and self-teacher be $\pi_\theta(\cdot\mid x,f)$ (same params; different conditioning). Their SDPO distillation loss is:

$$
\mathcal{L}_{\text{SDPO}}(\theta) := \sum_t \mathrm{KL}\!\Big( \pi_\theta(\cdot \mid x, y_{<t}) \ \|\ \mathrm{stopgrad}(\pi_\theta(\cdot \mid x, f, y_{<t})) \Big).
$$

[arXiv](https://arxiv.org/pdf/2601.20802)

The **stopgrad** is crucial: gradients only flow through the student side, preventing the teacher from collapsing into the student and ignoring feedback. [arXiv](https://arxiv.org/pdf/2601.20802)

They also provide an explicit gradient form (Prop 2.1), showing the loss behaves like a policy-gradient style update with dense “advantages” derived from log-ratios of student vs teacher probabilities. [arXiv+1](https://arxiv.org/pdf/2601.20802)

### “RL interpretation”: maximum entropy RL with an implicit reward

In Appendix C.1 they connect SDPO to **maximum entropy RL**:

MaxEnt RL objective:

$$
\max_\theta \ \mathbb{E}_{y\sim\pi_\theta}\left[\sum_t r(y_t\mid x,y_{<t})\right] + \lambda H[\pi_\theta(\cdot\mid x)].
$$

They show it is equivalent to minimizing a KL to an optimal “posterior” policy $\pi^\star$, and then note SDPO matches that form if you set an **implicit reward**

$$
r(y_t\mid x,y_{<t}) = \log q(y_t\mid x,f,y_{<t}),\quad \lambda=1,
$$

i.e., the self-teacher’s log-prob becomes the dense reward. [arXiv+1](https://arxiv.org/pdf/2601.20802)

### Stabilizing the teacher: trust region or EMA

They discuss that the self-teacher is not necessarily frozen; it can improve during training, but can also destabilize.

Two main stabilizers:

1.  **Trust-region teacher**: enforce  
    $\sum_t \mathrm{KL}(q(\cdot)\|q_{\theta_{\text{ref}}}(\cdot))\le \epsilon$, and the closed-form solution is a log-space mixture:
    

$$
q^\star \propto \exp((1-\alpha)\log q_{\theta_{\text{ref}}} + \alpha \log q_\theta).
$$

[arXiv](https://arxiv.org/pdf/2601.20802)

2.  **EMA teacher**: parameters $\theta'\leftarrow (1-\alpha)\theta' + \alpha\theta$, and under smoothness assumptions they derive an approximate trust-region bound. [arXiv](https://arxiv.org/pdf/2601.20802)
    

They also propose **top-K distillation** (approximate logit distillation) to reduce memory/compute. [arXiv](https://arxiv.org/pdf/2601.20802)

### Why this is “agent-relevant”

Because $f$ can be *any tokenized state/feedback*, SDPO is naturally suited to:

-   coding environments (compiler/runtime errors, failing tests),
    
-   tool-use environments (API error messages),
    
-   judge feedback (textual critiques).
    

This is exactly the kind of signal agents produce and consume. [arXiv+1](https://arxiv.org/pdf/2601.20802)

---

## 3) Self-Distillation Fine-Tuning (SDFT) for Continual Learning — arXiv:2601.19897

### What problem it targets

Continual learning: add new skills/knowledge without catastrophic forgetting.

They argue:

-   **On-policy** learning reduces forgetting,
    
-   but explicit rewards may not exist,
    
-   and SFT on demonstrations is **off-policy** and forgets.
    

So they propose **on-policy self-distillation from demonstrations**. [arXiv+1](https://arxiv.org/pdf/2601.19897)

### Teacher/student construction

Given a base model policy $\pi$:

-   Teacher: $\pi(\cdot\mid x,c)$ where $c$ is an **expert demonstration** (chosen per prompt $x$).
    
-   Student: $\pi_\theta(\cdot\mid x)$ without the demonstration.
    

They use a prompt template to present the demo as an example and then ask for a new response “of your own,” to avoid verbatim copying. [arXiv+1](https://arxiv.org/pdf/2601.19897)

### Core objective (main math): reverse KL on student rollouts

They sample **student** responses $y\sim \pi_\theta(\cdot\mid x)$ and minimize **reverse KL**:

$$
\mathcal{L}(\theta) = D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\ \|\ \pi(\cdot\mid x,c)\right) = \mathbb{E}_{y\sim \pi_\theta(y\mid x)}\left[\log\frac{\pi_\theta(y\mid x)}{\pi(y\mid x,c)}\right].
$$

[arXiv](https://arxiv.org/pdf/2601.19897)

They then derive a token-level gradient estimator:

$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{y\sim\pi_\theta}\left[ \sum_t \sum_{y_t\in V} \log\frac{\pi_\theta(y_t\mid y_{<t},x)}{\pi(y_t\mid y_{<t},x,c)} \ \nabla_\theta \log \pi_\theta(y_t\mid y_{<t},x) \right].
$$

[arXiv](https://arxiv.org/pdf/2601.19897)

Teacher weights: they often use an **EMA of student parameters** for the teacher (stabilization / smoother target). [arXiv](https://arxiv.org/pdf/2601.19897)

### IRL-style interpretation: demonstrations induce an “intrinsic reward”

They start from a trust-region RL objective (TRPO-like):

$$
\pi_{k+1}=\arg\max_\pi\ \mathbb{E}_{y\sim\pi}[r(y,x)]-\beta D_{\mathrm{KL}}(\pi(\cdot\mid x)\|\pi_k(\cdot\mid x)).
$$

Optimal policy has “tilted” form:

$$
\pi^\star_{k+1}(y\mid x)\propto \pi_k(y\mid x)\exp\left(\frac{1}{\beta}r(y,x)\right),
$$

so rearranging gives reward as a log-prob gap:

$$
r(y,x)=\beta\big(\log \pi^\star_{k+1}(y\mid x)-\log\pi_k(y\mid x)\big)+C.
$$

[arXiv](https://arxiv.org/pdf/2601.19897)

Their **in-context assumption** is:

$$
\pi^\star_{k+1}(y\mid x)\approx \pi(y\mid x,c),
$$

so the implicit reward becomes:

$$
r(y,x,c)=\log\pi(y\mid x,c)-\log\pi_k(y\mid x),
$$

and token-level:

$$
r_t=\log \frac{\pi(y_t\mid y_{<t},x,c)}{\pi_k(y_t\mid y_{<t},x)}.
$$

They then show optimizing this reward yields gradients equivalent (in expectation) to the reverse-KL distillation gradient. [arXiv+1](https://arxiv.org/pdf/2601.19897)

This is the conceptual bridge: **self-distillation from a demo-conditioned policy ≈ on-policy RL with an intrinsic reward induced by demos.**

---

# Can these be used to post-train Qwen3-4B into a coding agent or research agent?

Yes—*and the three methods are complementary*, because they assume different kinds of “privileged context”:

-   **OPSD**: you have **ground-truth solutions** $y^\star$ (or verified reasoning traces). Great for **offline post-training** on curated datasets. [arXiv+1](https://www.arxiv.org/pdf/2601.18734)
    
-   **SDPO**: you have **rich textual feedback** $f$ from an environment (errors, judge critiques, tool outputs). Great for **interactive agent training** and even **test-time adaptation**. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    
-   **SDFT**: you have **demonstrations** $c$ and want **continual learning without forgetting**. Great for adding new tools/domains over time. [arXiv+1](https://arxiv.org/pdf/2601.19897)
    

Also importantly: OPSD explicitly reports improvements on **Qwen3-4B** and notes benefits grow with model scale (self-distillation works better when the model can leverage privileged context). [arXiv](https://www.arxiv.org/pdf/2601.18734)  
SDPO similarly expects stronger in-context learners to benefit more from self-teaching. [arXiv+1](https://arxiv.org/pdf/2601.20802)

Below are concrete ways to use them for **coding** and **research** agent post-training.

---

## A practical recipe for a coding agent (Qwen3-4B)

### 1) Use SDPO as the main “agentic” post-training loop (most direct fit)

Coding naturally yields **rich feedback**:

-   compiler errors
    
-   runtime exceptions / stack traces
    
-   failing unit tests with diffs
    
-   static analysis warnings
    
-   LLM-judge critiques (“fails edge case X”)
    

That’s exactly SDPO’s RLRF setting. [arXiv+1](https://arxiv.org/pdf/2601.20802)

**Training sketch:**

1.  Sample a coding problem $x$.
    
2.  Student proposes a solution $y\sim \pi_\theta(\cdot\mid x)$.
    
3.  Execute it; produce feedback tokens $f$ (errors/tests output).
    
4.  Build self-teacher distribution via the same model conditioned on $(x,f)$.
    
5.  Update $\theta$ by minimizing  
    $\sum_t \mathrm{KL}(\pi_\theta(\cdot\mid x,y_{<t}) \ \| \ \text{stopgrad}(\pi_\theta(\cdot\mid x,f,y_{<t})))$. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    

**Stability / compute tips borrowed from the paper:**

-   Use **trust-region** or **EMA teacher** to avoid teacher drift. [arXiv](https://arxiv.org/pdf/2601.20802)
    
-   Use **top‑K distillation** rather than full softmax to reduce memory. [arXiv](https://arxiv.org/pdf/2601.20802)
    

### 2) Use OPSD when you have *verified reference solutions* (or can generate them)

For many coding datasets, you either have:

-   official reference solutions, or
    
-   a trusted solution found by search / best-of-n sampling, or
    
-   solutions verified by private tests.
    

OPSD’s “privileged info” $y^\star$ can be that reference solution. Then teacher is the same model conditioned on $y^\star$, and you distill onto the student’s own rollouts with dense token-level supervision. [arXiv+1](https://www.arxiv.org/pdf/2601.18734)

This is particularly attractive if you’re post-training on coding tasks with high-quality gold solutions (competitive programming, some LeetCode-like corpora).

### 3) Use SDFT for continual tool/API expansion without forgetting

A coding agent often needs to continually pick up:

-   new libraries
    
-   new tool APIs (linters, test runners, build systems)
    
-   new repo conventions
    

SDFT gives a simple mechanism: provide a **demonstration** $c$ (e.g., a solved example showing tool calls), use the demo-conditioned model as teacher, and do **on-policy** reverse-KL distillation on the student’s rollouts. [arXiv+1](https://arxiv.org/pdf/2601.19897)

This is a plausible approach to “keep the agent fresh” while reducing catastrophic forgetting.

---

## A practical recipe for a research agent

A research agent is harder because “correctness” is less easily verifiable. The key is to **design an environment that produces rich feedback** $f$ (or privileged context) about factuality, citations, and reasoning quality.

### 1) SDPO with “rich feedback” from fact-checking / citation checking

You can create tokenized feedback like:

-   “Claim 3 is not supported by cited sources”
    
-   “Citation \[2\] does not contain the quoted statistic”
    
-   “Contradiction: source A says X, draft says Y”
    
-   judge feedback on missing caveats or incorrect inference
    

Then SDPO can distill a feedback-conditioned self-teacher into the base policy, giving dense credit assignment about *where* the model went wrong in the generation. [arXiv+2arXiv+2](https://arxiv.org/pdf/2601.20802)

### 2) OPSD as “privileged retrieval distillation”

If you can provide privileged context to the teacher—e.g., the *correct set of retrieved passages* or a curated evidence bundle—OPSD becomes a kind of **RAG compression / evidence distillation**:

-   Student: answer from question alone (or with limited retrieval)
    
-   Teacher: same model conditioned on gold evidence (or gold answer), distill teacher next-token distributions along student rollouts. [arXiv+1](https://www.arxiv.org/pdf/2601.18734)
    

This can teach better citation habits and evidence-grounded phrasing.

### 3) SDFT for continual learning of new domains and research workflows

Research agent skills are highly “procedural”:

-   how to structure a literature review
    
-   how to compare papers
    
-   how to write with calibrated uncertainty
    
-   how to cite sources
    

SDFT naturally fits: demonstrations $c$ can be *exemplar writeups or tool-use traces* for specific domains, and reverse-KL on-policy distillation avoids the pure off-policy SFT failure modes. [arXiv+1](https://arxiv.org/pdf/2601.19897)

---

## How I’d combine them for Qwen3-4B (high-level training plan)

If the goal is “Qwen3-4B → agent” (coding or research), a reasonable combined plan is:

1.  **SDFT phase (skill bootstrapping from demonstrations)**  
    Teach the *workflow* (tool calling, planning templates, citation style) with demo-conditioned teacher and on-policy reverse-KL. [arXiv+1](https://arxiv.org/pdf/2601.19897)
    
2.  **SDPO phase (interactive environment refinement)**  
    Put the model in the real environment loop (execute code, run tests, web/retrieval + validators) and use rich feedback $f$ to do dense credit assignment via self-distillation. Add EMA/trust-region teacher for stability; use top‑K distillation to fit compute. [arXiv+2arXiv+2](https://arxiv.org/pdf/2601.20802)
    
3.  **OPSD phase (wherever gold solutions exist)**  
    For tasks with verified solutions or trusted traces, use OPSD’s privileged $y^\star$ to distill a teacher conditioned on the solution back into student behavior on student rollouts. OPSD already demonstrates strong results at **Qwen3-4B** scale and good token efficiency. [arXiv+1](https://www.arxiv.org/pdf/2601.18734)
    

---

## Main caveats (what can go wrong / what you must design carefully)

-   **These methods assume the model “gets wiser” with context.**  
    If Qwen3-4B cannot reliably correct itself given feedback/demos/solutions, self-distillation can plateau or even reinforce errors. Both OPSD and SDPO discuss sensitivity to model strength / scaling. [arXiv+2arXiv+2](https://www.arxiv.org/pdf/2601.18734)
    
-   **Feedback quality becomes your reward model.**  
    For research-agent training, the hardest part is generating *high-signal, non-gameable feedback* $f$. SDPO’s mechanism is powerful, but it will learn whatever the feedback implicitly rewards. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    
-   **Teacher drift / collapse is real.**  
    SDPO explicitly introduces trust-region or EMA teacher regularization; OPSD fixes teacher to the initial policy for stability. You’d likely need something similar for long-horizon agent training. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    

---

If you want, I can go one level deeper and propose:

-   an explicit **MDP / trajectory** formulation for tool-using agents under these losses,
    
-   a concrete **data + environment** spec for “coding agent SDPO” (what feedback to tokenize, how to structure teacher prompts),
    
-   and a minimal **training loop pseudocode** that mixes SDFT + SDPO + OPSD stably for a 4B model.

ChatGPT: 

SDPO for SWE small size agent training sounds promising, it would also be crucial to formulate a way of measuring the progression of teacher's ICL capability, do a research on current landscape to see if this has been done, picture training a local SWE agent, start with first looking into what they used SDPO for in the paper first

You: 

Reasoned for 30m 38s

ChatGPT: 

Below is (1) what **SDPO is actually used for** in the SDPO paper (arXiv:2601.20802), especially the “SWE-like” parts, then (2) what the paper already does to *quantify* the self-teacher / ICL angle, and (3) a short research survey on **how people measure “uses the feedback/context”** (i.e., ICL/context-usage) + a concrete measurement suite you can adopt when training a **local SWE agent (e.g., Qwen3-4B)**.

---

## 1) What SDPO is used for in the paper

The SDPO paper positions SDPO as **reinforcement learning with rich feedback (RLRF)** where the environment returns *textual feedback* (errors, test failures, etc.), and SDPO turns that into a **dense per-token learning signal** by distilling a “self-teacher” that *conditions on the feedback*. [arXiv+1](https://arxiv.org/html/2601.20802v1)

### A. Learning *without* rich environment feedback (RLVR-style): science Q&A + tool-use

Here the environment reward is basically scalar/binary (correct/incorrect), i.e., “RLVR.” SDPO still applies by **treating successful rollouts in the same batch as “feedback” for failed rollouts** on the same question. The teacher sees the (failed) attempt + a sample correct solution and can retrospectively spot where it went wrong. [arXiv](https://arxiv.org/pdf/2601.20802)

**Tasks & setup (Section 3.1):**

-   **Science Q&A**: undergraduate-level scientific reasoning using reasoning subsets (L3) from SciKnowEval in Chemistry/Physics/Biology/Materials. [arXiv](https://arxiv.org/pdf/2601.20802)
    
-   **Tool use**: ToolAlpaca, mapping an API spec + user request → correct tool call. [arXiv](https://arxiv.org/pdf/2601.20802)
    
-   Models: **Qwen3-8B** and **Olmo3-7B-Instruct** initial checkpoints. [arXiv](https://arxiv.org/pdf/2601.20802)
    

**Headline result pattern:** SDPO (on-policy) improves accuracy fast and can beat or match strong GRPO variants under the same wall-clock budget. For example, Table 3 reports best avg@16 within 1h/5h across domains and shows SDPO competitive or stronger in multiple columns for both Qwen3-8B and Olmo3-7B-Instruct. [arXiv+1](https://arxiv.org/pdf/2601.20802)

**Why it matters for SWE:** even if you don’t have rich feedback early (e.g., only “tests passed?”), SDPO can still use *successful attempts* as “feedback” and densify credit assignment.

---

### B. Learning *with* rich feedback: competitive programming with LeetCode-style outputs (closest analog to SWE tests)

This is the most “SWE agent training” relevant part.

**Benchmark & environment:**

-   Competitive programming on **LiveCodeBench v6** (LCBv6) with **LeetCode-style feedback** after submission. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    
-   They use a setup with **public tests for training feedback** and **private tests for validation**, explicitly noting this mirrors the classic public/private split in programming contests. [arXiv](https://arxiv.org/pdf/2601.20802)
    
-   The environment output includes error types (e.g., runtime error) and other submission feedback. [arXiv](https://arxiv.org/pdf/2601.20802)
    

**What SDPO is doing here:**

-   Student produces a solution attempt.
    
-   Environment returns **text feedback** (compile/runtime/wrong answer details).
    
-   The *same model*, now conditioned on (problem + attempt + feedback), acts as a **self-teacher** and produces a distribution over “what should have been said next.”
    
-   SDPO distills those **feedback-informed next-token probabilities** back into the student policy, giving dense signal even if no successful solution exists yet. [arXiv+1](https://arxiv.org/html/2601.20802v1)
    

**Results (Table 5):**  
For **Qwen3-8B**, SDPO improves LCBv6 accuracy from 35.0 to **48.8**, vs GRPO to 41.2 (in the paper’s reported setup). [arXiv](https://arxiv.org/pdf/2601.20802)  
They also report results on smaller Qwen3 variants (including **Qwen3-4B**). [arXiv](https://arxiv.org/pdf/2601.20802)

**Key analysis you should internalize for small SWE agents:**

-   They explicitly argue SDPO gains scale with model size because **larger models are better at in-context “retrospection.”** [arXiv+1](https://arxiv.org/pdf/2601.20802)
    
-   They note very small models may underperform unless combined with other signals (they analyze combining SDPO+GRPO). [arXiv](https://arxiv.org/pdf/2601.20802)
    

This is basically “unit tests as feedback,” i.e., the exact thing you’d want for SWE-bench-style patching.

---

### C. Test-time self-distillation: “solve one hard problem faster” by compressing feedback into weights

They introduce **Test-Time Self-Distillation**: instead of appending longer and longer feedback into context, repeatedly update weights with SDPO to “compress context into parameters.” [arXiv+1](https://arxiv.org/pdf/2601.20802)

They define:

-   **Discovery time** = #attempts until first success. [arXiv](https://arxiv.org/pdf/2601.20802)
    
-   **discovery@k** = probability a solution is found within k attempts. [arXiv](https://arxiv.org/pdf/2601.20802)
    

They show on subsets of hard LCBv6 questions (e.g., very hard: pass@64 < 0.03) SDPO yields substantially better discovery@k than best-of-k and multi-turn prompting. [arXiv+1](https://arxiv.org/pdf/2601.20802)

**Why it matters for SWE:** if your local agent is trying to solve a *single real issue* and you can afford some “in-issue learning,” SDPO’s test-time loop is directly suggestive.

---

## 2) What the SDPO paper already uses to “measure teacher ICL / retrospection” (and why it’s not quite enough)

You mentioned: *“crucial to formulate a way of measuring the progression of teacher’s ICL capability.”*  
Good news: the authors already include a few probes that are **very close** to that, but they’re not packaged as a general measurement suite.

### A. “Self-teacher improves during training”

In Section 4.3/4.4 discussion, they track **generative accuracy of the self-teacher vs student** during training and explicitly remark that **the student can surpass the initial teacher**, and that **EMA/trust-region teachers outperform a frozen teacher**, implying the teacher improves due to parameter sharing. [arXiv+1](https://arxiv.org/pdf/2601.20802)

This is essentially a “teacher progression” measurement, but it’s *task-score based*, not a direct “ICL capability” metric.

### B. “Which feedback is informative?” + teacher diversity / copying

Table 6 is extremely relevant to your question because it measures both:

-   teacher accuracy **before training** under different feedback types
    
-   student accuracy **after SDPO training**
    
-   **Same output (%)** = how often the teacher produces the *same environment output* as the student’s attempt (a proxy for “teacher is just repeating the same mistake / not exploring”)
    
-   teacher entropy (proxy for exploration / uncertainty) [arXiv+1](https://arxiv.org/pdf/2601.20802)
    

They find environment output and sample solutions are complementary, and that including the student’s original attempt can bias the teacher and reduce exploration. [arXiv+1](https://arxiv.org/pdf/2601.20802)

This is already *very close* to an ICL/retrospection measurement because the teacher is literally a feedback-conditioned policy.

---

## 3) Research landscape: has “context/feedback usage” been measured before?

Yes—just not always framed as “teacher ICL progression.” There are at least three relevant measurement traditions you can borrow.

### A. Context usage as an information-theoretic quantity (CXMI-style)

In context-aware machine translation, Fernandes et al. propose **Conditional Cross-Mutual Information (CXMI)** to quantify *how much the model’s predictions change/improve when you provide context*, because many “context-aware” models ignore context. [arXiv+1](https://arxiv.org/abs/2105.03482)

Their framing is: compare model likelihood (or cross-entropy) **with context vs without context**, and interpret the difference as “information gain from context.” [ar5iv](https://ar5iv.org/pdf/2105.03482)

This is highly portable to SDPO teacher measurement: treat **feedback tokens** as “context,” and ask “how much do they change the teacher’s distribution / likelihood of a correct solution?”

### B. Random / corrupted feedback ablations (very relevant to SWE)

RLEF (execution-feedback RL for code) does exactly the kind of ablation you want:

-   They provide **true execution feedback vs random execution feedback** (feedback from an unrelated problem) and show that with random feedback, **self-repair is severely impaired**, which demonstrates the model is *actually using* the feedback rather than just sampling diversity. [ar5iv+1](https://ar5iv.org/pdf/2410.02089)
    
-   They also use pass@1 vs pass@10 and analyze error recovery categories and edit distances (chrF) across successive attempts. [ar5iv+1](https://ar5iv.org/pdf/2410.02089)
    

This is an excellent template for SDPO teacher measurement: if you can swap in random/irrelevant test failures and performance barely changes, your “teacher ICL” is fake.

### C. Tracking ICL emergence across training checkpoints (pretraining dynamics)

There’s also a line of work that measures ICL ability (and sub-abilities like task recognition vs task learning) *over checkpoints*, explicitly treating ICL as something that evolves during training and can fluctuate. [arXiv](https://arxiv.org/html/2406.14022v1)  
This matters for your “progression” framing: it’s normal to evaluate ICL *as a curve* over training, not as a single endpoint metric.

---

## 4) A concrete measurement suite for “teacher ICL progression” in SDPO-style SWE training

Let me define the objects in SDPO language:

-   Student policy: $\pi_\theta(y_t \mid x, y_{<t})$
    
-   Teacher distribution: $q_{\theta'}(y_t \mid x, f, y_{<t})$ (same model family, but conditioned on feedback $f$; in practice $\theta'$ may be EMA/trust-region regularized) [arXiv+1](https://arxiv.org/pdf/2601.20802)
    

Your goal: measure how the teacher’s **ability to leverage feedback** changes over training steps $k$.

### Metric 1: Feedback gain curve (behavioral)

On a fixed eval set of issues $x$:

1.  sample a student attempt $y \sim \pi_\theta(\cdot\mid x)$
    
2.  run tests → get feedback $f$
    
3.  sample a teacher “repair” $y^{T} \sim q_{\theta'}(\cdot\mid x,f)$ (or a teacher patch)
    
4.  evaluate both (tests pass?) → reward $r(\cdot)$
    

Define:

$$
G_{\text{fb}}(k) = \mathbb{E}[r(y^T) - r(y)]
$$

Track $G_{\text{fb}}(k)$ over training checkpoints. If teacher ICL is improving, **the delta should grow**.

This is essentially what SDPO hints at with teacher-vs-student generative accuracy, but you’d make it explicit and run it on a stable held-out set. [arXiv](https://arxiv.org/pdf/2601.20802)

### Metric 2: Random-feedback impairment (RLEF-style)

Keep the same protocol, but replace the feedback with a random one $\tilde f$ (same format, different issue).

$$
\Delta_{\text{rand}}(k)= \mathbb{E}[r(y^T\mid f)] - \mathbb{E}[r(y^T\mid \tilde f)]
$$

-   If $\Delta_{\text{rand}}\approx 0$, your teacher isn’t using feedback.
    
-   If $\Delta_{\text{rand}}$ increases with training, your teacher is becoming a better in-context learner of feedback.
    

This mirrors RLEF’s “random execution feedback” ablation logic. [ar5iv+1](https://ar5iv.org/pdf/2410.02089)

### Metric 3: Context usage (CXMI-like, distributional)

If you have a reference “good patch” $y^\*$ (SWE-bench provides gold patches), you can directly measure “how much feedback helps the teacher assign higher likelihood to the good patch”:

$$
\text{CXMI}_f(k) = \mathbb{E}\left[\log p_{\theta'}(y^\*\mid x,f)-\log p_{\theta'}(y^\*\mid x)\right]
$$

This is directly analogous to CXMI in the MT literature (context vs no-context likelihood gap). [ar5iv+1](https://ar5iv.org/pdf/2105.03482)

Even if you *don’t* trust gold patches (or don’t have them), you can compute a **purely model-internal** context sensitivity score by comparing distributions:

$$
U_f(k) = \mathbb{E}\left[\mathrm{KL}\big(q_{\theta'}(\cdot\mid x,f,y_{<t}) \,\|\, q_{\theta'}(\cdot\mid x,\emptyset,y_{<t})\big)\right]
$$

This measures how much the teacher’s next-token distribution changes when you include feedback. It’s a very clean “ICL usage” probe.

### Metric 4: Copy-rate / exploration (SDPO Table 6 style)

Track the analog of SDPO’s **Same output (%)** and entropy:

-   “Same output” for SWE could be: does the teacher produce a patch that yields the **same failing tests** as the student attempt?
    
-   Or: does it reproduce the same exception/trace signature?
    

This helps catch failure modes where the teacher “looks at feedback” but doesn’t actually change strategy. SDPO uses this exact idea as a diversity proxy. [arXiv+1](https://arxiv.org/pdf/2601.20802)

### Metric 5: Targeted repair vs “just sampling diversity”

RLEF’s pass@1 vs pass@10 logic is useful: if feedback use mainly improves pass@10 but not pass@1, the model might just be generating more diverse candidates rather than performing targeted fixes. [ar5iv](https://ar5iv.org/pdf/2410.02089)  
For SWE, you can mirror this by:

-   pass@1 patch success
    
-   success@N (any of N sampled patches passes)
    

Then compare **true feedback vs random feedback**: targeted repair should show up primarily in pass@1.

---

## 5) “Picture training a local SWE agent” with SDPO on something like Qwen3-4B

### Why SWE-bench is the natural sandbox

SWE-bench is explicitly “given a codebase + GitHub issue, generate a patch that resolves it.” [GitHub+1](https://github.com/SWE-bench/SWE-bench)  
SWE-bench Lite is a 300-task subset for faster iteration. [SWE-bench+1](https://www.swebench.com/lite.html)

So the environment gives you exactly what SDPO wants:

-   Attempt → run tests → feedback (failed tests, traces, logs) → retry or distill.
    

### Training loop sketch (SDPO-flavored)

**Episode data** (per issue):

-   $x$: issue text + repo snapshot + relevant file context (retrieval)
    
-   student attempt $y$: patch / diff
    
-   environment output $f$: failing tests + trace + minimal reproduction
    
-   reward $r$: pass/fail on held-out evaluation tests (or full suite)
    

**SDPO update idea:**

-   Teacher prompt includes $x$ and $f$, and (optionally) an exemplar “solution” from another successful rollout if available.
    
-   SDPO distills teacher next-token distributions back onto student.
    

Two practical tricks from SDPO you likely want:

-   **Regularized teacher** (EMA or trust-region-style) for stability. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    
-   Be careful including the student attempt $y$ in the teacher context: SDPO finds including it can bias the teacher toward repeating and reduce exploration; environment output + sample solutions were complementary. [arXiv+1](https://arxiv.org/pdf/2601.20802)
    

### Prevent “test output copying” / leakage

Borrow RLEF’s safety/rigor pattern: use **public tests for feedback** and **private tests for evaluation**, so the model can’t just memorize expected outputs from feedback. [ar5iv+1](https://ar5iv.org/pdf/2410.02089)  
In SWE-bench-style setups, you can approximate this by:

-   selecting a subset of tests (or a smaller smoke suite) to show as feedback
    
-   keeping the full suite (or additional hidden tests) only for reward/eval
    

### Why Qwen3-4B is plausible, and what to watch

The SDPO paper explicitly evaluates Qwen3 sizes including **4B** on LCBv6, and generally emphasizes SDPO depends on **in-context “retrospection” quality** (better in larger models). [arXiv+1](https://arxiv.org/pdf/2601.20802)  
So Qwen3-4B is a reasonable “small-but-not-tiny” starting point, but you should expect:

-   If teacher ICL is weak early, SDPO might stall; consider:
    
    -   mixing in GRPO-style signals (they discuss combining SDPO and GRPO) [arXiv](https://arxiv.org/pdf/2601.20802)
        
    -   curriculum: start on easier/unit-test-localized issues before long-horizon ones.
        

### What “agentic” adds beyond patch generation

If you want a real SWE-agent (search files, run selective tests, etc.), the clean path is:

1.  First, SDPO-train **patch generation with feedback** (simpler action space).
    
2.  Then extend to tool-using agent trajectories (actions like `grep`, `pytest -k`, file open/edit) and treat tool outputs as part of feedback $f$.
    
3.  Keep the same teacher-ICL metrics above, but now feedback is richer (tool logs + test logs).
    

---

## 6) Practical recommendation: a minimal “teacher ICL progression dashboard” to build first

If you want something you can implement quickly and that will actually de-risk SDPO-on-SWE:

1.  **$G_{\text{fb}}(k)$**: teacher repair success − student attempt success (per checkpoint)
    
2.  **$\Delta_{\text{rand}}(k)$**: success(true feedback) − success(random feedback)
    
3.  **Copy-rate**: fraction of teacher repairs that reproduce the same failing tests / same trace signature
    
4.  **Entropy / diversity**: patch diff diversity across samples (or SDPO-style entropy proxy)
    
5.  Optional if you have gold patches: **CXMI-like likelihood gap** on gold patch with vs without feedback
    

Those 3–5 signals will tell you *very early* whether:

-   SDPO is actually teaching “use the feedback” vs “be more random”
    
-   the teacher’s in-context capability is improving
    
-   Qwen3-4B is above the “minimum viable retrospection” threshold for SDPO
    

---

If you want, next I can translate this into a concrete experimental plan with:

-   dataset split choices (e.g., SWE-bench Lite as eval, generated training issues elsewhere),
    
-   rollout format (how much repo context, how to format test logs),
    
-   and a compute-aware recipe for Qwen3-4B (sequence lengths, batch sizes, top-K distillation, EMA teacher settings) based directly on SDPO’s stability section. [arXiv+1](https://arxiv.org/pdf/2601.20802)

You: 

think about how we can do this, and check current practice: - train a SWE agent end-to-end, starting from a 4B base, two sub-questions: - SDPO is only for single-turn due to its teacher prompt construction, how to extend it to multi-turn? - before SDPO how do we first teach the skill of applying these tools in correct format? Maybe the Self-Distillation Enables Continual Learning approach?

ChatGPT: 

Reasoned for 20m 9s

You: 

Below is a concrete way to think about “end‑to‑end SWE agent training from a 4B base” in the SDPO/SDFT framing, grounded in what SDPO actually did, plus what people currently do for multi‑turn SWE RL and tool-format bootstrapping.

---

## What SDPO was used for in the paper (and why it’s relevant to SWE)

### SDPO’s core trick (dense credit from feedback, no external teacher)

SDPO (Self‑Distillation Policy Optimization) constructs a **self‑teacher** by taking the *same model* and conditioning it on **rich feedback** from the environment (e.g., test failures / runtime errors), then distills the teacher’s **feedback‑aware next‑token distribution** back into the student on the student’s own rollout. [arXiv](https://arxiv.org/html/2601.20802v1)

Concretely, SDPO:

1.  samples a rollout/attempt $y$ from the student policy $\pi_\theta(\cdot \mid x)$
    
2.  runs an environment to obtain feedback $f$ (tokenized)
    
3.  defines a teacher distribution using the same weights but richer context, roughly $\pi_{\text{teach}}(\cdot \mid x, f)$
    
4.  minimizes a **logit‑level distillation loss** (they present it as a KL between next‑token distributions). [arXiv](https://arxiv.org/html/2601.20802v1)
    

The key “teacher prompt construction” is explicitly shown in their Table 2 template: teacher context includes the original prompt, optional “successful previous rollout” (when available), and the environment output / feedback, while re‑evaluating the original student response. [arXiv](https://arxiv.org/html/2601.20802v1)

### Where SDPO was evaluated

They evaluate SDPO in two regimes:

**1) “Learning without rich feedback” (RLVR-ish, only scalar reward)**  
They still apply SDPO by treating **successful attempts in the same rollout group as implicit feedback** for failures (“sample solution if it was already solved with another attempt”). [arXiv+1](https://arxiv.org/html/2601.20802v1)  
They do this on:

-   **Science Q&A** (SciKnowEval L3 subsets)
    
-   **Tool use** (ToolAlpaca: “mapping tool-API specification + user request to correct tool call”) [arXiv+1](https://arxiv.org/html/2601.20802v1)
    

**2) “Learning with rich environment feedback” (coding)**  
They evaluate on **LiveCodeBench v6** and explicitly motivate coding as a setting with rich feedback like runtime errors / unit tests. [arXiv](https://arxiv.org/html/2601.20802v1)  
They use public/private tests and report SDPO > GRPO on that setting. [arXiv+1](https://arxiv.org/html/2601.20802v1)

### A big caveat for small models

SDPO’s gains are **tightly coupled to base model strength / in‑context learning ability**. Their scaling study shows SDPO helps more for larger Qwen3 sizes and can even **underperform** GRPO on small models (they mention Qwen2.5‑1.5B underperforming). [arXiv](https://arxiv.org/html/2601.20802v1)

They also propose a **hybrid SDPO+GRPO** advantage to make training more robust for weaker models. [arXiv](https://arxiv.org/html/2601.20802v1)

This matters a lot for Qwen3‑4B: it’s not tiny like 1.5B, but it’s also not 32B+; you likely want either (a) better initialization for tool/agent format, or (b) hybridization early.

---

## What current practice looks like for training SWE agents end-to-end

### Multi-turn SWE RL is now a real “thing”

A representative “pure RL for multi‑turn SWE” example: **Training Long‑Context, Multi‑Turn SWE Agents with RL** (Nebius) formulates SWE as a **POMDP** with long-horizon trajectories and sparse terminal reward (tests pass/fail), and trains with a modified **DAPO** pipeline. [arXiv](https://arxiv.org/html/2508.03501v1)  
Importantly, they explicitly say an early failure mode is **incorrect instruction following / commands not in the appropriate format**, and they start with **Rejection Fine‑Tuning (RFT)** to fix command formatting before RL. [arXiv](https://arxiv.org/html/2508.03501v1)

### GRPO-to-multi-turn “masking” is standard in practice

DeepSWE (Together) describes the common trick for extending GRPO to agent settings: **mask out environment observations/user messages** (in ChatML) so you only train on the model’s **action tokens**. [Together AI](https://www.together.ai/blog/deepswe)  
They also outline a typical SWE agent action space: Execute Bash / Search / File Editor / Finish, and sparse reward 1 if tests pass, else 0. [Together AI](https://www.together.ai/blog/deepswe)

### Tool-format bootstrapping often uses “segment trajectories into steps”

Nemotron‑Research‑Tool‑N1 (Tool‑N1) is a clean example of how people turn multi‑turn tool trajectories into trainable units:

-   preprocess tool trajectories, discard invalid JSON / invalid tool calls
    
-   **segment each multi‑turn trajectory into single‑step prediction instances**: target one tool call, previous steps as context [Preprints](https://www.preprints.org/manuscript/202504.2471)  
    This is exactly the kind of trick you can reuse to make SDPO work in multi‑turn.
    

### Small models *can* be trained as SWE agents via trajectories

A concrete datapoint on small size: **mini‑coder‑4b** describes itself as a 4B coding agent model trained from a larger coder via distillation and **trained on ~400k SWE trajectories** using “mini-swe-agent” and SWE-smith. [Hugging Face](https://huggingface.co/ricdomolm/mini-coder-4b)  
So: “4B SWE agent” is not hypothetical; trajectory-based post-training is already used at this size.

---

## Subquestion 1 — SDPO is single-turn; how do we extend it to multi-turn?

Your intuition is right: SDPO’s *paper template* is written as “one attempt → feedback → distill on that attempt.” [arXiv](https://arxiv.org/html/2601.20802v1)  
But the underlying idea (“feedback-conditioned self-teacher”) actually generalizes naturally to trajectories if you choose what counts as “an attempt” and what counts as “feedback.”

Here are three viable extensions, ordered from simplest to most “true agent RL”.

### Option A (most practical): Step‑SDPO via trajectory segmentation

Make each **agent step** a “mini single-turn SDPO instance”.

Let a trajectory be:

$$
h_t = (o_0, a_0, o_1, a_1, \dots, o_t)
$$

where $o_t$ is environment observation (tool output, logs), and $a_t$ is the model’s next action (bash command, file edit, submit).

After executing $a_t$, you get $o_{t+1}$. Treat $o_{t+1}$ as **rich feedback for $a_t$**.

Define:

-   student: $\pi_\theta(\cdot \mid h_t)$
    
-   self-teacher: same weights but with hindsight observation included: $\pi_{\text{teach}}(\cdot \mid h_t, o_{t+1})$
    

Then distill on the tokens of $a_t$ only:

$$
\mathcal{L}_{\text{step-SDPO}} = \sum_t \sum_{n \in \text{tokens}(a_t)} \text{KL}\!\left(\pi_{\text{teach}}(\cdot \mid h_t, o_{t+1}, a_{t,<n}) \;\|\; \pi_\theta(\cdot \mid h_t, a_{t,<n})\right)
$$

with stop‑gradient through the teacher (as in SDPO). [arXiv](https://arxiv.org/html/2601.20802v1)

**Why this matches current practice**

-   It’s exactly the “segment trajectories into single-step instances” pattern that Tool‑N1 uses (but with SDPO’s teacher instead of a reward). [Preprints](https://www.preprints.org/manuscript/202504.2471)
    
-   It’s compatible with the GRPO multi‑turn masking trick: you only compute loss on action tokens, not observation tokens. [Together AI](https://www.together.ai/blog/deepswe)
    

**Where this works best**

-   steps where the tool output is immediately informative (e.g., “command not found”, “tests failed with stack trace”, “file not found”, “grep returned nothing”).
    

**Main failure mode**

-   **delayed credit assignment**: a bad early edit may only be revealed after running tests 20 steps later, so $o_{t+1}$ isn’t always helpful.
    

### Option B: Trajectory‑SDPO using terminal feedback as hindsight for many steps

Use the final test output / failure logs as “feedback” for **multiple earlier actions**.

At episode end you have:

-   full transcript $h_T$
    
-   terminal feedback $F$ (full test logs, failing tests, etc.)
    
-   terminal reward $R \in \{0,1\}$
    

Define teacher for each step as:

$$
\pi_{\text{teach}}(\cdot \mid h_t, F)
$$

and distill each step’s action tokens against this “teacher that knows the ending”.

This is closer to how SDPO uses LeetCode‑style feedback to re-evaluate the original attempt under a feedback-augmented context. [arXiv+1](https://arxiv.org/html/2601.20802v1)

**Practical variant** (to control compute):

-   distill only the last $K$ steps (the ones closest to submission),
    
-   or distill only “key actions” (file edits + submit), skipping low-value search steps.
    

### Option C: Multi-turn SDPO as “masked SDPO advantages” inside a GRPO/DAPO loop

SDPO’s authors note you can reuse RLVR pipelines by swapping advantages; and they even propose SDPO+GRPO hybrid. [arXiv+1](https://arxiv.org/html/2601.20802v1)

So you can:

1.  run a multi-turn RL algorithm (GRPO/DAPO) like current SWE RL systems do [arXiv+1](https://arxiv.org/html/2508.03501v1)
    
2.  compute SDPO-style per-token “advantages” from teacher-vs-student disagreement
    
3.  apply them only on action tokens (mask observations)
    

This is the most faithful to “RL on trajectories” but is more engineering heavy.

### Recommendation for Qwen3‑4B

Start with **Option A (Step‑SDPO)** because:

-   it’s the cleanest mapping of SDPO’s “teacher sees feedback about your last action”
    
-   it aligns with the way tool models already segment multi-turn data [Preprints](https://www.preprints.org/manuscript/202504.2471)
    
-   it’s easy to implement with your existing agent runtime logging
    

Then add **Option B** as an auxiliary “terminal hindsight distillation” pass for delayed errors.

---

## Subquestion 2 — Before SDPO, how do we teach correct tool formats?

This is *the* practical blocker in SWE RL. Even the DAPO SWE RL paper says the base model fails mainly due to **commands not in the appropriate format**, and they fix it with **Rejection Fine‑Tuning** before RL. [arXiv](https://arxiv.org/html/2508.03501v1)

You asked whether **Self‑Distillation Enables Continual Learning** (SDFT) can be the precursor. Yes — it’s almost tailor‑made for this.

### Why SDFT fits tool/agent formatting

SDFT builds a teacher by conditioning the same model on an **expert demonstration** $d$, then trains the student (without $d$) by minimizing a reverse‑KL objective on the student’s own rollouts. [arXiv+1](https://arxiv.org/html/2601.19897v1)

The teacher prompt is literally:

-   Question
    
-   “This is an example response… <Demonstration>”
    
-   “Now answer…” [arXiv](https://arxiv.org/html/2601.19897v1)
    

So to teach tool format:

-   your demonstrations are short “correctly formatted tool calls” (or mini agent transcripts)
    
-   the teacher sees them; the student does not
    
-   the student learns to output correct format *without needing few-shot demos at inference*
    

SDFT is explicitly evaluated on **Tool Use** with ToolAlpaca (mapping tool-API spec + user request to correct tool call). [arXiv+1](https://arxiv.org/html/2601.19897v1)  
That’s essentially the same formatting problem you have for SWE tools (bash/search/edit/submit).

### Minimal “format-first” curriculum that mimics what works in practice

Here’s a pragmatic 3‑stage approach for Qwen3‑4B.

#### Stage 0: Pure format SFT / RFT (fast)

Goal: make the model reliably emit *parseable* actions (JSON / XML / tool\_call tags / command schema).

This is analogous to the DAPO pipeline’s “RFT to fix command format”. [arXiv](https://arxiv.org/html/2508.03501v1)

Data: synthetic + small curated:

-   “invalid → corrected” tool call examples
    
-   strict schema constraints (JSON parse, required fields)
    
-   optional “tool signature” context
    

#### Stage 1: SDFT to internalize tool behavior without forgetting

Now switch from off-policy SFT to on-policy distillation:

-   student samples outputs for tool tasks
    
-   teacher is the same model conditioned on a demonstration of correct tool usage
    
-   optimize reverse‑KL distillation objective (their Eq. 1–2 discussion). [arXiv+1](https://arxiv.org/html/2601.19897v1)
    

This should yield:

-   stronger robustness when the model drifts into weird states (because it’s trained on its own rollouts)
    
-   less catastrophic forgetting than sequential SFT [arXiv](https://arxiv.org/html/2601.19897v1)
    

#### Stage 2: Tool‑execution + SDPO (or hybrid SDPO+GRPO) for real environment feedback

Once tool calls are stable, bring in SWE environment signals:

-   failing tests
    
-   stack traces
    
-   lints
    
-   “file not found”
    
-   etc.
    

That’s exactly SDPO’s “rich environment feedback” premise. [arXiv+1](https://arxiv.org/html/2601.20802v1)

### A useful “engineering trick”: always mask observations, train only on action tokens

Current multi-turn RL practice highlights this explicitly (mask env observations / user messages). [Together AI](https://www.together.ai/blog/deepswe)  
You should do the same for:

-   SDFT on multi-turn transcripts
    
-   SDPO step-level distillation
    

That avoids the model trying to “predict” tool output and keeps learning focused on selecting actions.

---

## Putting it together: an end-to-end recipe for a local Qwen3‑4B SWE agent

### Environment choice (for local training)

If you want something local and tractable, pick one of:

-   a small SWE-bench subset / self-hosted docker tasks (mini action space like DeepSWE’s: bash/search/edit/submit) [Together AI](https://www.together.ai/blog/deepswe)
    
-   an environment with curated executable tasks like SWE-smith / SWE-Gym / R2E-Gym (but run a smaller subset) [arXiv+1](https://arxiv.org/html/2508.03501v1)
    

Given that there are already 4B agent-ish models trained on large sets of SWE trajectories, trajectory logging + fine-tuning is realistic at this size. [Hugging Face](https://huggingface.co/ricdomolm/mini-coder-4b)

### Training phases

1.  **Format bootstrapping (SFT/RFT)**
    

-   reward: parseability + allowed tool name + required args
    
-   stop when “invalid tool call” rate is near zero
    

2.  **SDFT for tool competency**
    

-   demonstrations: short transcripts of correct tool calls (and maybe minimal reasoning)
    
-   student generates; teacher sees demo; distill reverse-KL [arXiv+1](https://arxiv.org/html/2601.19897v1)
    
-   keep a KL-to-base regularization mindset: SDFT papers explicitly track teacher staying closer to base policy than SFT, which helps reduce forgetting. [arXiv](https://arxiv.org/html/2601.19897v1)
    

3.  **Agent imitation on SWE trajectories (optional but helpful)**
    

-   if you have trajectory data (from scripted heuristics, small teacher, or your own agent), do a short on-policy-ish imitation stage (can still use SDFT with demonstrations as the “teacher context”)
    

4.  **Step‑SDPO on real environment feedback**
    

-   for each action $a_t$, treat next observation $o_{t+1}$ as feedback
    
-   teacher sees $o_{t+1}$ and re-evaluates $a_t$
    
-   distill on $a_t$ tokens only (mask obs tokens)
    

5.  **Add terminal hindsight SDPO (delayed errors)**
    

-   teacher sees full final logs $F$
    
-   distill key steps (edits + submit)
    

6.  **Optional: hybridize early**  
    Because SDPO can be brittle on weaker models, consider mixing in GRPO/DAPO-style sparse reward advantages early, similar in spirit to SDPO+GRPO being more robust on weaker models. [arXiv](https://arxiv.org/html/2601.20802v1)
    

---

## Measuring “teacher ICL progression” (what’s been done + what to do for SWE)

You flagged this as crucial, and SDPO actually contains a direct precedent: they explicitly study that the **self-teacher improves during training** and plot teacher generative accuracy vs student, showing the student can surpass the initial teacher. [arXiv](https://arxiv.org/html/2601.20802v1)  
That’s a very straightforward “teacher ICL capability proxy”: *how good is the feedback-conditioned policy at solving?*

SDFT similarly validates its “ICL assumption” and measures divergence properties (teacher vs base) as part of explaining why it avoids forgetting. [arXiv+1](https://arxiv.org/html/2601.19897v1)

For SWE + multi-turn SDPO, I’d operationalize “teacher ICL progression” with 3 metrics:

### 1) Teacher solve-rate given feedback (retrospection competence)

Freeze a held-out batch of failures (trajectories or steps) and periodically evaluate:

-   student: $\pi_\theta(\cdot \mid h_t)$
    
-   teacher: $\pi_\theta(\cdot \mid h_t, \text{feedback})$
    

Measure:

-   step-level: “did the teacher choose an action that reduces error?” (e.g., fixes JSON, finds file, passes reproduction)
    
-   episode-level: “if we re-plan from step $t$ with teacher context, does success rate improve?”
    

This is the closest to SDPO’s “teacher accuracy” plot idea. [arXiv](https://arxiv.org/html/2601.20802v1)

### 2) Teacher disagreement / advantage density (is there learning signal?)

Track over training:

-   fraction of action tokens where teacher and student differ meaningfully
    
-   average KL(teacher || student) on action tokens
    

If this collapses to ~0 early, SDPO won’t learn (no gradient); if it’s huge and noisy, teacher may be unhelpful.

### 3) “Demo-conditioned uplift” (SDFT-style)

For SDFT pretraining and beyond, track:

-   performance uplift from adding 1–2 demonstrations to context
    
-   and how that uplift changes over time
    

If demo-conditioned uplift shrinks because the student internalized the behavior, that’s good (it means the student absorbed the ICL behavior into weights).

---

## Bottom line answers to your two sub-questions

### How to extend SDPO to multi-turn?

Use **Step‑SDPO**: treat each tool action as the “attempt” and the next observation/tool output as feedback, distilling teacher’s hindsight distribution back into the student, masking observation tokens. This mirrors the common “mask observations” practice in multi-turn GRPO RL. [Together AI+1](https://www.together.ai/blog/deepswe)  
Add a terminal hindsight pass for delayed feedback.

### Before SDPO, how to teach tool formatting?

Use a **format-first** stage (RFT/SFT) like multi-turn SWE RL pipelines do, then use **SDFT** to distill demonstration-conditioned tool competence into the student on-policy (and reduce forgetting), and only then move to SDPO with real environment feedback. [arXiv+2arXiv+2](https://arxiv.org/html/2508.03501v1)

---

If you want, I can sketch a concrete “agent message schema” (what you mask vs optimize) and a pseudocode training loop for **Step‑SDPO + SDFT pretraining** for an R2E‑Gym-style tool set (bash/search/edit/submit).