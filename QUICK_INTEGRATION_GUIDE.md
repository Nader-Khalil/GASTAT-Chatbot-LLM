# Quick Integration Guide - ReAct Agent

## 🎯 Goal
Replace your fixed threshold (`confidence_score > 0.5`) with ReAct agent for better accuracy.

---

## 📍 STEP 1: Find and Comment Out (2 functions)

### Location 1: Old check_relevance_node (~line 2167)

**FIND THIS:**
```python
def check_relevance_node(state: AgentState) -> AgentState:
    # from enum import Enum
    # import torch
    # torch.manual_seed(42)
    import torch
    import random
    ...
```

**COMMENT IT OUT** by adding `#` at the start of every line:
```python
# def check_relevance_node(state: AgentState) -> AgentState:
#     # from enum import Enum
#     # import torch
#     # torch.manual_seed(42)
#     import torch
#     import random
#     ...
#     return {
#         **state,
#         'relevance_check': relevance_check,
#         'previous_dataframes': relevance_check.relevant_dataframes,
#         'reranker_scores': reranker_scores,
#         'current_stage': 'relevance_checked'
#     }
```

**TIP:** Select the entire function and use `Ctrl+/` (or `Cmd+/` on Mac) to comment all at once.

---

### Location 2: Old route_after_relevance (~line 3730)

**FIND THIS:**
```python
def route_after_relevance(state: AgentState) -> Literal["decompose_query", "handle_irrelevant"]:
    """Route based on relevance check"""
    relevance = state['relevance_check']

    if relevance.is_relevant and relevance.confidence_score > 0.5:
        return "decompose_query"
    else:
        return "handle_irrelevant"
```

**COMMENT IT OUT:**
```python
# def route_after_relevance(state: AgentState) -> Literal["decompose_query", "handle_irrelevant"]:
#     """Route based on relevance check"""
#     relevance = state['relevance_check']
#
#     if relevance.is_relevant and relevance.confidence_score > 0.5:
#         return "decompose_query"
#     else:
#         return "handle_irrelevant"
```

---

## 📍 STEP 2: Add New Cells

**WHERE:** Right after your commented-out `check_relevance_node`

Add 6 new cells in order. Copy each cell from `NOTEBOOK_CELLS_REACT_AGENT.md`:

1. **Cell 1: Imports** - Import ReAct dependencies
2. **Cell 2: Tools** - Define the 3 tools for the agent
3. **Cell 3: Agent Node** - New `check_relevance_node` with ReAct
4. **Cell 4: Routing** - New `route_after_relevance` (no threshold)
5. **Cell 5: Rebuild** - Rebuild workflow
6. **Cell 6: Test** - Run your test

---

## 📍 STEP 3: Run and Test

Execute all the new cells in order, then run your existing test cell at the end of the notebook.

The ReAct agent will work internally - you'll see output like:

```
[NODE: check_relevance_react] Starting ReAct agent for: عدد المعتمرين سنة 1441

🤖 Agent Decision Summary:
After analyzing the question using all three tools:

Step 1: Conversation context check - No previous history
Step 2: Retrieved 5 candidates with scores
Step 3: Validated semantic relevance

RELEVANT_DATAFRAMES: ['df_umrah']
REASONING: The question asks for umrah statistics...

✅ Agent selected: ['df_umrah']
```

---

## 🔍 What to Look For

### ✅ Success Indicators:
- Agent output shows all 3 tool calls
- Final decision includes dataframes
- Reasoning explains why dataframes were selected
- Rest of workflow runs normally (decompose_query → generate_code → etc.)

### ❌ Potential Issues:

**Issue 1: "Tool not found" error**
- **Fix:** Make sure Cell 2 (tools) ran successfully
- Check output: `✅ ReAct tools defined successfully`

**Issue 2: Agent takes too long**
- **Fix:** In Cell 3, change model from `deepseek-r1:32b` to `llama3.1:8b`
```python
llm = ChatOllama(model="llama3.1:8b", temperature=0)  # Faster
```

**Issue 3: Parsing errors**
- **Fix:** Check agent output - it should end with `RELEVANT_DATAFRAMES:` and `REASONING:`
- If format is wrong, the code has fallback parsing

---

## 📊 Compare Results

### Old Implementation:
```
Confidence Score: 0.95
Threshold Check: 0.95 > 0.5 ✅
Decision: Relevant
```

### New ReAct Agent:
```
Tool 1: No follow-up detected
Tool 2: Found 5 candidates (top score: 0.92)
Tool 3: Validated - df_umrah matches semantically
Decision: Relevant (through reasoning, not threshold)
```

---

## 🎯 Key Differences

| Aspect | Old | New (ReAct) |
|--------|-----|-------------|
| Decision Logic | `if score > 0.5` | Multi-step reasoning |
| Tools Used | None | 3 tools (context, retrieval, validation) |
| Edge Cases | Misses borderline | Handles through reasoning |
| Follow-ups | Limited | Explicit context check |
| Threshold | Fixed 0.5 | None - adaptive |

---

## ⚡ Quick Troubleshooting

**Q: Can I see what tools the agent used?**
A: Yes, check the console output - it shows each tool call.

**Q: What if agent gives wrong results?**
A: Check the reasoning - it explains the decision. You can adjust the system prompt in Cell 3.

**Q: Can I still see the old implementation results?**
A: Yes - uncomment the old `check_relevance_node` temporarily and rename it to `check_relevance_node_old` for comparison.

**Q: How do I revert back?**
A: Uncomment the two functions from Step 1, delete the 6 new cells, and rebuild the workflow.

---

## ✅ Checklist

- [ ] Commented out old `check_relevance_node` (~line 2167)
- [ ] Commented out old `route_after_relevance` (~line 3730)
- [ ] Added Cell 1 (Imports) - runs without errors
- [ ] Added Cell 2 (Tools) - shows "✅ ReAct tools defined"
- [ ] Added Cell 3 (Agent Node) - shows "✅ ReAct agent node defined"
- [ ] Added Cell 4 (Routing) - shows "✅ New routing function defined"
- [ ] Added Cell 5 (Rebuild) - shows "✅ Workflow rebuilt"
- [ ] Ran Cell 6 (Test) - agent output appears
- [ ] Final result looks correct

---

**Ready to start? Open `NOTEBOOK_CELLS_REACT_AGENT.md` and copy Cell 1!**
