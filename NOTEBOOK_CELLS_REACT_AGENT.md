# ReAct Agent - Notebook Integration Guide

## 📋 Step-by-Step Instructions

### STEP 1: Comment Out Old Implementation

Find and **COMMENT OUT** these sections in your notebook:

#### 1.1 Comment out the old `check_relevance_node` function
Search for: `def check_relevance_node(state: AgentState) -> AgentState:`
- This is around line 2167 in your notebook
- Comment out the ENTIRE function (until the return statement)
- Keep it commented for now (you can delete later if ReAct works well)

```python
# # ============================================================================
# # Part 3: Enhanced Check Relevance with Memory
# # ============================================================================
#
# def check_relevance_node(state: AgentState) -> AgentState:
#     # ... COMMENT OUT THE ENTIRE OLD FUNCTION ...
#     return {
#         **state,
#         'relevance_check': relevance_check,
#         'previous_dataframes': relevance_check.relevant_dataframes,
#         'reranker_scores': reranker_scores,
#         'current_stage': 'relevance_checked'
#     }
```

#### 1.2 Comment out the old routing function
Search for: `def route_after_relevance(state: AgentState)`
- Comment out this function

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

### STEP 2: Add New Cells

Add the following cells **AFTER** your commented-out `check_relevance_node`:

---

## 📝 CELL 1: Import ReAct Agent Dependencies

```python
# ============================================================================
# REACT AGENT IMPLEMENTATION - Part 1: Imports
# ============================================================================

from typing import List, Dict, Any, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
import json
import torch
import random
import numpy as np

print("✅ ReAct Agent dependencies imported successfully")
```

---

## 📝 CELL 2: Define Tools for ReAct Agent

```python
# ============================================================================
# REACT AGENT IMPLEMENTATION - Part 2: Tool Definitions
# ============================================================================

@tool
def retrieve_and_rank_dataframes(
    question: str,
    top_k: int = 12,
    rerank_top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve candidate dataframes and rank them by relevance using hybrid search + CrossEncoder.

    This tool performs:
    1. Hybrid retrieval (BM25 70% + Semantic 30%)
    2. CrossEncoder reranking for better accuracy

    Args:
        question: User's question to search for
        top_k: Number of initial candidates to retrieve
        rerank_top_k: Number of top candidates to return after reranking

    Returns:
        Dictionary containing:
        - candidates: List of candidate dataframe metadata
        - scores: Dictionary mapping dataframe names to relevance scores (0-1)
        - top_score: Highest relevance score
        - avg_score: Average relevance score
    """
    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize retriever and cross-encoder
    retriever = initialize_retriever()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = initialize_cross_encoder(device=device)

    # Stage 1: Hybrid retrieval
    initial_candidates = retriever.retrieve_top_k(question, top_k=top_k)
    candidate_metadata = [md for _, md in initial_candidates]

    if not candidate_metadata:
        return {
            "candidates": [],
            "scores": {},
            "top_score": 0.0,
            "avg_score": 0.0,
            "message": "No candidates found"
        }

    # Stage 2: CrossEncoder reranking
    pairs = [(question, retriever._build_search_text(md)) for md in candidate_metadata]
    scores = cross_encoder.predict(pairs)

    # Sort and select top_k
    scored = sorted(
        zip(candidate_metadata, scores),
        key=lambda x: x[1],
        reverse=True
    )[:rerank_top_k]

    # Build result
    candidates = [md for md, _ in scored]
    score_dict = {md.name: float(score) for md, score in scored}

    top_score = float(max(scores)) if len(scores) > 0 else 0.0
    avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

    return {
        "candidates": [
            {
                "name": md.name,
                "category": md.category,
                "sub_category": md.sub_category,
                "description_en": md.description_en,
                "description_ar": md.description_ar,
                "columns": md.columns[:10],  # Limit for context
                "sample_values": {k: v[:5] for k, v in (md.sample_values or {}).items()}
            }
            for md in candidates
        ],
        "scores": score_dict,
        "top_score": top_score,
        "avg_score": avg_score,
        "message": f"Found {len(candidates)} relevant candidates"
    }


@tool
def check_conversation_context(
    current_question: str,
    message_history: List[Dict[str, str]],
    previous_dataframes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze if current question is a follow-up and should inherit context from history.

    Args:
        current_question: Current user question
        message_history: Previous conversation messages
        previous_dataframes: Dataframes used in previous turn

    Returns:
        Dictionary containing:
        - is_followup: Whether this is a follow-up question
        - should_use_history: Whether to inherit dataframes from history
        - previous_dataframes: List of dataframe names from history (if applicable)
        - reasoning: Explanation of the decision
    """
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    if not message_history or len(message_history) == 0:
        return {
            "is_followup": False,
            "should_use_history": False,
            "previous_dataframes": [],
            "reasoning": "No conversation history available"
        }

    # Get last 3 messages for context
    recent_history = message_history[-3:]
    conversation_context = "\n".join([
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')[:200]}"
        for msg in recent_history
    ])

    class ContextRelevance(BaseModel):
        is_followup: bool = Field(description="Is this a follow-up question?")
        should_use_history: bool = Field(description="Should we use dataframes from history?")
        reasoning: str = Field(description="Explanation of the decision")

    parser = PydanticOutputParser(pydantic_object=ContextRelevance)

    template = """You are analyzing conversation context for GASTAT chatbot.

Previous Conversation:
{conversation_context}

Previous Dataframes Used: {previous_dfs}

Current Question: {current_question}

Determine:
1. Is this a follow-up question that refers to previous context?
2. Should we inherit dataframes from the previous turn?

Guidelines:
- Follow-up indicators: "it", "this", "that", "these", "also", "what about", "how about", "وماذا", "أيضا", "كذلك"
- If question is completely new topic, is_followup=false
- If follow-up AND topic matches, should_use_history=true

{format_instructions}

Respond with JSON only:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["conversation_context", "previous_dfs", "current_question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = ChatOllama(model="deepseek-r1:32b", temperature=0)
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "conversation_context": conversation_context,
            "previous_dfs": ", ".join(previous_dataframes) if previous_dataframes else "None",
            "current_question": current_question
        })

        return {
            "is_followup": result.is_followup,
            "should_use_history": result.should_use_history,
            "previous_dataframes": previous_dataframes if result.should_use_history else [],
            "reasoning": result.reasoning
        }
    except Exception as e:
        print(f"⚠️  Error checking context: {e}")
        return {
            "is_followup": False,
            "should_use_history": False,
            "previous_dataframes": [],
            "reasoning": f"Error occurred: {str(e)}"
        }


@tool
def validate_dataframe_relevance(
    question: str,
    candidates: List[Dict[str, Any]],
    scores: Dict[str, float],
    detected_language: str = "en",
    conversation_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use LLM to validate which dataframes are truly relevant to the question.

    This tool provides semantic understanding beyond just retrieval scores.

    Args:
        question: User's question
        candidates: List of candidate dataframe metadata
        scores: Relevance scores from retrieval/reranking
        detected_language: Language of the question (en/ar)
        conversation_context: Optional conversation history

    Returns:
        Dictionary containing:
        - relevant_dataframes: List of relevant dataframe names
        - reasoning: Detailed explanation
        - analysis: Per-dataframe analysis
    """
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    from enum import Enum

    if not candidates:
        return {
            "relevant_dataframes": [],
            "reasoning": "No candidates provided",
            "analysis": {}
        }

    # Build dataframe context
    desc_field = "description_ar" if detected_language == "ar" else "description_en"
    context_parts = []

    for i, df in enumerate(candidates, 1):
        score = scores.get(df['name'], 0.0)
        desc = df.get(desc_field, df.get('description_en', 'N/A'))

        context_parts.append(f"""
DataFrame {i}: {df['name']} [Retrieval Score: {score:.3f}]
Category: {df.get('category', 'N/A')}
Description: {desc}
Columns: {', '.join(df.get('columns', [])[:10])}
Sample Values: {json.dumps(df.get('sample_values', {}), ensure_ascii=False)[:200]}
""")

    dataframe_context = "\n".join(context_parts)

    # Create dynamic enum
    DataFrameEnum = Enum('DataFrameEnum', {df['name']: df['name'] for df in candidates})

    class RelevanceValidation(BaseModel):
        relevant_dataframes: List[DataFrameEnum] = Field(
            description="List of EXACT dataframe names that are relevant"
        )
        reasoning: str = Field(
            description="Detailed explanation of why these dataframes are relevant"
        )

    parser = PydanticOutputParser(pydantic_object=RelevanceValidation)

    conv_context = f"\n\nConversation Context:\n{conversation_context}" if conversation_context else ""

    template = """You are an expert data analyst for GASTAT (General Authority for Statistics, Saudi Arabia).
Analyze the user's question and determine which dataframes are TRULY relevant.

**LANGUAGE**: Question is in **{detected_language}**

{conv_context}

AVAILABLE DATAFRAMES:
{dataframe_context}

USER QUESTION: {question}

CRITICAL RULES:
1. Return ALL dataframes that can help answer the question
2. Consider both exact matches and related context
3. Look at sample values carefully - they reveal actual data content
4. For multi-topic questions, include dataframes for EACH topic
5. If NO dataframes match, return empty list
6. Base decision on semantic meaning, not just keyword matching

DECISION CRITERIA:
- High retrieval score (>0.7) + relevant description = INCLUDE
- Medium score (0.3-0.7) but highly relevant content = INCLUDE
- Low score (<0.3) = Only include if clearly relevant from description/samples
- Unrelated topic regardless of score = EXCLUDE

{format_instructions}

Respond with JSON only:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["detected_language", "conv_context", "dataframe_context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = ChatOllama(model="deepseek-r1:32b", temperature=0)
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "detected_language": detected_language,
            "conv_context": conv_context,
            "dataframe_context": dataframe_context,
            "question": question
        })

        # Extract enum values
        relevant_dfs = [df.value for df in result.relevant_dataframes]

        return {
            "relevant_dataframes": relevant_dfs,
            "reasoning": result.reasoning,
            "analysis": {df: scores.get(df, 0.0) for df in relevant_dfs}
        }
    except Exception as e:
        print(f"⚠️  Error validating relevance: {e}")
        # Fallback: use score threshold
        fallback_dfs = [df['name'] for df in candidates if scores.get(df['name'], 0) > 0.5]
        return {
            "relevant_dataframes": fallback_dfs,
            "reasoning": f"Fallback to score threshold due to error: {str(e)}",
            "analysis": {df: scores.get(df, 0.0) for df in fallback_dfs}
        }

print("✅ ReAct tools defined successfully")
print("   - retrieve_and_rank_dataframes")
print("   - check_conversation_context")
print("   - validate_dataframe_relevance")
```

---

## 📝 CELL 3: Define ReAct Agent Node

```python
# ============================================================================
# REACT AGENT IMPLEMENTATION - Part 3: ReAct Agent Node
# ============================================================================

def check_relevance_node(state: AgentState) -> AgentState:
    """
    🤖 NEW: ReAct agent node that uses tools to determine dataframe relevance.

    Key difference from original: NO FIXED THRESHOLD
    Agent decides through multi-step reasoning.
    """
    question = state['question']
    message_history = state.get('message_history', [])
    previous_dataframes = state.get('previous_dataframes', [])
    detected_language = state.get('detected_language', 'en')

    print(f"\n[NODE: check_relevance_react] Starting ReAct agent for: {question}")

    # Prepare tools
    tools = [
        retrieve_and_rank_dataframes,
        check_conversation_context,
        validate_dataframe_relevance
    ]

    # System prompt for the agent
    system_prompt = f"""You are a specialized relevance checking agent for GASTAT chatbot.

Your task: Determine which dataframes are relevant to answer the user's question.

Available Tools:
1. retrieve_and_rank_dataframes - Get candidate dataframes with relevance scores
2. check_conversation_context - Analyze if this is a follow-up question
3. validate_dataframe_relevance - Validate which candidates are truly relevant

WORKFLOW (YOU MUST FOLLOW THIS ORDER):
1. FIRST: Use check_conversation_context to see if this is a follow-up
2. SECOND: Use retrieve_and_rank_dataframes to get candidates and scores
3. THIRD: Use validate_dataframe_relevance to make final decision

IMPORTANT RULES:
- You MUST use ALL three tools in sequence
- DO NOT use fixed thresholds (like score > 0.5) for decisions
- Base your final decision on semantic relevance, not just scores
- Consider conversation history if it's a follow-up
- If a dataframe has a lower score but clearly matches the question content, INCLUDE it
- Return clear reasoning for your decision

Current Context:
- Question: {question}
- Language: {detected_language}
- Has conversation history: {len(message_history) > 0}
- Previous dataframes: {previous_dataframes}

After using all tools, provide your final answer in this EXACT format:
RELEVANT_DATAFRAMES: [list of dataframe names]
REASONING: [your detailed reasoning based on tool outputs]
"""

    # Create ReAct agent
    llm = ChatOllama(model=state.get('model_name', 'deepseek-r1:32b'), temperature=0)
    agent = create_react_agent(llm, tools)

    # Prepare input for agent
    agent_input = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Analyze relevance for this question: {question}

Context Details:
- Message History: {json.dumps(message_history[-3:], ensure_ascii=False) if message_history else "None"}
- Previous Dataframes: {previous_dataframes}
- Detected Language: {detected_language}

Use the three tools in order and provide your final decision.""")
        ]
    }

    # Invoke agent
    try:
        result = agent.invoke(agent_input)

        # Parse agent's final response
        final_message = result['messages'][-1].content
        print(f"\n🤖 Agent Decision Summary:")
        print(f"{final_message[:300]}...")

        # Extract dataframes and reasoning from agent response
        relevant_dfs = []
        reasoning = ""

        # Parse the structured output
        if "RELEVANT_DATAFRAMES:" in final_message:
            df_line = final_message.split("RELEVANT_DATAFRAMES:")[1].split("REASONING:")[0].strip()
            # Try to parse as list
            try:
                import ast
                relevant_dfs = ast.literal_eval(df_line)
                if not isinstance(relevant_dfs, list):
                    relevant_dfs = [relevant_dfs] if relevant_dfs else []
            except:
                # Fallback: extract names
                relevant_dfs = [name.strip().strip("'\"") for name in df_line.replace('[', '').replace(']', '').split(',') if name.strip()]

        if "REASONING:" in final_message:
            reasoning = final_message.split("REASONING:")[1].strip()

        # Create RelevanceCheck result (compatible with your existing state structure)
        relevance_check_result = type('RelevanceCheck', (), {
            'is_relevant': len(relevant_dfs) > 0,
            'relevant_dataframes': relevant_dfs,
            'confidence_score': 1.0 if len(relevant_dfs) > 0 else 0.0,
            'reasoning': reasoning or final_message
        })()

        print(f"✅ Agent selected: {relevant_dfs}")
        print(f"📝 Reasoning: {reasoning[:150]}...")

        return {
            **state,
            'relevance_check': relevance_check_result,
            'previous_dataframes': relevant_dfs,
            'current_stage': 'relevance_checked'
        }

    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to empty result
        relevance_check_result = type('RelevanceCheck', (), {
            'is_relevant': False,
            'relevant_dataframes': [],
            'confidence_score': 0.0,
            'reasoning': f"Agent failed with error: {str(e)}"
        })()

        return {
            **state,
            'relevance_check': relevance_check_result,
            'current_stage': 'relevance_checked'
        }


print("✅ ReAct agent node defined: check_relevance_node()")
```

---

## 📝 CELL 4: Define New Routing Function

```python
# ============================================================================
# REACT AGENT IMPLEMENTATION - Part 4: Routing Logic (No Threshold!)
# ============================================================================

def route_after_relevance(state: AgentState) -> Literal["decompose_query", "handle_irrelevant"]:
    """
    🆕 Route based on ReAct agent's decision (NO FIXED THRESHOLD).

    The agent has already made an intelligent decision through multi-step reasoning.
    We simply check if relevant dataframes were found.
    """
    relevance = state['relevance_check']

    # Simple boolean check - agent already decided through reasoning
    if relevance.relevant_dataframes and len(relevance.relevant_dataframes) > 0:
        print(f"✅ Routing to decompose_query - Found {len(relevance.relevant_dataframes)} relevant dataframes")
        return "decompose_query"
    else:
        print(f"❌ Routing to handle_irrelevant - No relevant dataframes found")
        return "handle_irrelevant"

print("✅ New routing function defined: route_after_relevance() [NO threshold!]")
```

---

## 📝 CELL 5: Test the Implementation

```python
# ============================================================================
# TEST: ReAct Agent with Full Workflow
# ============================================================================

# Rebuild the workflow (this will use the NEW check_relevance_node and route_after_relevance)
app = build_agent_workflow()

print("✅ Workflow rebuilt with ReAct agent")
print("   The new check_relevance_node uses ReAct agent with NO fixed threshold")
print("   Ready to test with your full workflow!")
```

---

## 📝 CELL 6: Run Your Existing Test

```python
# ============================================================================
# RUN YOUR TEST (Same as before - it will now use ReAct agent internally)
# ============================================================================

# Use your existing test code from the last cell
# Example:

question = "عدد المعتمرين سنة 1441"

result = app.invoke({
    'conversation_id': 'test_react_001',
    'question': question,
    'original_question': question,
    'model_name': 'deepseek-r1:32b',
    'message_history': [],
    'previous_dataframes': [],
    'previous_results': None,
    'available_dataframes': [],
    'relevance_check': None,
    'query_decomposition': None,
    'current_df_index': 0,
    'dataframes_to_process': [],
    'intermediate_results': [],
    'final_response': None,
    'current_stage': 'initialized',
    'detected_language': None,
    'required_desc_field': None,
    'filtered_columns_context': None,
    'context_evaluation': None,
    'should_use_history': False,
    'previous_query': None,
})

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"\nQuestion: {result['question']}")
print(f"Language: {result.get('detected_language', 'N/A')}")
print(f"\n📊 Relevance Check (ReAct Agent):")
print(f"   Is Relevant: {result['relevance_check'].is_relevant}")
print(f"   Dataframes: {result['relevance_check'].relevant_dataframes}")
print(f"   Reasoning: {result['relevance_check'].reasoning[:200]}...")

if result.get('final_response'):
    print(f"\n✅ Final Answer:")
    print(result['final_response'].answer)
else:
    print(f"\n❌ No final response generated")
```

---

## ✅ Summary

After adding these cells:

1. **Cell 1**: Imports
2. **Cell 2**: Three tools for the ReAct agent
3. **Cell 3**: New `check_relevance_node` using ReAct agent
4. **Cell 4**: New `route_after_relevance` (no threshold)
5. **Cell 5**: Rebuild workflow
6. **Cell 6**: Test with your existing test code

### What Changed:

- ✅ **Old**: `if confidence_score > 0.5` → **New**: Agent reasoning
- ✅ **Old**: Single LLM call → **New**: Multi-step reasoning with 3 tools
- ✅ **Old**: Fixed threshold → **New**: Adaptive decision making
- ✅ Your existing workflow nodes remain unchanged
- ✅ Test the FULL flow like before - ReAct works internally

### The ReAct agent will:
1. Check if it's a follow-up question
2. Retrieve and rank candidate dataframes
3. Validate relevance semantically
4. Make final decision through reasoning (no threshold!)
