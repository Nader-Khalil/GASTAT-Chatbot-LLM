"""
ReAct Agent Implementation for Relevance Checking
==================================================
This module implements a LangGraph ReAct agent that uses tools to determine
dataframe relevance WITHOUT fixed thresholds for better generalization.

Key Features:
- Multi-step reasoning with tool use
- No hard-coded confidence thresholds
- Context-aware decision making
- Adaptive to different query types
"""

from typing import List, Dict, Any, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
import json


# ============================================================================
# Part 1: Tool Definitions for ReAct Agent
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
    import torch
    import random
    import numpy as np
    from sentence_transformers import CrossEncoder

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
                "sample_values": {k: v[:5] for k, v in (md.sample_values or {}).items()}  # Limit samples
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
    from langchain_community.chat_models import ChatOllama

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
- Follow-up indicators: "it", "this", "that", "these", "also", "what about", "how about"
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
        print(f"Error checking context: {e}")
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
    from langchain_community.chat_models import ChatOllama
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
        print(f"Error validating relevance: {e}")
        # Fallback: use score threshold
        fallback_dfs = [df['name'] for df in candidates if scores.get(df['name'], 0) > 0.5]
        return {
            "relevant_dataframes": fallback_dfs,
            "reasoning": f"Fallback to score threshold due to error: {str(e)}",
            "analysis": {df: scores.get(df, 0.0) for df in fallback_dfs}
        }


# ============================================================================
# Part 2: ReAct Agent Node
# ============================================================================

def relevance_react_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct agent node that uses tools to determine dataframe relevance.

    Key difference from original: NO FIXED THRESHOLD
    Agent decides through multi-step reasoning.
    """
    print(f"\n[NODE: relevance_react_agent] Starting ReAct agent for: {state['question']}")

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

WORKFLOW:
1. First, check if this is a follow-up question using conversation context
2. Then retrieve and rank candidate dataframes
3. Finally, validate which candidates are truly relevant

IMPORTANT RULES:
- Use ALL three tools in sequence for comprehensive analysis
- DO NOT use fixed thresholds (like >0.5) for decisions
- Base your final decision on semantic relevance, not just scores
- Consider conversation history if it's a follow-up
- Return clear reasoning for your decision

Current Context:
- Question: {state['question']}
- Language: {state.get('detected_language', 'en')}
- Has conversation history: {len(state.get('message_history', [])) > 0}

After using all tools, provide your final answer in this EXACT format:
RELEVANT_DATAFRAMES: [list of dataframe names]
REASONING: [your detailed reasoning]
"""

    # Create ReAct agent
    llm = ChatOllama(model=state.get('model_name', 'deepseek-r1:32b'), temperature=0)
    agent = create_react_agent(llm, tools)

    # Prepare input for agent
    agent_input = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Analyze relevance for this question: {state['question']}

Message History: {json.dumps(state.get('message_history', [])[-3:], ensure_ascii=False)}
Previous Dataframes: {state.get('previous_dataframes', [])}
Detected Language: {state.get('detected_language', 'en')}

Use the tools to determine which dataframes are relevant.""")
        ]
    }

    # Invoke agent
    try:
        result = agent.invoke(agent_input)

        # Parse agent's final response
        final_message = result['messages'][-1].content
        print(f"\n🤖 Agent Decision:\n{final_message}\n")

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
                    relevant_dfs = [relevant_dfs]
            except:
                # Fallback: extract names
                relevant_dfs = [name.strip() for name in df_line.replace('[', '').replace(']', '').split(',') if name.strip()]

        if "REASONING:" in final_message:
            reasoning = final_message.split("REASONING:")[1].strip()

        # Create RelevanceCheck result
        relevance_check = {
            "is_relevant": len(relevant_dfs) > 0,
            "relevant_dataframes": relevant_dfs,
            "confidence_score": 1.0 if len(relevant_dfs) > 0 else 0.0,  # Binary decision by agent
            "reasoning": reasoning or final_message,
            "agent_trace": [msg.content for msg in result['messages']]  # Full trace for debugging
        }

        print(f"✅ Agent selected: {relevant_dfs}")
        print(f"📝 Reasoning: {reasoning[:200]}...")

        return {
            **state,
            'relevance_check': relevance_check,
            'previous_dataframes': relevant_dfs,
            'current_stage': 'relevance_checked'
        }

    except Exception as e:
        print(f"❌ Agent error: {e}")
        # Fallback to empty result
        return {
            **state,
            'relevance_check': {
                "is_relevant": False,
                "relevant_dataframes": [],
                "confidence_score": 0.0,
                "reasoning": f"Agent failed with error: {str(e)}",
                "agent_trace": []
            },
            'current_stage': 'relevance_checked'
        }


# ============================================================================
# Part 3: Updated Routing Logic (No Fixed Threshold)
# ============================================================================

def route_after_relevance_react(state: Dict[str, Any]) -> Literal["decompose_query", "handle_irrelevant"]:
    """
    Route based on ReAct agent's decision (NO FIXED THRESHOLD).

    The agent has already made an intelligent decision through multi-step reasoning.
    We simply check if relevant dataframes were found.
    """
    relevance = state['relevance_check']

    # Simple boolean check - agent already decided
    if relevance['relevant_dataframes'] and len(relevance['relevant_dataframes']) > 0:
        print(f"✅ Routing to decompose_query - Found {len(relevance['relevant_dataframes'])} relevant dataframes")
        return "decompose_query"
    else:
        print(f"❌ Routing to handle_irrelevant - No relevant dataframes found")
        return "handle_irrelevant"


# ============================================================================
# Part 4: Helper Functions (Need to be imported from your notebook)
# ============================================================================

# These need to be available in the notebook's global scope:
# - initialize_retriever()
# - initialize_cross_encoder(device)
# - get_dataframe_registry()
# - DataFrameRetriever class
# - DataFrameInfo class

def initialize_retriever():
    """Initialize retriever - should be imported from notebook"""
    global GLOBAL_RETRIEVER
    if GLOBAL_RETRIEVER is None:
        from your_notebook import get_dataframe_registry, DataFrameRetriever
        registry = get_dataframe_registry()
        GLOBAL_RETRIEVER = DataFrameRetriever()
        GLOBAL_RETRIEVER.index_dataframes(registry)
    return GLOBAL_RETRIEVER


def initialize_cross_encoder(device=None):
    """Initialize cross-encoder - should be imported from notebook"""
    global GLOBAL_CROSS_ENCODER
    if GLOBAL_CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        GLOBAL_CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
    return GLOBAL_CROSS_ENCODER


# Global variables
GLOBAL_RETRIEVER = None
GLOBAL_CROSS_ENCODER = None


# ============================================================================
# Part 5: Integration Instructions
# ============================================================================

"""
HOW TO INTEGRATE INTO YOUR NOTEBOOK:
=====================================

1. Add this cell after your existing check_relevance_node:

```python
# Import the ReAct agent implementation
from react_relevance_agent import (
    relevance_react_agent_node,
    route_after_relevance_react,
    retrieve_and_rank_dataframes,
    check_conversation_context,
    validate_dataframe_relevance
)

# Make sure global functions are available
import react_relevance_agent
react_relevance_agent.GLOBAL_RETRIEVER = GLOBAL_RETRIEVER
react_relevance_agent.GLOBAL_CROSS_ENCODER = GLOBAL_CROSS_ENCODER
react_relevance_agent.initialize_retriever = lambda: GLOBAL_RETRIEVER
react_relevance_agent.initialize_cross_encoder = lambda device=None: GLOBAL_CROSS_ENCODER
```

2. Update your workflow to use the ReAct agent:

```python
# OPTION A: Replace the old node completely
workflow.add_node("check_relevance", relevance_react_agent_node)  # NEW

# OPTION B: Add alongside for comparison
workflow.add_node("check_relevance_react", relevance_react_agent_node)  # NEW
workflow.add_node("check_relevance_original", check_relevance_node)  # OLD

# Use the new routing logic
workflow.add_conditional_edges(
    "check_relevance",
    route_after_relevance_react,  # NEW - No threshold!
    {
        "decompose_query": "decompose_query",
        "handle_irrelevant": "handle_irrelevant"
    }
)
```

3. Test both implementations side-by-side:

```python
# Test query
test_query = "عدد المعتمرين سنة 1441"

# Run with ReAct agent
result_react = app.invoke({
    'conversation_id': 'test',
    'question': test_query,
    'original_question': test_query,
    'model_name': 'deepseek-r1:32b',
    'message_history': [],
    # ... rest of state
})

print("ReAct Agent Decision:", result_react['relevance_check'])
print("\\nAgent Reasoning Trace:")
for i, msg in enumerate(result_react['relevance_check']['agent_trace']):
    print(f"Step {i+1}: {msg[:200]}...")
```

KEY BENEFITS:
=============
✅ NO fixed threshold (0.5) - agent decides adaptively
✅ Multi-step reasoning with tool use
✅ Better handling of edge cases
✅ Full reasoning trace for debugging
✅ Can add more tools easily (e.g., similarity search, metadata lookup)
✅ Scales to new dataframes without threshold tuning

COMPARISON:
===========
Old: confidence_score > 0.5 (fixed)
New: Agent reasoning → tool use → semantic validation → decision

The agent might say:
"The retrieval score is 0.48 but the sample values clearly match the query,
so I will include this dataframe despite the lower score."
"""
