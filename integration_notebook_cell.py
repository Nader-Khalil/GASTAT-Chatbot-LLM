"""
INTEGRATION CELL FOR NOTEBOOK
==============================
Copy this entire cell into your notebook to integrate the ReAct agent.
Add it AFTER your existing check_relevance_node definition.
"""

# ============================================================================
# STEP 1: Import ReAct Agent Components
# ============================================================================

import sys
sys.path.append('/home/user/GASTAT-Chatbot-LLM')

from react_relevance_agent import (
    relevance_react_agent_node,
    route_after_relevance_react,
    retrieve_and_rank_dataframes,
    check_conversation_context,
    validate_dataframe_relevance
)
import react_relevance_agent

# ============================================================================
# STEP 2: Connect Global Variables
# ============================================================================

# Make sure the ReAct agent can access your existing global variables
react_relevance_agent.GLOBAL_RETRIEVER = GLOBAL_RETRIEVER
react_relevance_agent.GLOBAL_CROSS_ENCODER = GLOBAL_CROSS_ENCODER

# Override the initialization functions
react_relevance_agent.initialize_retriever = lambda: GLOBAL_RETRIEVER
react_relevance_agent.initialize_cross_encoder = lambda device=None: GLOBAL_CROSS_ENCODER

print("✅ ReAct agent components imported successfully")

# ============================================================================
# STEP 3: Update Workflow (Choose Option A or B)
# ============================================================================

# OPTION A: Complete Replacement (Recommended for testing)
# ---------------------------------------------------------
def build_agent_workflow_with_react():
    """Build workflow using ReAct agent for relevance checking"""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("detect_language", detect_language_node)
    workflow.add_node("filter_columns", filter_columns_node)
    workflow.add_node("evaluate_context_relevance", evaluate_context_relevance_node)

    # 🔥 NEW: Use ReAct agent instead of old check_relevance
    workflow.add_node("check_relevance", relevance_react_agent_node)

    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("prepare_current_dataframe", prepare_current_dataframe_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_current_code", execute_current_code_node)
    workflow.add_node("collect_result", collect_result_node)
    workflow.add_node("format_combined_response", format_combined_response_node)
    workflow.add_node("handle_irrelevant", handle_irrelevant_node)

    # Build edges
    workflow.add_edge(START, "detect_language")
    workflow.add_edge("detect_language", "filter_columns")
    workflow.add_edge("filter_columns", "evaluate_context_relevance")
    workflow.add_edge("evaluate_context_relevance", "check_relevance")

    # 🔥 NEW: Use ReAct routing (no threshold!)
    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance_react,
        {
            "decompose_query": "decompose_query",
            "handle_irrelevant": "handle_irrelevant"
        }
    )

    # Rest of the workflow (unchanged)
    workflow.add_edge("decompose_query", "prepare_current_dataframe")
    workflow.add_edge("prepare_current_dataframe", "generate_code")
    workflow.add_edge("generate_code", "execute_current_code")
    workflow.add_edge("execute_current_code", "collect_result")

    workflow.add_conditional_edges(
        "collect_result",
        route_after_collect,
        {
            "prepare_current_dataframe": "prepare_current_dataframe",
            "format_combined_response": "format_combined_response"
        }
    )

    workflow.add_edge("format_combined_response", END)
    workflow.add_edge("handle_irrelevant", END)

    return workflow.compile()


# OPTION B: Side-by-side Comparison
# ----------------------------------
def build_comparison_workflow():
    """Build workflow with BOTH implementations for testing"""
    workflow = StateGraph(AgentState)

    # Add both relevance checking nodes
    workflow.add_node("check_relevance_original", check_relevance_node)
    workflow.add_node("check_relevance_react", relevance_react_agent_node)

    # ... add other nodes ...

    # You can manually switch between them for testing
    return workflow


# ============================================================================
# STEP 4: Create New Agent Instance
# ============================================================================

# Rebuild the agent with ReAct implementation
app_with_react = build_agent_workflow_with_react()

print("✅ ReAct agent workflow compiled successfully")
print("   Use 'app_with_react' to test the new implementation")
print("   Your original 'app' is still available for comparison")


# ============================================================================
# STEP 5: Helper Function for Testing
# ============================================================================

def test_relevance_comparison(question, message_history=None):
    """
    Test both implementations side-by-side and compare results.

    Args:
        question: User question to test
        message_history: Optional conversation history

    Returns:
        Dictionary with comparison results
    """
    import time

    message_history = message_history or []

    base_state = {
        'conversation_id': f'test_{int(time.time())}',
        'question': question,
        'original_question': question,
        'model_name': 'deepseek-r1:32b',
        'message_history': message_history,
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
    }

    print("=" * 80)
    print(f"TESTING QUESTION: {question}")
    print("=" * 80)

    # Test Original Implementation
    print("\n🔵 ORIGINAL IMPLEMENTATION (with threshold 0.5)")
    print("-" * 80)
    start_old = time.time()
    try:
        result_old = app.invoke(base_state.copy())
        time_old = time.time() - start_old

        relevance_old = result_old.get('relevance_check', {})
        print(f"✅ Completed in {time_old:.2f}s")
        print(f"   Is Relevant: {relevance_old.get('is_relevant', False)}")
        print(f"   Confidence: {relevance_old.get('confidence_score', 0):.3f}")
        print(f"   Dataframes: {relevance_old.get('relevant_dataframes', [])}")
        print(f"   Reasoning: {relevance_old.get('reasoning', 'N/A')[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        result_old = None
        time_old = None

    # Test ReAct Implementation
    print("\n🟢 REACT AGENT IMPLEMENTATION (no threshold)")
    print("-" * 80)
    start_react = time.time()
    try:
        result_react = app_with_react.invoke(base_state.copy())
        time_react = time.time() - start_react

        relevance_react = result_react.get('relevance_check', {})
        print(f"✅ Completed in {time_react:.2f}s")
        print(f"   Is Relevant: {relevance_react.get('is_relevant', False)}")
        print(f"   Confidence: {relevance_react.get('confidence_score', 0):.3f}")
        print(f"   Dataframes: {relevance_react.get('relevant_dataframes', [])}")
        print(f"   Reasoning: {relevance_react.get('reasoning', 'N/A')[:200]}...")

        # Show agent trace
        if 'agent_trace' in relevance_react:
            print(f"\n   🤖 Agent Reasoning Trace ({len(relevance_react['agent_trace'])} steps):")
            for i, step in enumerate(relevance_react['agent_trace'][:3], 1):  # Show first 3 steps
                print(f"      Step {i}: {step[:150]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        result_react = None
        time_react = None

    # Comparison
    print("\n📊 COMPARISON")
    print("-" * 80)
    if result_old and result_react:
        dfs_old = set(result_old['relevance_check'].get('relevant_dataframes', []))
        dfs_react = set(result_react['relevance_check'].get('relevant_dataframes', []))

        common = dfs_old & dfs_react
        only_old = dfs_old - dfs_react
        only_react = dfs_react - dfs_old

        print(f"   Common dataframes: {len(common)} - {list(common)}")
        print(f"   Only in Original: {len(only_old)} - {list(only_old)}")
        print(f"   Only in ReAct: {len(only_react)} - {list(only_react)}")
        print(f"   Time difference: {abs(time_react - time_old):.2f}s")

        if dfs_old == dfs_react:
            print("   ✅ Both implementations agree!")
        else:
            print("   ⚠️  Implementations disagree - review reasoning")

    print("=" * 80 + "\n")

    return {
        'question': question,
        'original': result_old,
        'react': result_react,
        'time_old': time_old,
        'time_react': time_react
    }


print("\n" + "=" * 80)
print("INTEGRATION COMPLETE!")
print("=" * 80)
print("\nYou can now test using:")
print("  1. test_relevance_comparison('عدد المعتمرين سنة 1441')")
print("  2. test_relevance_comparison('What is the population growth rate?')")
print("  3. app_with_react.invoke({...})  # Direct usage")
print("\n" + "=" * 80)
